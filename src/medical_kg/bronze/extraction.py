from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass

from medical_kg.config import AppSettings
from medical_kg.db.repository import ClaimedJob, ExtractionRunSpec, KnowledgeRepository
from medical_kg.landing.chunking import DocumentChunk, chunk_document
from medical_kg.llm.base import LLMClient, LLMResponse
from medical_kg.models.assertion import AssertionOutput, ExtractionOutput
from medical_kg.prompts import PromptRegistry

logger = logging.getLogger(__name__)


@dataclass
class RunStatistics:
    documents_processed: int = 0
    documents_successful: int = 0
    documents_failed: int = 0
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost: float | None = None

    def add(self, other: RunStatistics) -> None:
        self.documents_processed += other.documents_processed
        self.documents_successful += other.documents_successful
        self.documents_failed += other.documents_failed
        self.requests += other.requests
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens


class MinuteRateLimiter:
    """Conservative local request/token limiter for one process."""

    def __init__(self, requests_per_minute: int, tokens_per_minute: int) -> None:
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self._events: list[tuple[float, int]] = []
        self._lock = asyncio.Lock()

    async def acquire(self, estimated_tokens: int) -> None:
        # A single full document may exceed the configured minute budget. Admit one such
        # request into an otherwise empty window instead of waiting forever.
        estimated_tokens = min(estimated_tokens, self.tokens_per_minute)
        while True:
            async with self._lock:
                now = time.monotonic()
                self._events = [(at, tokens) for at, tokens in self._events if now - at < 60]
                requests = len(self._events)
                tokens = sum(item[1] for item in self._events)
                if (
                    requests < self.requests_per_minute
                    and tokens + estimated_tokens <= self.tokens_per_minute
                ):
                    self._events.append((now, estimated_tokens))
                    return
                delay = max(0.05, 60 - (now - self._events[0][0])) if self._events else 0.05
            await asyncio.sleep(delay)


class BronzeExtractor:
    def __init__(
        self,
        *,
        settings: AppSettings,
        repository: KnowledgeRepository,
        llm: LLMClient,
        prompts: PromptRegistry,
    ) -> None:
        self.settings = settings
        self.repository = repository
        self.llm = llm
        self.prompts = prompts
        self.limiter = MinuteRateLimiter(
            settings.processing.requests_per_minute, settings.processing.tokens_per_minute
        )

    @property
    def job_stages(self) -> list[str]:
        return [f"extract:{pass_name}" for pass_name in self.settings.extraction.passes]

    def stage_version(self, chunk_size: int | None, chunk_overlap: int) -> str:
        # Validate the options before jobs are enqueued. A chunked run is a distinct resumable
        # processing unit from a full-document run.
        chunk_document("", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if chunk_size is None:
            return self.settings.extraction.stage_version
        version = (
            f"{self.settings.extraction.stage_version}|chunk={chunk_size},overlap={chunk_overlap}"
        )
        if len(version) > 128:
            raise ValueError("effective extraction stage version exceeds 128 characters")
        return version

    async def enqueue(
        self,
        document_ids: list[str],
        *,
        chunk_size: int | None = None,
        chunk_overlap: int = 0,
    ) -> int:
        return await self.repository.enqueue_jobs(
            document_ids=document_ids,
            stages=self.job_stages,
            stage_version=self.stage_version(chunk_size, chunk_overlap),
        )

    async def run(
        self,
        *,
        worker_id: str,
        document_id: str | None = None,
        chunk_size: int | None = None,
        chunk_overlap: int = 0,
    ) -> RunStatistics:
        stage_version = self.stage_version(chunk_size, chunk_overlap)
        workers = [
            asyncio.create_task(
                self._worker(
                    f"{worker_id}-{index}",
                    document_id,
                    stage_version,
                    chunk_size,
                    chunk_overlap,
                )
            )
            for index in range(self.settings.processing.max_concurrency)
        ]
        results = await asyncio.gather(*workers)
        total = RunStatistics()
        for result in results:
            total.add(result)
        return total

    async def _worker(
        self,
        worker_id: str,
        document_id: str | None,
        stage_version: str,
        chunk_size: int | None,
        chunk_overlap: int,
    ) -> RunStatistics:
        statistics = RunStatistics()
        while True:
            job = await self.repository.claim_job(
                stages=self.job_stages,
                stage_version=stage_version,
                worker_id=worker_id,
                document_id=document_id,
            )
            if job is None:
                return statistics
            outcome = await self._process(job, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            statistics.add(outcome)

    async def _process(
        self,
        job: ClaimedJob,
        *,
        chunk_size: int | None = None,
        chunk_overlap: int = 0,
    ) -> RunStatistics:
        statistics = RunStatistics(documents_processed=1)
        pass_name = job.stage.split(":", 1)[1]
        prompt = self.prompts.extraction(pass_name)
        started = time.monotonic()
        try:
            chunks = chunk_document(job.content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            responses: list[tuple[DocumentChunk, LLMResponse]] = []
            for chunk in chunks:
                response = await self._request(
                    chunk.content,
                    prompt.system_prompt,
                    prompt.render(pass_name=pass_name, document=chunk.content),
                    statistics,
                )
                responses.append((chunk, response))

            output = ExtractionOutput(
                assertions=self._unique_assertions(
                    assertion
                    for _, response in responses
                    for assertion in response.output.assertions
                )
            )
            if chunk_size is None:
                raw_output = responses[0][1].raw_output
            else:
                raw_output = {
                    "chunking": {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
                    "chunks": [
                        {
                            "index": chunk.index,
                            "character_start": chunk.character_start,
                            "character_end": chunk.character_end,
                            "raw_output": response.raw_output,
                        }
                        for chunk, response in responses
                    ],
                }
            run_id = await self.repository.complete_extraction(
                job=job,
                run_spec=ExtractionRunSpec(
                    model_provider=self.settings.llm.provider,
                    model_name=self.settings.llm.model,
                    prompt_name=prompt.name,
                    prompt_version=prompt.version,
                    pass_name=pass_name,
                    temperature=self.settings.llm.temperature,
                    code_version=self.settings.extraction.code_version,
                ),
                output=output,
                raw_output=raw_output,
            )
            statistics.documents_successful = 1
            statistics.input_tokens += sum(response.input_tokens for _, response in responses)
            statistics.output_tokens += sum(response.output_tokens for _, response in responses)
            logger.info(
                "extraction succeeded",
                extra={
                    "document_id": job.document_id,
                    "stage": job.stage,
                    "extraction_run_id": run_id,
                    "model": self.settings.llm.model,
                    "elapsed_time": time.monotonic() - started,
                    "input_tokens": statistics.input_tokens,
                    "output_tokens": statistics.output_tokens,
                    "chunks": len(chunks),
                    "status": "SUCCESS",
                },
            )
        except Exception as error:
            await self.repository.fail_job(job.job_id, f"{type(error).__name__}: {error}")
            statistics.documents_failed = 1
            logger.exception(
                "extraction failed",
                extra={
                    "document_id": job.document_id,
                    "stage": job.stage,
                    "model": self.settings.llm.model,
                    "elapsed_time": time.monotonic() - started,
                    "status": "FAILED",
                    "error": str(error),
                },
            )
        return statistics

    async def _request(
        self,
        content: str,
        system_prompt: str,
        user_prompt: str,
        statistics: RunStatistics,
    ) -> LLMResponse:
        estimated_tokens = max(1, len(content) // 4)
        attempts = self.settings.processing.max_retries + 1
        for attempt_index in range(attempts):
            try:
                statistics.requests += 1
                await self.limiter.acquire(estimated_tokens)
                return await asyncio.wait_for(
                    self.llm.extract_document(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=self.settings.llm.temperature,
                    ),
                    timeout=self.settings.processing.request_timeout,
                )
            except Exception:
                if attempt_index + 1 >= attempts:
                    raise
                delay = min(self.settings.processing.retry_backoff * (2**attempt_index), 60)
                await asyncio.sleep(delay)
        raise RuntimeError("Extraction retry loop ended unexpectedly")

    @staticmethod
    def _unique_assertions(assertions: Iterable[AssertionOutput]) -> list[AssertionOutput]:
        unique: list[AssertionOutput] = []
        seen: set[str] = set()
        for assertion in assertions:
            identity = json.dumps(
                assertion.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if identity not in seen:
                seen.add(identity)
                unique.append(assertion)
        return unique
