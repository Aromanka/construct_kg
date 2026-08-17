from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections.abc import Awaitable, Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass
from email.utils import parsedate_to_datetime

import httpx

from medical_kg.config import AppSettings
from medical_kg.db.repository import (
    ClaimedJob,
    ExtractionRunSpec,
    KnowledgeRepository,
    StoredChunkResult,
)
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


class SmoothRateLimiter:
    """Smooth request/token reservations for the single-process fallback path."""

    def __init__(self, requests_per_minute: int, tokens_per_minute: int) -> None:
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute
        self._next_request_at = 0.0
        self._next_token_at = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self, estimated_tokens: int) -> None:
        estimated_tokens = min(estimated_tokens, self.tokens_per_minute)
        async with self._lock:
            now = time.monotonic()
            scheduled_at = max(now, self._next_request_at, self._next_token_at)
            self._next_request_at = scheduled_at + 60 / self.requests_per_minute
            self._next_token_at = scheduled_at + 60 * estimated_tokens / self.tokens_per_minute
            delay = scheduled_at - now
        if delay > 0:
            await asyncio.sleep(delay)

    async def reconcile(self, estimated_tokens: int, actual_tokens: int) -> None:
        difference = actual_tokens - estimated_tokens
        if difference == 0:
            return
        async with self._lock:
            now = time.monotonic()
            self._next_token_at = max(
                now,
                self._next_token_at + 60 * difference / self.tokens_per_minute,
            )


@dataclass
class _ScheduledRequest:
    execute: Callable[[], Awaitable[LLMResponse]]
    future: asyncio.Future[LLMResponse]


class APIRequestScheduler:
    """Bounded queue whose worker count is the hard API concurrency limit."""

    def __init__(self, concurrency: int, queue_size: int) -> None:
        self.concurrency = concurrency
        self.queue: asyncio.Queue[_ScheduledRequest | None] = asyncio.Queue(queue_size)
        self.workers: list[asyncio.Task[None]] = []

    @property
    def running(self) -> bool:
        return bool(self.workers)

    async def start(self) -> None:
        if self.workers:
            return
        self.workers = [asyncio.create_task(self._worker()) for _ in range(self.concurrency)]

    async def submit(self, execute: Callable[[], Awaitable[LLMResponse]]) -> LLMResponse:
        if not self.workers:
            raise RuntimeError("API request scheduler is not running")
        future: asyncio.Future[LLMResponse] = asyncio.get_running_loop().create_future()
        await self.queue.put(_ScheduledRequest(execute=execute, future=future))
        return await future

    async def close(self) -> None:
        if not self.workers:
            return
        for _ in self.workers:
            await self.queue.put(None)
        await asyncio.gather(*self.workers)
        self.workers = []

    async def _worker(self) -> None:
        while True:
            item = await self.queue.get()
            try:
                if item is None:
                    return
                if item.future.cancelled():
                    continue
                try:
                    response = await item.execute()
                except Exception as error:
                    if not item.future.done():
                        item.future.set_exception(error)
                else:
                    if not item.future.done():
                        item.future.set_result(response)
            finally:
                self.queue.task_done()


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
        self.limiter = SmoothRateLimiter(
            settings.processing.requests_per_minute, settings.processing.tokens_per_minute
        )
        api_concurrency = getattr(settings.processing, "api_concurrency", 100)
        queue_size = getattr(settings.processing, "chunk_queue_size", api_concurrency * 3)
        self.scheduler = APIRequestScheduler(api_concurrency, queue_size)

    @property
    def job_stages(self) -> list[str]:
        return [f"extract:{pass_name}" for pass_name in self.settings.extraction.passes]

    def stage_version(self, chunk_size: int | None, chunk_overlap: int) -> str:
        # Validate the options before jobs are enqueued. A chunked run is a distinct resumable
        # processing unit from a full-document run.
        chunk_document("", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        prompt_versions = ",".join(
            f"{pass_name}:{self.prompts.extraction(pass_name).version}"
            for pass_name in sorted(self.settings.extraction.passes)
        )
        version = f"{self.settings.extraction.stage_version}|prompts={prompt_versions}"
        if chunk_size is not None:
            version += f"|chunk={chunk_size},overlap={chunk_overlap}"
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
        progress: Callable[[RunStatistics], None] | None = None,
    ) -> RunStatistics:
        stage_version = self.stage_version(chunk_size, chunk_overlap)
        job_concurrency = getattr(self.settings.processing, "job_concurrency", 100)
        job_claimers = min(
            getattr(self.settings.processing, "job_claimers", 8), job_concurrency
        )
        job_queue: asyncio.Queue[ClaimedJob | None] = asyncio.Queue(job_concurrency * 2)
        available_job_slots = asyncio.Semaphore(job_concurrency)
        await self.scheduler.start()
        consumers = [
            asyncio.create_task(
                self._job_consumer(
                    job_queue,
                    available_job_slots,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    progress=progress,
                )
            )
            for _ in range(job_concurrency)
        ]
        claimers = [
            asyncio.create_task(
                self._claim_jobs(
                    job_queue,
                    worker_id=f"{worker_id}-{index}",
                    document_id=document_id,
                    stage_version=stage_version,
                    available_job_slots=available_job_slots,
                )
            )
            for index in range(job_claimers)
        ]
        try:
            await asyncio.gather(*claimers)
            for _ in consumers:
                await job_queue.put(None)
            results = await asyncio.gather(*consumers)
        finally:
            for task in claimers + consumers:
                if not task.done():
                    task.cancel()
            await self.scheduler.close()
        total = RunStatistics()
        for result in results:
            total.add(result)
        return total

    async def _claim_jobs(
        self,
        queue: asyncio.Queue[ClaimedJob | None],
        worker_id: str,
        document_id: str | None,
        stage_version: str,
        available_job_slots: asyncio.Semaphore,
    ) -> None:
        while True:
            await available_job_slots.acquire()
            try:
                job = await self.repository.claim_job(
                    stages=self.job_stages,
                    stage_version=stage_version,
                    worker_id=worker_id,
                    document_id=document_id,
                    lease_seconds=getattr(self.settings.processing, "job_lease_seconds", 900.0),
                )
            except BaseException:
                available_job_slots.release()
                raise
            if job is None:
                available_job_slots.release()
                return
            await queue.put(job)

    async def _job_consumer(
        self,
        queue: asyncio.Queue[ClaimedJob | None],
        available_job_slots: asyncio.Semaphore,
        *,
        chunk_size: int | None,
        chunk_overlap: int,
        progress: Callable[[RunStatistics], None] | None,
    ) -> RunStatistics:
        statistics = RunStatistics()
        while True:
            job = await queue.get()
            try:
                if job is None:
                    return statistics
                outcome = await self._process_with_heartbeat(
                    job, chunk_size=chunk_size, chunk_overlap=chunk_overlap
                )
            finally:
                if job is not None:
                    # A claimer can reserve another SQLite job only after this one has
                    # completed, so queued jobs never sit without a heartbeat lease.
                    available_job_slots.release()
                queue.task_done()
            statistics.add(outcome)
            if progress is not None:
                progress(outcome)

    async def _process_with_heartbeat(
        self,
        job: ClaimedJob,
        *,
        chunk_size: int | None,
        chunk_overlap: int,
    ) -> RunStatistics:
        heartbeat: asyncio.Task[None] | None = None
        if job.worker_id:
            heartbeat = asyncio.create_task(self._heartbeat(job))
        try:
            return await self._process(
                job, chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
        finally:
            if heartbeat is not None:
                heartbeat.cancel()
                with suppress(asyncio.CancelledError):
                    await heartbeat

    async def _heartbeat(self, job: ClaimedJob) -> None:
        interval = getattr(self.settings.processing, "heartbeat_interval", 30.0)
        lease_seconds = getattr(self.settings.processing, "job_lease_seconds", 900.0)
        while True:
            await asyncio.sleep(interval)
            alive = await self.repository.heartbeat_job(
                job.job_id, worker_id=job.worker_id, lease_seconds=lease_seconds
            )
            if not alive:
                logger.warning(
                    "job lease was lost",
                    extra={"job_id": str(job.job_id), "worker_id": job.worker_id},
                )
                return

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
        owns_scheduler = not self.scheduler.running
        if owns_scheduler:
            await self.scheduler.start()
        try:
            chunks = chunk_document(job.content, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            stored = await self.repository.prepare_chunks(
                job=job, pass_name=pass_name, chunks=chunks
            )
            pending = [
                asyncio.create_task(
                    self._process_chunk(
                        job=job,
                        chunk=chunk,
                        system_prompt=prompt.system_prompt,
                        user_prompt=prompt.render(pass_name=pass_name, document=chunk.content),
                        stored=stored.get(chunk.index),
                        statistics=statistics,
                    )
                )
                for chunk in chunks
            ]
            outcomes = await asyncio.gather(*pending, return_exceptions=True)
            errors = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
            if errors:
                raise errors[0]
            responses = [
                (chunk, outcome)
                for chunk, outcome in zip(chunks, outcomes, strict=True)
                if isinstance(outcome, LLMResponse)
            ]

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
                            "model": {
                                "provider": response.metadata.get("model_provider"),
                                "name": response.metadata.get("model"),
                                "temperature": response.metadata.get("temperature"),
                                "legacy_metadata": response.metadata.get(
                                    "legacy_model_metadata", False
                                ),
                            },
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
        finally:
            if owns_scheduler:
                await self.scheduler.close()
        return statistics

    async def _process_chunk(
        self,
        *,
        job: ClaimedJob,
        chunk: DocumentChunk,
        system_prompt: str,
        user_prompt: str,
        stored: StoredChunkResult | None,
        statistics: RunStatistics,
    ) -> LLMResponse:
        if stored is not None:
            return LLMResponse(
                output=ExtractionOutput.model_validate(stored.validated_output),
                raw_output=stored.raw_output,
                input_tokens=stored.input_tokens,
                output_tokens=stored.output_tokens,
                metadata={
                    "checkpoint": True,
                    "model_provider": stored.model_provider,
                    "model": stored.model_name,
                    "temperature": stored.temperature,
                    "legacy_model_metadata": stored.model_name is None,
                },
            )
        await self.repository.start_chunk(job.job_id, chunk.index, job.worker_id)

        async def execute() -> LLMResponse:
            return await self._request(
                chunk.content, system_prompt, user_prompt, statistics
            )

        try:
            response = await self.scheduler.submit(execute)
            model_provider = str(self.settings.llm.provider)
            model_name = str(response.metadata.get("model") or self.settings.llm.model)
            temperature = float(self.settings.llm.temperature)
            await self.repository.complete_chunk(
                job.job_id,
                chunk.index,
                response,
                model_provider=model_provider,
                model_name=model_name,
                temperature=temperature,
            )
            response.metadata.update(
                {
                    "model_provider": model_provider,
                    "model": model_name,
                    "temperature": temperature,
                }
            )
            return response
        except Exception as error:
            await self.repository.fail_chunk(
                job.job_id, chunk.index, f"{type(error).__name__}: {error}"
            )
            raise

    async def _request(
        self,
        content: str,
        system_prompt: str,
        user_prompt: str,
        statistics: RunStatistics,
    ) -> LLMResponse:
        del content
        estimated_tokens = self._estimate_tokens(system_prompt, user_prompt)
        attempts = self.settings.processing.max_retries + 1
        for attempt_index in range(attempts):
            try:
                statistics.requests += 1
                await self._acquire_capacity(estimated_tokens)
                response = await asyncio.wait_for(
                    self.llm.extract_document(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=self.settings.llm.temperature,
                    ),
                    timeout=(
                        self.settings.processing.request_timeout
                        * max(1, int(getattr(type(self.llm), "max_request_count", 1)))
                    ),
                )
                statistics.requests += max(
                    0, int(response.metadata.get("request_count", 1)) - 1
                )
                await self._reconcile_tokens(
                    estimated_tokens, response.input_tokens + response.output_tokens
                )
                return response
            except Exception as error:
                request_count = int(getattr(error, "request_count", 1))
                statistics.requests += max(0, request_count - 1)
                actual_tokens = int(getattr(error, "input_tokens", 0)) + int(
                    getattr(error, "output_tokens", 0)
                )
                if actual_tokens > 0:
                    await self._reconcile_tokens(estimated_tokens, actual_tokens)
                if attempt_index + 1 >= attempts or not self._is_retryable(error):
                    raise
                retry_after = self._retry_after(error)
                base_delay = min(
                    self.settings.processing.retry_backoff * (2**attempt_index), 60
                )
                delay = max(retry_after, base_delay + random.uniform(0, base_delay * 0.25))
                await asyncio.sleep(delay)
        raise RuntimeError("Extraction retry loop ended unexpectedly")

    def _estimate_tokens(self, system_prompt: str, user_prompt: str) -> int:
        text = system_prompt + user_prompt
        non_ascii = sum(ord(character) > 127 for character in text)
        ascii_characters = len(text) - non_ascii
        prompt_tokens = non_ascii + (ascii_characters + 3) // 4
        reserved = getattr(self.settings.processing, "reserved_output_tokens", 4096)
        return max(1, prompt_tokens + reserved)

    @property
    def _limiter_key(self) -> str:
        provider = getattr(self.settings.llm, "provider", "compatible")
        model = getattr(self.settings.llm, "model", "unknown")
        return f"{provider}:{model}"

    async def _acquire_capacity(self, estimated_tokens: int) -> None:
        if getattr(self.settings.processing, "distributed_rate_limit", False):
            delay = await self.repository.reserve_api_capacity(
                limiter_key=self._limiter_key,
                estimated_tokens=estimated_tokens,
                requests_per_minute=self.settings.processing.requests_per_minute,
                tokens_per_minute=self.settings.processing.tokens_per_minute,
            )
            if delay > 0:
                await asyncio.sleep(delay)
            return
        await self.limiter.acquire(estimated_tokens)

    async def _reconcile_tokens(self, estimated_tokens: int, actual_tokens: int) -> None:
        if actual_tokens <= 0:
            return
        if getattr(self.settings.processing, "distributed_rate_limit", False):
            await self.repository.reconcile_api_tokens(
                limiter_key=self._limiter_key,
                estimated_tokens=estimated_tokens,
                actual_tokens=actual_tokens,
                tokens_per_minute=self.settings.processing.tokens_per_minute,
            )
            return
        await self.limiter.reconcile(estimated_tokens, actual_tokens)

    @staticmethod
    def _is_retryable(error: Exception) -> bool:
        if isinstance(error, (asyncio.TimeoutError, httpx.TransportError)):
            return True
        if isinstance(error, httpx.HTTPStatusError):
            return error.response.status_code in {408, 429} or error.response.status_code >= 500
        return False

    @staticmethod
    def _retry_after(error: Exception) -> float:
        if not isinstance(error, httpx.HTTPStatusError):
            return 0.0
        value = error.response.headers.get("Retry-After")
        if not value:
            return 0.0
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                parsed = parsedate_to_datetime(value)
                return max(0.0, parsed.timestamp() - time.time())
            except (TypeError, ValueError, OverflowError):
                return 0.0

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
