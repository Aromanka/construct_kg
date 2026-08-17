from __future__ import annotations

import hashlib
import json
import uuid
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert

from medical_kg.db.models import (
    Assertion,
    AssertionEvidence,
    Document,
    Entity,
    EntityAlias,
    EntityExternalId,
    EntityMention,
    EntityResolution,
    RawAssertion,
    RelationType,
)
from medical_kg.db.repository import KnowledgeRepository
from medical_kg.llm.base import LLMClient
from medical_kg.prompts import PromptRegistry
from medical_kg.silver.deduplication import (
    canonical_qualifiers,
    normalized_assertion_identity,
)
from medical_kg.silver.entity_resolution import (
    CandidateRetriever,
    ConservativeEntityResolver,
    EntityCandidate,
    ResolutionDecision,
    normalized_alias,
    preferred_canonical_name,
)
from medical_kg.silver.relation_normalization import ExactRelationNormalizer


@dataclass(frozen=True)
class CanonicalizationResult:
    mentions_considered: int
    mentions_resolved: int
    entities_created: int
    aliases_created: int
    exact_or_synonym_matches: int
    semantic_matches: int
    raw_assertions_considered: int
    canonical_assertions_created: int
    duplicate_assertions_aggregated: int
    evidence_links_created: int
    relations_other: int


@dataclass
class _EntityRecord:
    entity_id: str
    canonical_name: str
    entity_type: str
    sources: list[dict[str, Any]]


@dataclass(frozen=True)
class _ResolutionRecord:
    mention_id: uuid.UUID
    entity_id: str
    method: str
    confidence: float
    candidates: list[dict[str, Any]]


@dataclass(frozen=True)
class _Fact:
    identity: str
    subject_entity_id: str
    object_entity_id: str
    relation_name: str
    qualifiers: dict[str, Any]
    negated: bool
    speculative: bool
    raw_assertions: tuple[RawAssertion, ...]
    sources: list[dict[str, Any]]


def _merge_sources(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for group in groups:
        for source in group:
            key = json.dumps(source, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            unique[key] = source
    return [unique[key] for key in sorted(unique)]


def _entity_id(entity_type: str, canonical_name: str) -> str:
    identity = f"{entity_type}\0{normalized_alias(canonical_name)}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20].upper()
    return f"ENT_{digest}"


class CanonicalizationPipeline:
    """Materialize conservative Silver entities and evidence-aggregated Gold facts."""

    def __init__(
        self,
        *,
        repository: KnowledgeRepository,
        vocabulary: list[str],
        prompts: PromptRegistry,
        llm: LLMClient | None = None,
        semantic: bool = False,
        confidence_threshold: float = 0.85,
        candidate_top_k: int = 8,
    ) -> None:
        self.repository = repository
        self.vocabulary = vocabulary
        self.prompts = prompts
        self.llm = llm
        self.semantic = semantic
        self.confidence_threshold = confidence_threshold
        self.retriever = CandidateRetriever(top_k=candidate_top_k)
        self.resolver = ConservativeEntityResolver()
        self.relation_normalizer = ExactRelationNormalizer(vocabulary)

    async def run(self, *, document_id: str | None = None) -> CanonicalizationResult:
        await self.repository.create_schema()
        await self.repository.seed_relations(self.vocabulary)
        snapshot = await self._load_snapshot(document_id=document_id)
        raw_assertions: list[RawAssertion] = snapshot["raw_assertions"]
        mentions: dict[uuid.UUID, EntityMention] = snapshot["mentions"]
        documents: dict[str, Document] = snapshot["documents"]
        existing_resolutions: dict[uuid.UUID, str] = snapshot["resolutions"]

        entities: dict[str, _EntityRecord] = {
            item.entity_id: _EntityRecord(
                item.entity_id, item.canonical_name, item.entity_type, list(item.sources)
            )
            for item in snapshot["entities"]
        }
        aliases: defaultdict[str, set[str]] = defaultdict(set)
        for alias in snapshot["aliases"]:
            aliases[alias.entity_id].add(alias.alias)
        external_ids: defaultdict[str, list[str]] = defaultdict(list)
        for mapping in snapshot["external_ids"]:
            external_ids[mapping.entity_id].append(
                f"{mapping.namespace}:{mapping.accession}"
            )

        mention_context: dict[uuid.UUID, RawAssertion] = {}
        for raw in raw_assertions:
            mention_context.setdefault(raw.subject_mention_id, raw)
            mention_context.setdefault(raw.object_mention_id, raw)

        planned_entities: dict[str, _EntityRecord] = {}
        planned_aliases: set[tuple[str, str, str, float]] = set()
        planned_resolutions: list[_ResolutionRecord] = []
        resolution_map = dict(existing_resolutions)
        deterministic_matches = 0
        semantic_matches = 0

        for mention_id in sorted(mention_context, key=str):
            if mention_id in resolution_map:
                continue
            mention = mentions[mention_id]
            candidates = self._candidates(
                entities, aliases, external_ids, entity_type=mention.entity_type
            )
            retrieved = self.retriever.retrieve(
                mention.mention_text, mention.entity_type, candidates
            )
            decision = self.resolver.resolve_decision(
                mention.mention_text, mention.entity_type, candidates
            )
            canonical_name = preferred_canonical_name(
                mention.mention_text, mention.entity_type
            )
            if decision.entity_id is not None:
                deterministic_matches += 1
            elif self.semantic and retrieved:
                decision, canonical_name = await self._semantic_entity_decision(
                    mention=mention,
                    raw=mention_context[mention_id],
                    document=documents.get(mention.document_id),
                    retrieved=retrieved,
                )
                if decision.entity_id is not None:
                    semantic_matches += 1

            if decision.entity_id is None:
                creation_method = (
                    decision.method
                    if decision.method.startswith("semantic_")
                    else "new_entity"
                )
                creation_confidence = (
                    decision.confidence
                    if decision.method.startswith("semantic_")
                    else 1.0
                )
                entity_id = _entity_id(mention.entity_type, canonical_name)
                if entity_id not in entities:
                    record = _EntityRecord(
                        entity_id=entity_id,
                        canonical_name=canonical_name,
                        entity_type=mention.entity_type,
                        sources=list(mention.sources),
                    )
                    entities[entity_id] = record
                    planned_entities[entity_id] = record
                else:
                    entities[entity_id].sources = _merge_sources(
                        entities[entity_id].sources, list(mention.sources)
                    )
                decision = ResolutionDecision(
                    entity_id, creation_method, creation_confidence
                )

            entity_id = decision.entity_id
            if entity_id is None:
                raise RuntimeError(f"Unable to resolve mention {mention_id}")
            resolution_map[mention_id] = entity_id
            aliases[entity_id].add(mention.mention_text)
            planned_aliases.add(
                (entity_id, mention.mention_text, decision.method, decision.confidence)
            )
            planned_resolutions.append(
                _ResolutionRecord(
                    mention_id=mention_id,
                    entity_id=entity_id,
                    method=decision.method,
                    confidence=decision.confidence,
                    candidates=[
                        {
                            "entity_id": item.candidate.entity_id,
                            "canonical_name": item.candidate.canonical_name,
                            "entity_type": item.candidate.entity_type,
                            "aliases": list(item.candidate.aliases),
                            "external_ids": list(item.candidate.external_ids),
                            "retrieval_score": round(item.score, 6),
                            "retrieval_reason": item.reason,
                        }
                        for item in retrieved
                    ],
                )
            )

        grouped: defaultdict[str, list[RawAssertion]] = defaultdict(list)
        fact_values: dict[str, tuple[str, str, str, dict[str, Any], bool, bool]] = {}
        relations_other = 0
        for raw in raw_assertions:
            subject_id = resolution_map.get(raw.subject_mention_id)
            object_id = resolution_map.get(raw.object_mention_id)
            if not subject_id or not object_id:
                continue
            relation = self.relation_normalizer.normalize(raw.detailed_relation)
            if relation.canonical_relation.casefold() == "other" and self.semantic:
                relation = await self._semantic_relation_decision(raw)
            if relation.canonical_relation.casefold() == "other":
                relations_other += 1
            qualifiers = canonical_qualifiers(raw.qualifiers)
            identity = normalized_assertion_identity(
                subject_entity_id=subject_id,
                canonical_relation_id=relation.canonical_relation,
                object_entity_id=object_id,
                qualifiers=qualifiers,
                negated=raw.negated,
                speculative=raw.speculative,
            )
            grouped[identity].append(raw)
            fact_values[identity] = (
                subject_id,
                object_id,
                relation.canonical_relation,
                qualifiers,
                raw.negated,
                raw.speculative,
            )

        facts: list[_Fact] = []
        for identity, support in grouped.items():
            subject_id, object_id, relation_name, qualifiers, negated, speculative = (
                fact_values[identity]
            )
            facts.append(
                _Fact(
                    identity=identity,
                    subject_entity_id=subject_id,
                    object_entity_id=object_id,
                    relation_name=relation_name,
                    qualifiers=qualifiers,
                    negated=negated,
                    speculative=speculative,
                    raw_assertions=tuple(support),
                    sources=_merge_sources(*(list(raw.sources) for raw in support)),
                )
            )

        created_entities, created_aliases, created_assertions, created_evidence = (
            await self._persist(
                planned_entities=planned_entities,
                planned_aliases=planned_aliases,
                planned_resolutions=planned_resolutions,
                facts=facts,
            )
        )
        return CanonicalizationResult(
            mentions_considered=len(mention_context),
            mentions_resolved=len(resolution_map),
            entities_created=created_entities,
            aliases_created=created_aliases,
            exact_or_synonym_matches=deterministic_matches,
            semantic_matches=semantic_matches,
            raw_assertions_considered=len(raw_assertions),
            canonical_assertions_created=created_assertions,
            duplicate_assertions_aggregated=sum(
                max(0, len(fact.raw_assertions) - 1) for fact in facts
            ),
            evidence_links_created=created_evidence,
            relations_other=relations_other,
        )

    async def _load_snapshot(self, *, document_id: str | None) -> dict[str, Any]:
        raw_statement = select(RawAssertion).order_by(RawAssertion.created_at)
        subject_mentions = select(
            RawAssertion.subject_mention_id.label("mention_id")
        )
        object_mentions = select(
            RawAssertion.object_mention_id.label("mention_id")
        )
        document_ids = select(RawAssertion.document_id.label("document_id")).distinct()
        if document_id:
            raw_statement = raw_statement.where(RawAssertion.document_id == document_id)
            subject_mentions = subject_mentions.where(
                RawAssertion.document_id == document_id
            )
            object_mentions = object_mentions.where(
                RawAssertion.document_id == document_id
            )
            document_ids = document_ids.where(RawAssertion.document_id == document_id)

        # Keep large identifier sets inside SQLite. Expanding them into ``IN (?, ...)``
        # parameters exceeds SQLITE_MAX_VARIABLE_NUMBER on realistically sized corpora.
        mention_ids = subject_mentions.union(object_mentions).subquery()
        document_ids = document_ids.subquery()
        async with self.repository.sessions() as session:
            raw_assertions = list((await session.scalars(raw_statement)).all())
            mentions = list(
                (
                    await session.scalars(
                        select(EntityMention).join(
                            mention_ids,
                            EntityMention.mention_id == mention_ids.c.mention_id,
                        )
                    )
                ).all()
            )
            documents = list(
                (
                    await session.scalars(
                        select(Document).join(
                            document_ids,
                            Document.document_id == document_ids.c.document_id,
                        )
                    )
                ).all()
            )
            resolutions = list(
                (
                    await session.scalars(
                        select(EntityResolution).join(
                            mention_ids,
                            EntityResolution.mention_id == mention_ids.c.mention_id,
                        )
                    )
                ).all()
            )
            return {
                "raw_assertions": raw_assertions,
                "mentions": {item.mention_id: item for item in mentions},
                "documents": {item.document_id: item for item in documents},
                "resolutions": {
                    item.mention_id: item.entity_id for item in resolutions
                },
                "entities": list((await session.scalars(select(Entity))).all()),
                "aliases": list((await session.scalars(select(EntityAlias))).all()),
                "external_ids": list(
                    (await session.scalars(select(EntityExternalId))).all()
                ),
            }

    @staticmethod
    def _candidates(
        entities: dict[str, _EntityRecord],
        aliases: defaultdict[str, set[str]],
        external_ids: defaultdict[str, list[str]],
        *,
        entity_type: str,
    ) -> list[EntityCandidate]:
        return [
            EntityCandidate(
                entity_id=entity.entity_id,
                canonical_name=entity.canonical_name,
                entity_type=entity.entity_type,
                aliases=tuple(sorted(aliases[entity.entity_id])),
                external_ids=tuple(sorted(external_ids[entity.entity_id])),
            )
            for entity in entities.values()
            if entity.entity_type == entity_type
        ]

    async def _semantic_entity_decision(
        self,
        *,
        mention: EntityMention,
        raw: RawAssertion,
        document: Document | None,
        retrieved: list[Any],
    ) -> tuple[ResolutionDecision, str]:
        if self.llm is None:
            return ResolutionDecision(None, "new", 1.0), mention.mention_text
        prompt = self.prompts.entity_canonicalization()
        context = raw.evidence_text
        if document and mention.character_start is not None:
            start = max(0, mention.character_start - 350)
            end = min(len(document.content), mention.character_start + 350)
            context = document.content[start:end]
        candidates = [
            {
                "entity_id": item.candidate.entity_id,
                "canonical_name": item.candidate.canonical_name,
                "entity_type": item.candidate.entity_type,
                "aliases": list(item.candidate.aliases),
                "external_ids": list(item.candidate.external_ids),
                "retrieval_score": round(item.score, 6),
            }
            for item in retrieved
        ]
        output = await self.llm.canonicalize_entity(
            system_prompt=prompt.system_prompt,
            user_prompt=prompt.render(
                mention=mention.mention_text,
                entity_type=mention.entity_type,
                evidence_sentence=raw.evidence_text,
                local_context=context,
                document_title=(document.title if document else None) or raw.document_id,
                candidates=json.dumps(candidates, ensure_ascii=False),
            ),
            temperature=0.0,
        )
        candidate_ids = {item.candidate.entity_id for item in retrieved}
        if (
            output.decision == "MATCH"
            and output.entity_id in candidate_ids
            and output.confidence >= self.confidence_threshold
        ):
            return (
                ResolutionDecision(output.entity_id, "semantic_match", output.confidence),
                mention.mention_text,
            )
        canonical_name = (
            output.canonical_name
            if output.decision == "NEW"
            and output.confidence >= self.confidence_threshold
            and output.canonical_name
            else mention.mention_text
        )
        return ResolutionDecision(None, "semantic_new", output.confidence), canonical_name

    async def _semantic_relation_decision(self, raw: RawAssertion) -> Any:
        if self.llm is None:
            return self.relation_normalizer.normalize(raw.detailed_relation)
        prompt = self.prompts.relation_canonicalization()
        output = await self.llm.canonicalize_relation(
            system_prompt=prompt.system_prompt,
            user_prompt=prompt.render(
                subject=raw.subject_mention,
                detailed_relation=raw.detailed_relation,
                object=raw.object_mention,
                evidence_sentence=raw.evidence_text,
                qualifiers=json.dumps(raw.qualifiers, ensure_ascii=False),
                candidates=json.dumps(self.vocabulary, ensure_ascii=False),
            ),
            temperature=0.0,
        )
        allowed = {item.casefold(): item for item in self.vocabulary}
        selected = allowed.get(output.canonical_relation.casefold())
        if selected and output.confidence >= self.confidence_threshold:
            from medical_kg.silver.relation_normalization import RelationMapping

            return RelationMapping(selected, output.confidence)
        return self.relation_normalizer.normalize(raw.detailed_relation)

    async def _persist(
        self,
        *,
        planned_entities: dict[str, _EntityRecord],
        planned_aliases: set[tuple[str, str, str, float]],
        planned_resolutions: list[_ResolutionRecord],
        facts: list[_Fact],
    ) -> tuple[int, int, int, int]:
        entities_created = aliases_created = assertions_created = evidence_created = 0
        async with self.repository._write_session() as session:
            for record in planned_entities.values():
                if await session.get(Entity, record.entity_id) is None:
                    session.add(
                        Entity(
                            entity_id=record.entity_id,
                            canonical_name=record.canonical_name,
                            entity_type=record.entity_type,
                            sources=record.sources,
                        )
                    )
                    entities_created += 1
            await session.flush()

            for entity_id, alias, method, confidence in sorted(planned_aliases):
                statement = (
                    insert(EntityAlias)
                    .values(
                        entity_id=entity_id,
                        alias=alias,
                        alias_source=method,
                        confidence=confidence,
                    )
                    .on_conflict_do_nothing(index_elements=["entity_id", "alias"])
                )
                result = await session.execute(statement)
                aliases_created += result.rowcount or 0

            for record in planned_resolutions:
                statement = (
                    insert(EntityResolution)
                    .values(
                        mention_id=record.mention_id,
                        entity_id=record.entity_id,
                        resolution_method=record.method,
                        confidence=record.confidence,
                        candidate_snapshot=record.candidates,
                    )
                    .on_conflict_do_nothing(index_elements=["mention_id"])
                )
                await session.execute(statement)

            relation_rows = list((await session.scalars(select(RelationType))).all())
            relations = {item.canonical_name.casefold(): item for item in relation_rows}
            existing_assertions = {
                item.normalized_identity: item
                for item in (await session.scalars(select(Assertion))).all()
            }
            for fact in facts:
                assertion = existing_assertions.get(fact.identity)
                if assertion is None:
                    relation = relations[fact.relation_name.casefold()]
                    assertion = Assertion(
                        raw_assertion_id=fact.raw_assertions[0].raw_assertion_id,
                        subject_entity_id=fact.subject_entity_id,
                        object_entity_id=fact.object_entity_id,
                        canonical_relation_id=relation.relation_id,
                        qualifiers=fact.qualifiers,
                        negated=fact.negated,
                        speculative=fact.speculative,
                        normalized_identity=fact.identity,
                        sources=fact.sources,
                    )
                    session.add(assertion)
                    await session.flush()
                    existing_assertions[fact.identity] = assertion
                    assertions_created += 1
                else:
                    assertion.sources = _merge_sources(
                        list(assertion.sources), fact.sources
                    )

                for raw in fact.raw_assertions:
                    statement = (
                        insert(AssertionEvidence)
                        .values(
                            assertion_id=assertion.assertion_id,
                            raw_assertion_id=raw.raw_assertion_id,
                            document_id=raw.document_id,
                            evidence_text=raw.evidence_text,
                            llm_confidence=raw.llm_confidence,
                            sources=raw.sources,
                        )
                        .on_conflict_do_nothing(index_elements=["raw_assertion_id"])
                    )
                    result = await session.execute(statement)
                    evidence_created += result.rowcount or 0
        return entities_created, aliases_created, assertions_created, evidence_created
