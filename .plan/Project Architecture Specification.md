# Medical Literature Knowledge Graph — Project Architecture Specification

## 1. Project Objective

Build a maintainable Python pipeline that converts large-scale local biomedical literature into a structured, traceable, incrementally updatable medical knowledge graph.

The system focuses on four core capabilities:

1. **Full-document LLM knowledge extraction**
2. **Entity and relation canonicalization**
3. **Evidence- and provenance-aware knowledge storage**
4. **Resumable and parallel large-scale processing**

The first version uses **PostgreSQL as the authoritative knowledge store**. Graph databases, ontology enrichment, Parquet snapshots, and integration with external knowledge graphs are downstream extensions rather than prerequisites.

---

# 2. Knowledge Representation

The fundamental unit of extracted knowledge is an **Assertion**:

```text
Subject Entity
    │
    ├── detailed relation
    ├── canonical relation
    ▼
Object Entity
```

Every assertion is supported by evidence from a specific document.

Conceptually:

```text
(subject, relation, object, evidence, context, provenance)
```

The internal representation must preserve substantially more information than a traditional triple.

---

# 3. Core Data Pipeline

```text
Local Documents
      │
      ▼
Landing
      │
      │ document registration
      │ metadata
      │ content hash
      ▼
Bronze
      │
      │ full-document LLM extraction
      │ entity mentions
      │ raw assertions
      │ evidence
      ▼
Silver
      │
      │ entity canonicalization
      │ relation canonicalization
      │ alias aggregation
      │ deduplication
      ▼
Gold
      │
      │ canonical entities
      │ canonical assertions
      │ provenance
      │ validation
      ▼
PostgreSQL Knowledge Base
```

Each stage must persist its outputs independently.

Silver processing must never overwrite or destroy Bronze extraction results.

---

# 4. Document Model

Each input paper must have a stable `document_id`.

Required document fields:

```text
document_id
file_path
title
doi
pmid

content
content_hash

created_at
updated_at
```

`content_hash` determines whether a document has changed.

If the same `document_id` is encountered with an unchanged hash, completed processing stages should not run again unless explicitly requested.

---

# 5. Full-Document Extraction

## 5.1 Extraction Principle

The default extraction unit is the **complete paper**, not an independently processed text chunk.

Each LLM extraction request should receive the full available document so that the model can reason over:

```text
cross-section relations
cross-paragraph relations
coreference
study population
experimental design
temporal context
treatment context
outcome context
```

Chunking may be used internally for storage or preprocessing, but should not define the semantic extraction boundary.

---

## 5.2 Multi-Pass Extraction

A document may undergo multiple independently configurable extraction passes.

Recommended initial passes:

```text
general biomedical knowledge
disease / phenotype
drug / treatment
gene / protein / pathway
physiology / biochemistry
risk / lifestyle / environment
clinical outcome / population
```

Each pass receives the complete paper.

For the first implementation, a smaller set may be used, for example:

```text
general
molecular
clinical
```

Every extraction pass has an explicit:

```text
pass_name
prompt_version
model_name
extraction_run_id
```

---

# 6. Entity Model

Extraction and canonicalization must be separate.

```text
EntityMention
      │
      ▼
CanonicalEntity
      │
      ├── aliases
      └── external identifiers
```

## 6.1 EntityMention

Represents the text actually appearing in a paper.

Required fields:

```text
mention_id
document_id

mention_text
entity_type
entity_type_detail

section
paragraph
sentence
character_start
character_end
page

extraction_run_id
```

Position fields are nullable when unavailable.

The extraction model must not replace the paper's wording with a preferred ontology term.

Example:

```text
mention_text = "T2DM"
```

rather than automatically converting it to:

```text
"type 2 diabetes mellitus"
```

---

## 6.2 CanonicalEntity

Represents a concept shared across documents.

```text
entity_id
canonical_name
entity_type
description

created_at
updated_at
```

Internal identifiers should be stable and independent of display names:

```text
ENT_00000001
ENT_00000002
...
```

Changing aliases or ontology mappings must not change `entity_id`.

---

## 6.3 Entity Types

The extraction schema should support common biomedical classes:

```text
DISEASE
PHENOTYPE
SYMPTOM

DRUG
COMPOUND
METABOLITE

GENE
PROTEIN
PATHWAY

CELL
TISSUE
ORGAN

BIOMARKER
LAB_MEASUREMENT

PHYSIOLOGICAL_PROCESS
BIOLOGICAL_PROCESS

TREATMENT
INTERVENTION
PROCEDURE

DIET
NUTRIENT
FOOD
EXERCISE

BEHAVIOR
LIFESTYLE_FACTOR
ENVIRONMENTAL_EXPOSURE
RISK_FACTOR

POPULATION
CLINICAL_OUTCOME

OTHER
```

`OTHER` must remain valid.

When `OTHER` is used, the model should additionally provide:

```text
entity_type_detail
```

No useful entity should be discarded solely because it is not represented by the current controlled vocabulary.

---

# 7. Entity Alias Model

Aliases are stored independently from canonical entities.

```text
entity_aliases
```

Fields:

```text
alias_id
entity_id

alias
alias_source
confidence

created_at
```

Example:

```text
ENT_000042
├── T2DM
├── type II diabetes
├── type 2 diabetes
└── type 2 diabetes mellitus
```

---

# 8. External Identifier Model

Ontology/database mapping is enrichment rather than an extraction dependency.

Use a dedicated table:

```text
entity_external_ids
```

Fields:

```text
mapping_id
entity_id

namespace
accession
normalized_id

mapping_method
mapping_source
mapping_confidence

is_primary
created_at
```

Example:

```text
entity_id          ENT_000042
namespace          MONDO
accession          0005148
normalized_id      MONDO_0005148
mapping_method     exact_synonym
mapping_confidence 0.98
```

Potential future namespaces include:

```text
UMLS
MeSH
SNOMED CT
ICD
MONDO
HPO

HGNC
NCBI Gene
Ensembl
UniProt

ChEBI
DrugBank

GO
UBERON
Reactome
```

Ontology mapping failure must never cause an extracted entity to be deleted.

---

# 9. Raw Assertion Model

Bronze stores LLM extraction as faithfully as possible.

Required fields:

```text
raw_assertion_id

subject_mention_id
object_mention_id

subject_mention
subject_type

object_mention
object_type

detailed_relation

llm_confidence

document_id
evidence_text

qualifiers

negated
speculative

extraction_run_id

created_at
```

---

## 9.1 Detailed Relation

`detailed_relation` represents the specific relation expressed by the source paper.

Example:

```text
significantly reduced fasting plasma glucose after 12 weeks of treatment
```

It should not be prematurely compressed during extraction.

---

# 10. Canonical Assertion Model

Silver converts raw assertions into canonical graph knowledge.

```text
assertion_id

subject_entity_id
object_entity_id

raw_assertion_id

canonical_relation_id

negated
speculative

qualifiers

created_at
updated_at
```

The raw assertion remains permanently available through `raw_assertion_id`.

---

# 11. Relation Model

Relations have three semantic levels:

```text
source language
      ↓
detailed_relation
      ↓
canonical_relation
```

Example:

```text
"was associated with an elevated risk of"
             ↓
increases_risk_of
```

The canonical vocabulary is configurable and must not be embedded in business logic.

---

## 11.1 Initial Canonical Relation Vocabulary

```text
associated_with
positively_associated_with
negatively_associated_with

correlated_with

increases
decreases

increases_risk_of
decreases_risk_of

causes
contributes_to

prevents
protects_against

treats
ameliorates
worsens

activates
inhibits
regulates
upregulates
downregulates

interacts_with
binds_to

expressed_in
located_in

part_of
includes

biomarker_of
predicts
diagnoses

produces
metabolizes
converts_to

required_for
promotes
suppresses

OTHER
```

If no relation can be mapped without semantic distortion:

```text
canonical_relation = OTHER
```

The original `detailed_relation` remains authoritative.

---

## 11.2 Relation Type Table

```text
relation_types
```

Fields:

```text
relation_id
canonical_name
description

parent_relation_id
inverse_relation_id

deprecated

created_at
updated_at
```

Optional ontology mapping fields may later be added without modifying assertions.

---

# 12. Qualifiers

Medical assertions frequently depend on contextual conditions.

`qualifiers` should therefore be stored as structured JSON.

Recommended fields:

```json
{
  "species": null,
  "population": null,
  "age": null,
  "sex": null,
  "disease_state": null,
  "experimental_model": null,

  "dose": null,
  "route": null,
  "frequency": null,

  "duration": null,
  "timepoint": null,

  "measurement_method": null,

  "direction": null,
  "effect_size": null,
  "statistical_significance": null,

  "condition": null,
  "tissue": null,
  "cell_type": null,

  "study_type": null
}
```

Only information explicitly supported by the source paper should be populated.

Additional qualifier keys should be permitted.

---

# 13. Evidence Model

Evidence is first-class data.

At minimum:

```text
document_id
evidence_text
```

When source parsing supports it:

```text
section
paragraph
sentence

character_start
character_end

page
```

The system should validate that `evidence_text` occurs in the corresponding source document whenever exact text matching is possible.

Evidence text must not be generated from model background knowledge.

---

# 14. Confidence

Each extracted assertion contains:

```text
llm_confidence
```

Constraint:

```text
0.0 <= llm_confidence <= 1.0
```

This represents model self-assessment of extraction support, not a calibrated probability.

Future assertion-level scores may include:

```text
verification_score
supporting_document_count
contradicting_document_count
```

These should remain separate from `llm_confidence`.

---

# 15. PostgreSQL Schema

The authoritative database should contain at least:

```text
documents
processing_jobs

extraction_runs

entity_mentions
entities
entity_aliases
entity_external_ids

raw_assertions
assertions

relation_types
```

---

## 15.1 documents

```text
document_id          PK
file_path
title
doi
pmid

content
content_hash

created_at
updated_at
```

Recommended constraints:

```text
UNIQUE(document_id)
INDEX(content_hash)
INDEX(doi)
INDEX(pmid)
```

---

## 15.2 extraction_runs

Tracks the exact extraction configuration.

```text
extraction_run_id    PK

model_provider
model_name

prompt_name
prompt_version
pass_name

temperature

code_version

created_at
```

---

## 15.3 processing_jobs

Tracks processing state by document and stage.

```text
job_id               PK

document_id           FK

stage
stage_version

status

retry_count

worker_id

started_at
finished_at

error_message
```

Valid states:

```text
PENDING
RUNNING
SUCCESS
FAILED
```

Recommended uniqueness:

```text
UNIQUE(
    document_id,
    stage,
    stage_version
)
```

---

## 15.4 entity_mentions

```text
mention_id            PK
document_id           FK
extraction_run_id     FK

mention_text
entity_type
entity_type_detail

section
paragraph
sentence

character_start
character_end
page

created_at
```

---

## 15.5 entities

```text
entity_id             PK

canonical_name
entity_type
description

created_at
updated_at
```

---

## 15.6 entity_aliases

```text
alias_id              PK
entity_id             FK

alias
alias_source
confidence

created_at
```

Recommended constraint:

```text
UNIQUE(entity_id, alias)
```

---

## 15.7 entity_external_ids

```text
mapping_id            PK
entity_id             FK

namespace
accession
normalized_id

mapping_method
mapping_source
mapping_confidence

is_primary

created_at
```

Recommended constraint:

```text
UNIQUE(entity_id, namespace, accession)
```

---

## 15.8 raw_assertions

```text
raw_assertion_id       PK

document_id            FK
extraction_run_id      FK

subject_mention_id     FK
object_mention_id      FK

subject_mention
subject_type

object_mention
object_type

detailed_relation

llm_confidence

evidence_text

qualifiers             JSONB

negated
speculative

raw_llm_output         JSONB

created_at
```

---

## 15.9 relation_types

```text
relation_id            PK

canonical_name
description

parent_relation_id
inverse_relation_id

deprecated

created_at
updated_at
```

Recommended constraint:

```text
UNIQUE(canonical_name)
```

---

## 15.10 assertions

```text
assertion_id            PK

raw_assertion_id        FK

subject_entity_id       FK
object_entity_id        FK

canonical_relation_id   FK

qualifiers              JSONB

negated
speculative

created_at
updated_at
```

A normalized assertion identity may be generated from:

```text
subject_entity_id
canonical_relation_id
object_entity_id
qualifiers
negated
speculative
```

to support deduplication.

---

# 16. LLM Structured Output Schema

Pydantic should validate all model outputs before persistence.

Example extraction schema:

```python
class EntityMentionOutput(BaseModel):
    mention: str
    entity_type: str
    entity_type_detail: str | None = None


class Qualifiers(BaseModel):
    species: str | None = None
    population: str | None = None
    age: str | None = None
    sex: str | None = None
    disease_state: str | None = None
    experimental_model: str | None = None

    dose: str | None = None
    route: str | None = None
    frequency: str | None = None

    duration: str | None = None
    timepoint: str | None = None

    measurement_method: str | None = None

    direction: str | None = None
    effect_size: str | None = None
    statistical_significance: str | None = None

    condition: str | None = None
    tissue: str | None = None
    cell_type: str | None = None

    study_type: str | None = None


class AssertionOutput(BaseModel):
    subject: EntityMentionOutput
    object: EntityMentionOutput

    detailed_relation: str

    evidence_text: str

    qualifiers: Qualifiers

    negated: bool = False
    speculative: bool = False

    llm_confidence: float = Field(ge=0.0, le=1.0)


class ExtractionOutput(BaseModel):
    assertions: list[AssertionOutput]
```

Position metadata should be added during deterministic post-processing when possible rather than relying entirely on the LLM.

---

# 17. LLM Abstraction

Provider-specific APIs should be isolated behind one interface.

```text
LLMClient
```

Core methods:

```python
extract_document(...)
canonicalize_entity(...)
canonicalize_relation(...)
```

Provider implementations may include:

```text
OpenAIClient
AnthropicClient
GeminiClient
CompatibleAPIClient
```

Business logic must depend only on `LLMClient`.

---

# 18. Prompt Management

Prompts should be versioned files rather than Python constants.

Recommended layout:

```text
prompts/
├── extraction.yaml
├── entity_canonicalization.yaml
└── relation_canonicalization.yaml
```

Each prompt definition contains:

```text
name
version
system_prompt
user_template
```

Example:

```text
extraction_general_v1
entity_canonicalization_v1
relation_canonicalization_v1
```

Extraction runs must record the prompt version used.

The canonical relation vocabulary should also be loaded from configuration.

---

# 19. Extraction Constraints

The extraction prompt must explicitly require the model to:

```text
extract only claims supported by the paper

maximize recall without adding external biomedical knowledge

distinguish:
association
correlation
causation
mechanism

preserve:
negation
speculation
population
species
experimental context
dose
duration

provide exact supporting evidence

allow multiple relations between the same entity pair
```

Entity co-occurrence alone is not sufficient evidence for a relation.

Statements appearing in background sections may be extracted as statements made by the paper, but must retain their evidence and contextual qualifiers.

---

# 20. Entity Canonicalization

Canonicalization is a separate Silver-stage operation.

Input:

```text
EntityMention
+
existing CanonicalEntities
+
aliases
+
optional ontology information
```

Output:

```text
existing entity_id
```

or:

```text
new entity
```

Canonicalization should prioritize semantic identity rather than string equality.

The system must preserve ambiguous or unresolved mentions rather than incorrectly merging entities.

---

# 21. Relation Canonicalization

Input:

```text
subject entity/type
detailed_relation
object entity/type
qualifiers
relation candidate pool
```

Output:

```text
canonical_relation
mapping_confidence
```

If no candidate preserves the original semantics:

```text
OTHER
```

Relation normalization must never modify `detailed_relation`.

---

# 22. Resumable Processing

Each processing unit is identified by:

```text
document_id
stage
stage_version
```

Before execution:

```text
SUCCESS → skip
FAILED  → retry if requested
PENDING → process
```

A stage version should change when relevant implementation semantics change, for example:

```text
extract:v1
entity_canonicalize:v2
relation_canonicalize:v1
```

Writes should use database transactions and UPSERT where applicable.

Processing state must reside in PostgreSQL rather than in-memory state.

---

# 23. Parallel Extraction

LLM extraction should use:

```text
asyncio
+
bounded semaphore
```

Required runtime configuration:

```yaml
processing:
  max_concurrency:
  requests_per_minute:
  tokens_per_minute:
  max_retries:
  request_timeout:
  retry_backoff:
```

Retry applies to:

```text
rate limit
timeout
temporary provider errors
connection failures
malformed structured output
```

Retries are bounded.

One document failure must not terminate processing of unrelated documents.

---

# 24. Configuration

Use:

```text
config.yaml
```

and:

```text
config.example.yaml
```

Recommended structure:

```yaml
llm:
  provider: openai
  model: example-model
  api_key: xxx
  base_url:
  temperature: 0
  timeout: 120

processing:
  max_concurrency: 8
  requests_per_minute: 100
  tokens_per_minute: 1000000
  max_retries: 4
  retry_backoff: 2

database:
  host: localhost
  port: 5432
  database: medical_kg
  user:
  password:

extraction:
  passes:
    - general
    - molecular
    - clinical

relations:
  vocabulary_file: config/relations.yaml
```

`config.yaml` is local-only and excluded from version control.

---

# 25. Logging and Run Statistics

Structured logs should include:

```text
document_id
stage
extraction_run_id

model
attempt

elapsed_time

input_tokens
output_tokens

status
error
```

Run-level statistics:

```text
documents_processed
documents_successful
documents_failed

requests

input_tokens
output_tokens

estimated_cost
```

No additional observability infrastructure is required for the initial implementation.

---

# 26. Validation Before Gold

The Silver → Gold transition should enforce:

```text
subject entity exists
object entity exists

detailed relation is non-empty

confidence ∈ [0,1]

evidence belongs to source document

canonical relation exists

foreign keys are valid

duplicate normalized assertions are resolved
```

Invalid records should remain traceable rather than silently discarded.

---

# 27. Project Structure

```text
medical-kg/
│
├── README.md
├── pyproject.toml
├── .gitignore
│
├── config.example.yaml
│
├── config/
│   └── relations.yaml
│
├── prompts/
│   ├── extraction.yaml
│   ├── entity_canonicalization.yaml
│   └── relation_canonicalization.yaml
│
├── src/
│   └── medical_kg/
│
│       ├── config.py
│       │
│       ├── models/
│       │   ├── document.py
│       │   ├── entity.py
│       │   ├── assertion.py
│       │   └── job.py
│       │
│       ├── llm/
│       │   ├── base.py
│       │   ├── client.py
│       │   └── schemas.py
│       │
│       ├── landing/
│       │   └── loader.py
│       │
│       ├── bronze/
│       │   └── extraction.py
│       │
│       ├── silver/
│       │   ├── entity_resolution.py
│       │   ├── relation_normalization.py
│       │   └── deduplication.py
│       │
│       ├── gold/
│       │   └── validation.py
│       │
│       ├── db/
│       │   ├── models.py
│       │   ├── repository.py
│       │   └── migrations/
│       │
│       ├── pipeline/
│       │   ├── runner.py
│       │   └── worker.py
│       │
│       └── cli.py
│
└── tests/
    ├── test_extraction_schema.py
    ├── test_repository.py
    ├── test_entity_resolution.py
    └── test_pipeline_resume.py
```

The important architectural boundaries are:

```text
Extraction
≠
Canonicalization
≠
Persistence
≠
Pipeline orchestration
```

---

# 28. CLI

Minimum commands:

```bash
python -m medical_kg.cli ingest ./papers

python -m medical_kg.cli extract

python -m medical_kg.cli canonicalize

python -m medical_kg.cli run

python -m medical_kg.cli retry-failed

python -m medical_kg.cli status
```

Small-scale execution:

```bash
python -m medical_kg.cli run --limit 100
```

Useful optional selectors:

```text
--document-id
--stage
--stage-version
--pass-name
```

---

# 29. Minimum Viable Pipeline

The first implementation should stop after a reliable Bronze pipeline:

```text
local papers
     ↓
document registration
     ↓
content hashing
     ↓
parallel full-document extraction
     ↓
Pydantic validation
     ↓
entity mentions
     ↓
raw assertions
     ↓
PostgreSQL
```

Required first-stage components:

```text
configuration
prompt versioning
PostgreSQL schema
LLM client abstraction
parallel extraction
structured output validation
processing_jobs
retry
resumability
run statistics
```

Entity/relation canonicalization should be implemented only after this pipeline is stable.

---

# 30. Development Stages

## Phase I — Literature Extraction

```text
documents
extraction_runs
processing_jobs
entity_mentions
raw_assertions
```

Goal:

> reliably transform large numbers of papers into evidence-backed raw assertions.

---

## Phase II — Knowledge Canonicalization

Add:

```text
entities
entity_aliases
entity_external_ids
relation_types
assertions
```

Implement:

```text
entity resolution
alias aggregation
relation normalization
deduplication
```

Goal:

> create a stable cross-document biomedical knowledge graph.

---

## Phase III — Knowledge Enrichment and Export

Potential extensions:

```text
ontology mapping
external terminology enrichment

support / contradiction aggregation

Parquet snapshots
Neo4j export

graph analytics
GraphRAG
```

These features should not affect the core extraction architecture.

---

# 31. Future Extensions

The schema should remain compatible with future integration of external biomedical resources.

Potential tasks include:

```text
additional ontology mappings

external knowledge graph entity alignment

relation semantic mapping

merging selected external KG subgraphs

knowledge conflict analysis

evidence aggregation across literature and curated databases
```

A possible future external source is **OptimusKG**. Integration should be performed through entity-ID and relation-semantic mapping at that stage; the native literature KG schema should not be reduced in advance merely to match an external graph.

---

# 32. Final Architecture Summary

The project centers on five persistent objects:

```text
Document
EntityMention
CanonicalEntity
RawAssertion
Assertion
```

and three transformation stages:

```text
Extraction
      ↓
Canonicalization
      ↓
Knowledge aggregation
```

The key data flow is:

```text
Paper
  ↓
EntityMention + RawAssertion + Evidence
  ↓
CanonicalEntity + CanonicalRelation
  ↓
Assertion
  ↓
Medical Knowledge Base
```

The PostgreSQL representation is the authoritative source of truth.

Any future graph representation, ontology mapping, external KG integration, or GraphRAG system should be derived from this knowledge layer rather than redefining the extraction schema.