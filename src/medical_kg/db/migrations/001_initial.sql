-- Reference DDL is generated from src/medical_kg/db/models.py.
-- `medical-kg init-db` applies the SQLAlchemy metadata and the additive upgrades below.

ALTER TABLE processing_jobs ADD COLUMN IF NOT EXISTS heartbeat_at TIMESTAMPTZ;
ALTER TABLE processing_jobs ADD COLUMN IF NOT EXISTS lease_expires_at TIMESTAMPTZ;

CREATE TABLE IF NOT EXISTS extraction_chunk_jobs (
    chunk_job_id UUID PRIMARY KEY,
    job_id UUID NOT NULL REFERENCES processing_jobs(job_id) ON DELETE CASCADE,
    document_id VARCHAR(255) NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    pass_name VARCHAR(128) NOT NULL,
    stage_version VARCHAR(128) NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    chunk_index INTEGER NOT NULL,
    character_start INTEGER NOT NULL,
    character_end INTEGER NOT NULL,
    status VARCHAR(16) NOT NULL,
    worker_id VARCHAR(255),
    retry_count INTEGER NOT NULL DEFAULT 0,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    error_message TEXT,
    validated_output JSONB,
    raw_output JSONB,
    model_provider VARCHAR(64),
    model_name VARCHAR(255),
    temperature DOUBLE PRECISION,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_extraction_chunk_unit UNIQUE
        (document_id, pass_name, stage_version, content_hash, chunk_index),
    CONSTRAINT ck_extraction_chunk_status CHECK
        (status IN ('PENDING', 'RUNNING', 'SUCCESS', 'FAILED')),
    CONSTRAINT ck_extraction_chunk_input_tokens CHECK (input_tokens >= 0),
    CONSTRAINT ck_extraction_chunk_output_tokens CHECK (output_tokens >= 0)
);

CREATE INDEX IF NOT EXISTS ix_extraction_chunks_job_status
    ON extraction_chunk_jobs(job_id, status);

-- Additive upgrade for databases whose chunk table predates model provenance. The nullable
-- columns deliberately preserve and reuse legacy SUCCESS checkpoints.
ALTER TABLE extraction_chunk_jobs ADD COLUMN IF NOT EXISTS model_provider VARCHAR(64);
ALTER TABLE extraction_chunk_jobs ADD COLUMN IF NOT EXISTS model_name VARCHAR(255);
ALTER TABLE extraction_chunk_jobs ADD COLUMN IF NOT EXISTS temperature DOUBLE PRECISION;

CREATE TABLE IF NOT EXISTS api_rate_limits (
    limiter_key VARCHAR(512) PRIMARY KEY,
    next_request_at TIMESTAMPTZ NOT NULL,
    next_token_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
