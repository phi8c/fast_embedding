from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import json
import logging
import math
import os
import random
import shutil
import signal
import sqlite3
import sys
import time
import tempfile
from contextlib import contextmanager
from itertools import islice
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ClientAuthenticationError, HttpResponseError, ServiceRequestError, ServiceResponseError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from datasets import load_dataset
from dotenv import load_dotenv
from FlagEmbedding import BGEM3FlagModel
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url, get_hf_file_metadata
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "progress.db"
DRY_RUN_DB_PATH = ROOT / "progress.dryrun.db"
LOG_DIR = ROOT / "logs"

DEFAULT_SAFE_MAX_TOKENS = 7900
DEFAULT_CHUNK_TOKENS = 4096
DEFAULT_UPLOAD_BATCH = 25
MAX_UPLOAD_BYTES = 14 * 1024 * 1024
MAX_PENDING_VECTORS = 250
MAX_UPLOAD_ATTEMPTS = 6
MAX_SHARD_CACHE_BYTES = 512 * 1024 * 1024  # Legacy constant; shard size is now disk-budget based.
PARQUET_READ_BATCH_ROWS = 32
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}
STATUSES = ("PENDING", "EMBEDDED", "UPLOADED", "FAILED_RETRYABLE", "BLOCKED")

STOP_REQUESTED = False


class UploadRetriesExhausted(RuntimeError):
    def __init__(self, uploaded: int, pending: int):
        self.uploaded = uploaded
        self.pending = pending
        super().__init__(f"Azure retry budget exhausted: {pending} unconfirmed documents retained; stopping pipeline.")


@dataclass(frozen=True)
class Config:
    azure_endpoint: str
    azure_key: str
    azure_index: str
    dataset_id: str
    model_id: str
    dimensions: int
    safe_max_tokens: int
    chunk_tokens: int
    upload_batch_size: int
    dry_run: bool
    limit: int | None
    test_namespace: str
    min_free_gb: float
    cpu_threads: int | None
    db_path: Path


@dataclass
class RuntimeStats:
    read: int = 0
    embedded: int = 0
    uploaded: int = 0
    retrying: int = 0
    blocked: int = 0
    started_at: float = 0.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger("legal_embedding")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = logging.FileHandler(LOG_DIR / f"run-{datetime.now():%Y%m%d-%H%M%S}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


logger = setup_logging()


def handle_signal(signum: int, frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    logger.warning("Shutdown requested; current record/batch will be made safe before exit.")


def load_config() -> Config:
    load_dotenv(ROOT / ".env")
    required = [
        "AZURE_SEARCH_ENDPOINT",
        "AZURE_SEARCH_API_KEY",
        "AZURE_SEARCH_INDEX_NAME",
        "HF_DATASET_ID",
        "EMBEDDING_MODEL",
        "EMBEDDING_DIMENSIONS",
    ]
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(f"Missing required .env variables: {', '.join(missing)}")

    parser = argparse.ArgumentParser(description="Embed Vietnamese legal dataset and upload to Azure AI Search.")
    parser.add_argument("--limit", type=int, default=None, help="Process at most N new records.")
    parser.add_argument("--dry-run", action="store_true", help="Embed and validate records without uploading to Azure.")
    parser.add_argument("--test-namespace", default=None, help="ID namespace, e.g. smoke or prod.")
    parser.add_argument("--safe-max-tokens", type=int, default=DEFAULT_SAFE_MAX_TOKENS)
    parser.add_argument("--chunk-tokens", type=int, default=DEFAULT_CHUNK_TOKENS)
    parser.add_argument("--upload-batch-size", type=int, default=DEFAULT_UPLOAD_BATCH)
    parser.add_argument("--min-free-gb", type=float, default=5.0)
    parser.add_argument("--cpu-threads", type=int, default=None)
    args = parser.parse_args()
    if os.environ["EMBEDDING_MODEL"] != "BAAI/bge-m3" or int(os.environ["EMBEDDING_DIMENSIONS"]) != 1024:
        raise RuntimeError("This pipeline requires BAAI/bge-m3 with 1024 dimensions.")
    if not 128 <= args.safe_max_tokens <= 8192:
        raise RuntimeError("--safe-max-tokens must be between 128 and 8192.")

    if args.limit is not None and args.limit < 1:
        raise RuntimeError("--limit must be positive.")
    if args.upload_batch_size < 1 or args.upload_batch_size > 25:
        raise RuntimeError("--upload-batch-size must be between 1 and 25.")
    if args.chunk_tokens < 128 or args.chunk_tokens > args.safe_max_tokens - 2:
        raise RuntimeError("--chunk-tokens must be >= 128 and <= --safe-max-tokens.")

    return Config(
        azure_endpoint=os.environ["AZURE_SEARCH_ENDPOINT"].rstrip("/"),
        azure_key=os.environ["AZURE_SEARCH_API_KEY"],
        azure_index=os.environ["AZURE_SEARCH_INDEX_NAME"],
        dataset_id=os.environ["HF_DATASET_ID"],
        model_id=os.environ["EMBEDDING_MODEL"],
        dimensions=int(os.environ["EMBEDDING_DIMENSIONS"]),
        safe_max_tokens=args.safe_max_tokens,
        chunk_tokens=args.chunk_tokens,
        upload_batch_size=args.upload_batch_size,
        dry_run=args.dry_run,
        limit=args.limit,
        test_namespace=(args.test_namespace.strip() if args.test_namespace else ("dryrun" if args.dry_run else "prod")),
        min_free_gb=args.min_free_gb,
        cpu_threads=args.cpu_threads,
        db_path=DRY_RUN_DB_PATH if args.dry_run else DB_PATH,
    )


def connect_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS run_config (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            dataset_pos INTEGER NOT NULL,
            source_document_id TEXT,
            status TEXT NOT NULL CHECK(status IN ('PENDING','EMBEDDED','UPLOADED','FAILED_RETRYABLE','BLOCKED')),
            retry_count INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL,
            last_error TEXT,
            payload_json TEXT,
            embedding_json TEXT
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS checkpoint (
            id INTEGER PRIMARY KEY CHECK(id = 1),
            last_dataset_pos INTEGER NOT NULL DEFAULT -1,
            updated_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "INSERT OR IGNORE INTO checkpoint(id, last_dataset_pos, updated_at) VALUES(1, -1, ?)",
        (utc_now(),),
    )
    columns = {row[1] for row in conn.execute("PRAGMA table_info(checkpoint)")}
    if "reader_state" not in columns:
        conn.execute("ALTER TABLE checkpoint ADD COLUMN reader_state TEXT")
    conn.commit()
    return conn


def config_fingerprint(cfg: Config, dataset_revision: str) -> dict[str, Any]:
    return {
        "dataset_id": cfg.dataset_id,
        "dataset_revision": dataset_revision,
        "model_id": cfg.model_id,
        "embedding_dimensions": cfg.dimensions,
        "max_tokens_per_chunk": cfg.chunk_tokens,
        "pooling": "weighted_mean_by_chunk_token_count_l2_normalized",
        "azure_index_name": cfg.azure_index,
        "test_namespace": cfg.test_namespace,
    }


def assert_or_store_run_config(conn: sqlite3.Connection, fp: dict[str, Any]) -> None:
    existing = dict(conn.execute("SELECT key, value FROM run_config").fetchall())
    if not existing:
        with conn:
            for key, value in fp.items():
                conn.execute("INSERT INTO run_config(key, value) VALUES(?, ?)", (key, json.dumps(value, ensure_ascii=False)))
        return
    mismatches = []
    for key, value in fp.items():
        old = existing.get(key)
        if old is not None and json.loads(old) != value:
            mismatches.append((key, json.loads(old), value))
    if mismatches:
        details = "; ".join(f"{k}: checkpoint={old!r}, current={new!r}" for k, old, new in mismatches)
        raise RuntimeError(f"Checkpoint configuration mismatch. Stop for manual migration/reset: {details}")


def get_dataset_revision(dataset_id: str) -> str:
    for attempt in range(MAX_UPLOAD_ATTEMPTS):
        try:
            revision = HfApi().dataset_info(dataset_id).sha
            if not revision:
                raise RuntimeError("Dataset revision is missing.")
            return revision
        except Exception as exc:
            if not hf_retryable(exc) or attempt == MAX_UPLOAD_ATTEMPTS - 1:
                raise
            interruptible_wait(retry_delay(attempt))
    raise InterruptedError("Shutdown requested.")


def make_clients(cfg: Config) -> tuple[SearchIndexClient, SearchClient]:
    credential = AzureKeyCredential(cfg.azure_key)
    return (
        SearchIndexClient(endpoint=cfg.azure_endpoint, credential=credential),
        SearchClient(endpoint=cfg.azure_endpoint, index_name=cfg.azure_index, credential=credential,
                     retry_total=0, connection_timeout=15, read_timeout=60),
    )


def azure_type(field: Any) -> str:
    field_type = getattr(field, "type", "")
    # SDK 12 returns Enum members; str(member) is not the wire-format EDM type.
    return str(getattr(field_type, "value", field_type))


def preflight(cfg: Config, index_client: SearchIndexClient) -> dict[str, str]:
    free_gb = shutil.disk_usage(ROOT).free / (1024**3)
    if free_gb < cfg.min_free_gb:
        raise RuntimeError(f"Insufficient disk space: {free_gb:.1f} GB free, need at least {cfg.min_free_gb:.1f} GB.")

    logger.info("Checking Azure index schema.")
    index = index_client.get_index(cfg.azure_index)
    fields = {field.name: field for field in index.fields}
    if "embedding" not in fields:
        raise RuntimeError("Azure index does not contain vector field 'embedding'.")
    vector_dims = getattr(fields["embedding"], "vector_search_dimensions", None)
    if vector_dims != cfg.dimensions:
        raise RuntimeError(f"Azure embedding dimensions mismatch: index={vector_dims}, config={cfg.dimensions}.")
    if "id" not in fields:
        raise RuntimeError("Azure index must contain key field 'id'.")
    key_type = azure_type(fields["id"])
    key_flag = getattr(fields["id"], "key", None)
    if key_flag is not True or key_type != "Edm.String":
        raise RuntimeError(f"Azure key must be the string field 'id': key={key_flag!r}, type={key_type!r}, raw_type={fields['id'].type!r}.")
    if azure_type(fields["embedding"]) != "Collection(Edm.Single)":
        raise RuntimeError(f"Azure embedding must be Collection(Edm.Single), got {azure_type(fields['embedding'])!r}.")
    logger.info("Azure schema validated: id key=true type=%s, embedding dimensions=%d", key_type, vector_dims)
    if "document_id" not in fields:
        logger.warning("Azure index has no 'document_id' field; only key 'id' will be uploaded.")
    return {name: azure_type(field) for name, field in fields.items()}


def load_model(cfg: Config) -> BGEM3FlagModel:
    if cfg.cpu_threads:
        torch.set_num_threads(cfg.cpu_threads)
    logger.info("Loading embedding model on CPU: %s", cfg.model_id)
    return BGEM3FlagModel(cfg.model_id, use_fp16=False, devices="cpu")


def get_tokenizer(model: BGEM3FlagModel) -> Any:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None and hasattr(model, "model"):
        tokenizer = getattr(model.model, "tokenizer", None)
    if tokenizer is None:
        raise RuntimeError("Unable to access BGE-M3 tokenizer from FlagEmbedding model.")
    return tokenizer


def token_count(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=True))


def decode_chunks(tokenizer: Any, text: str, chunk_tokens: int) -> Iterable[tuple[str, int]]:
    # Bound tokenization even for a single extremely long paragraph.
    for start in range(0, len(text), 16384):
        part = text[start:start + 16384]
        yield from bounded_chunks(tokenizer, part, chunk_tokens)


def bounded_chunks(tokenizer: Any, text: str, budget: int) -> Iterable[tuple[str, int]]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= budget:
        if ids:
            yield text, len(ids)
        return
    middle = len(text) // 2
    if middle == 0:
        raise ValueError("A single character exceeds the token budget.")
    boundary = text.rfind(" ", max(1, middle // 2), middle + 1)
    middle = boundary + 1 if boundary > 0 else middle
    yield from bounded_chunks(tokenizer, text[:middle], budget)
    yield from bounded_chunks(tokenizer, text[middle:], budget)


def dense_vector(result: Any) -> list[float]:
    if isinstance(result, dict):
        vec = result.get("dense_vecs")
        if vec is None:
            vec = result.get("dense_vec")
    else:
        vec = result
    if hasattr(vec, "tolist"):
        vec = vec.tolist()
    if vec and isinstance(vec[0], list):
        vec = vec[0]
    return [float(x) for x in vec]


def embed_one(model: BGEM3FlagModel, text: str, max_length: int) -> list[float]:
    with torch.inference_mode():
        return dense_vector(model.encode([text], batch_size=1, max_length=max_length))


def validate_vector(vec: list[float], dimensions: int) -> None:
    if len(vec) != dimensions:
        raise RuntimeError(f"Embedding dimension mismatch: got {len(vec)}, expected {dimensions}.")
    if not all(math.isfinite(x) for x in vec):
        raise RuntimeError("Embedding contains NaN or Infinity.")


def l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm <= 0 or not math.isfinite(norm):
        raise RuntimeError("Cannot normalize empty or invalid embedding vector.")
    return [float(x / norm) for x in vec]


def embed_document(model: BGEM3FlagModel, tokenizer: Any, text: str, cfg: Config) -> list[float]:
    if not text or not text.strip():
        raise ValueError("vn_text is empty.")
    if len(text) <= 16384 and token_count(tokenizer, text) <= cfg.safe_max_tokens:
        vec = embed_one(model, text, cfg.safe_max_tokens)
        validate_vector(vec, cfg.dimensions)
        return l2_normalize(vec)

    weighted = [0.0] * cfg.dimensions
    total_weight = 0
    for chunk_text, weight in decode_chunks(tokenizer, text, cfg.chunk_tokens):
        if STOP_REQUESTED:
            raise InterruptedError("Shutdown during embedding; checkpoint unchanged.")
        if token_count(tokenizer, chunk_text) > cfg.safe_max_tokens:
            raise RuntimeError("Chunk exceeds model input token limit; refusing truncation.")
        vec = embed_one(model, chunk_text, cfg.safe_max_tokens)
        validate_vector(vec, cfg.dimensions)
        for i, value in enumerate(vec):
            weighted[i] += value * weight
        total_weight += weight
        del vec
    if total_weight <= 0:
        raise ValueError("Document produced no non-empty chunks.")
    pooled = [value / total_weight for value in weighted]
    return l2_normalize(pooled)


def stable_doc_id(record: dict[str, Any], pos: int, namespace: str) -> str:
    for name in ("id", "document_id", "_id", "doc_id"):
        value = record.get(name)
        if value is not None and (isinstance(value, (dict, list, tuple, bool)) or
                                  not isinstance(value, (str, int, float)) or
                                  isinstance(value, float) and not math.isfinite(value) or
                                  isinstance(value, str) and value != "" and not value.strip()):
            raise ValueError(f"Invalid source ID in field {name}")
        if value:
            break
    raw_id = record.get("id") or record.get("document_id") or record.get("_id") or record.get("doc_id")
    if isinstance(raw_id, (dict, list, tuple)):
        raise ValueError("Source document ID must be scalar")
    source = str(raw_id) if raw_id not in (None, "") else f"pos:{pos}"
    digest = hashlib.sha256(f"{namespace}:{source}".encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest[:24]).decode("ascii").rstrip("=")


def block_invalid_id(conn: sqlite3.Connection, record: dict[str, Any], pos: int,
                     namespace: str, revision: str, error: str) -> str:
    internal_id = "blocked_" + hashlib.sha256(f"{namespace}:{revision}:{pos}".encode("utf-8")).hexdigest()
    with conn:
        upsert_status(conn, internal_id, pos, f"dataset_pos:{pos}", "BLOCKED", error=error)
        set_checkpoint(conn, pos, record.get("_pipeline_cursor"))
    logger.warning("Record blocked at pos %d under internal ID %s: %s", pos, internal_id, error)
    return internal_id


def source_document_id(record: dict[str, Any], pos: int) -> str:
    raw_id = record.get("id") or record.get("document_id") or record.get("_id") or record.get("doc_id")
    return str(raw_id) if raw_id not in (None, "") else f"dataset_pos:{pos}"


def parse_date(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return (value if value.tzinfo else value.replace(tzinfo=timezone.utc)).isoformat()
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).isoformat()
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y/%m/%d", "%d-%m-%Y", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc).isoformat()
        except ValueError:
            pass
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()
    except ValueError:
        raise ValueError("Invalid date metadata")


def as_string_list(value: Any) -> list[str] | None:
    if value in (None, ""):
        return None
    if isinstance(value, str) and value.strip().startswith("["):
        value = json.loads(value)
        if not isinstance(value, list):
            raise ValueError("Expected JSON string array")
    if isinstance(value, list):
        items = value
    elif isinstance(value, tuple):
        items = list(value)
    else:
        items = [value]
    if any(isinstance(item, (dict, list, tuple)) for item in items):
        raise ValueError("Nested collection metadata is invalid")
    out = [str(item).strip() for item in items if item not in (None, "") and str(item).strip()]
    return out or None


def coerce_field(field_type: str, value: Any) -> Any:
    if value in (None, ""):
        return None
    if isinstance(value, dict):
        raise ValueError("Dictionary is not scalar metadata")
    if field_type == "Edm.DateTimeOffset":
        return parse_date(value)
    if field_type == "Edm.Int32" or field_type == "Edm.Int64":
        try:
            if isinstance(value, bool) or isinstance(value, float) and (not math.isfinite(value) or not value.is_integer()):
                raise ValueError("Boolean or fractional value is not integer metadata")
            number = int(value)
            bits = 32 if field_type == "Edm.Int32" else 64
            if not -(2**(bits - 1)) <= number < 2**(bits - 1):
                raise ValueError("Integer metadata is out of range")
            return number
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid integer metadata") from exc
    if field_type == "Edm.Double":
        try:
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("Non-finite metadata")
            return number
        except (TypeError, ValueError):
            raise ValueError("Invalid numeric metadata")
    if field_type == "Edm.Boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower().strip() in {"true", "false"}:
            return value.lower().strip() == "true"
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        raise ValueError("Invalid boolean metadata")
    if field_type == "Collection(Edm.String)":
        return as_string_list(value)
    if field_type == "Edm.String":
        if isinstance(value, (list, tuple)):
            raise ValueError("Collection is not string metadata")
        return str(value)
    raise ValueError(f"Unsupported metadata type: {field_type}")


def metadata_payload(record: dict[str, Any], schema: dict[str, str]) -> dict[str, Any]:
    protected = {"id", "document_id", "embedding", "vn_text", "en_text"}
    payload: dict[str, Any] = {}
    aliases = {
        "year": ("year", "nam"),
        "tier": ("tier", "cap_ban_hanh"),
        "issue_date": ("issue_date", "ngay_ban_hanh"),
        "effective_date": ("effective_date", "ngay_hieu_luc"),
        "update_date": ("update_date", "ngay_cap_nhat"),
        "parent_acts": ("parent_acts", "van_ban_goc"),
        "article_titles": ("article_titles", "dieu_khoan"),
        "citations": ("citations", "trich_dan"),
    }
    for field, field_type in schema.items():
        if field in protected:
            continue
        keys = aliases.get(field, (field,))
        raw = next((record[k] for k in keys if k in record), None)
        try:
            coerced = coerce_field(field_type, raw)
        except (ValueError, TypeError) as exc:
            logger.warning("Optional metadata field %s omitted: %s", field, exc)
            continue
        if coerced is not None:
            payload[field] = coerced
    return payload


def build_document(record: dict[str, Any], pos: int, embedding: list[float], schema: dict[str, str], cfg: Config) -> dict[str, Any]:
    doc_id = stable_doc_id(record, pos, cfg.test_namespace)
    doc = {"id": doc_id, "embedding": embedding}
    if "document_id" in schema:
        doc["document_id"] = source_document_id(record, pos)
    doc.update(metadata_payload(record, schema))
    return doc


def checkpoint_pos(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT last_dataset_pos FROM checkpoint WHERE id = 1").fetchone()[0])


def set_checkpoint(conn: sqlite3.Connection, pos: int, cursor: Any = None) -> None:
    conn.execute("UPDATE checkpoint SET last_dataset_pos = ?, updated_at = ?, reader_state=? WHERE id = 1",
                 (pos, utc_now(), json.dumps(cursor) if cursor is not None else None))


def upsert_status(
    conn: sqlite3.Connection,
    doc_id: str,
    pos: int,
    source_id: str,
    status: str,
    error: str | None = None,
    payload: dict[str, Any] | None = None,
    embedding: list[float] | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO documents(doc_id, dataset_pos, source_document_id, status, retry_count, updated_at, last_error, payload_json, embedding_json)
        VALUES(?, ?, ?, ?, 0, ?, ?, ?, ?)
        ON CONFLICT(doc_id) DO UPDATE SET
            dataset_pos=excluded.dataset_pos,
            source_document_id=excluded.source_document_id,
            status=excluded.status,
            updated_at=excluded.updated_at,
            last_error=excluded.last_error,
            payload_json=COALESCE(excluded.payload_json, documents.payload_json),
            embedding_json=COALESCE(excluded.embedding_json, documents.embedding_json)
        WHERE documents.status != 'UPLOADED'
        """,
        (
            doc_id,
            pos,
            source_id,
            status,
            utc_now(),
            error,
            json.dumps(payload, ensure_ascii=False) if payload is not None else None,
            json.dumps(embedding) if embedding is not None else None,
        ),
    )


def mark_uploaded(conn: sqlite3.Connection, doc_id: str) -> None:
    conn.execute(
        "UPDATE documents SET status='UPLOADED', updated_at=?, last_error=NULL, payload_json=NULL, embedding_json=NULL WHERE doc_id=?",
        (utc_now(), doc_id),
    )


def mark_retryable(conn: sqlite3.Connection, doc_id: str, error: str) -> None:
    conn.execute(
        """
        UPDATE documents
        SET status='FAILED_RETRYABLE', retry_count=retry_count+1, updated_at=?, last_error=?
        WHERE doc_id=?
        """,
        (utc_now(), error[:1000], doc_id),
    )


def pending_uploads(conn: sqlite3.Connection, limit: int, after_pos: int = -1, through_pos: int = 9223372036854775807) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT doc_id, payload_json, embedding_json
        FROM documents
        WHERE status IN ('EMBEDDED', 'FAILED_RETRYABLE') AND payload_json IS NOT NULL AND embedding_json IS NOT NULL
        AND dataset_pos > ? AND dataset_pos <= ?
        ORDER BY dataset_pos
        LIMIT ?
        """,
        (after_pos, through_pos, limit),
    ).fetchall()
    docs = []
    for doc_id, payload_json, embedding_json in rows:
        payload = json.loads(payload_json)
        payload["embedding"] = json.loads(embedding_json)
        docs.append(payload)
    return docs


def split_by_payload_size(docs: list[dict[str, Any]], batch_size: int = DEFAULT_UPLOAD_BATCH) -> Iterable[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    size = 32
    for doc in docs:
        doc_size = len(json.dumps({**doc, "@search.action": "mergeOrUpload"}, ensure_ascii=True, allow_nan=False).encode("utf-8")) + 2
        if doc_size + 32 > MAX_UPLOAD_BYTES:
            raise RuntimeError(f"Document {doc['id']} exceeds Azure request size limit; payload preserved.")
        if batch and (len(batch) >= batch_size or size + doc_size > MAX_UPLOAD_BYTES):
            yield batch
            batch, size = [], 32
        batch.append(doc)
        size += doc_size
    if batch:
        yield batch


def retry_delay(attempt: int, exc: Exception | None = None) -> float:
    retry_after = None
    response = getattr(exc, "response", None)
    if response is not None:
        retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return max(float(retry_after), 1.0)
        except ValueError:
            pass
    return min(60.0, (2**attempt) + random.uniform(0.2, 1.5))


def is_retryable_exception(exc: Exception) -> bool:
    if isinstance(exc, (ServiceRequestError, ServiceResponseError)):
        return True
    if isinstance(exc, HttpResponseError):
        return getattr(exc, "status_code", None) in RETRYABLE_STATUS
    return isinstance(exc, (TimeoutError, ConnectionError))


def interruptible_wait(seconds: float) -> None:
    deadline = time.monotonic() + min(seconds, 60.0)
    while not STOP_REQUESTED and time.monotonic() < deadline:
        time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
    if STOP_REQUESTED:
        raise InterruptedError("Shutdown requested during retry.")


def hf_retryable(exc: Exception) -> bool:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is not None:
        return status in RETRYABLE_STATUS
    return isinstance(exc, (TimeoutError, ConnectionError, OSError)) or type(exc).__name__ in {
        "ConnectError", "ReadTimeout", "ConnectTimeout", "ConnectionError", "Timeout", "RemoteProtocolError"}


def fatal_azure(status: Any, error: Any) -> bool:
    code = getattr(error, "code", "") or getattr(getattr(error, "error", None), "code", "")
    text = f"{code} {error}".lower()
    return status in {401, 403, 404} or any(word in text for word in (
        "quota", "storage limit", "index is full", "index full", "maximum storage", "capacity exceeded",
        "invalid dimension", "dimensions", "schema", "invalidapikey", "indexnotfound"))


def upload_documents(conn: sqlite3.Connection, client: SearchClient, docs: list[dict[str, Any]], dry_run: bool) -> int:
    if not docs:
        return 0
    if dry_run:
        logger.info("Dry-run: validated %d documents, upload skipped.", len(docs))
        with conn:
            for doc in docs:
                mark_uploaded(conn, doc["id"])
        return len(docs)

    remaining = docs
    uploaded = 0
    for attempt in range(MAX_UPLOAD_ATTEMPTS):
        if STOP_REQUESTED:
            return uploaded
        try:
            results = list(client.merge_or_upload_documents(remaining))
        except (HttpResponseError, ServiceRequestError, ServiceResponseError, TimeoutError, ConnectionError) as exc:
            with conn:
                for doc in remaining:
                    mark_retryable(conn, doc["id"], str(exc))
            if fatal_azure(getattr(exc, "status_code", None), exc) or not is_retryable_exception(exc):
                raise
            if attempt < MAX_UPLOAD_ATTEMPTS - 1:
                interruptible_wait(retry_delay(attempt, exc))
            continue
        # Match by key and reject duplicate confirmations for that key.
        by_id = {doc["id"]: doc for doc in remaining}
        by_key: dict[str, list[Any]] = {}
        for result in results:
            key = getattr(result, "key", None)
            if key not in by_id:
                logger.error("Azure returned an unknown result key: %r", key)
                continue
            by_key.setdefault(key, []).append(result)
        failed = []
        fatal_error = None
        successes = 0
        with conn:
            for key, doc in by_id.items():
                matches = by_key.get(key, [])
                if len(matches) != 1:
                    mark_retryable(conn, key, "Missing or duplicate Azure result key")
                    failed.append(doc)
                    continue
                result = matches[0]
                if getattr(result, "succeeded", False) is True:
                    mark_uploaded(conn, key)
                    successes += 1
                    continue
                error = str(getattr(result, "error_message", None) or "Azure upload failed")
                status = getattr(result, "status_code", None)
                mark_retryable(conn, key, error)
                if fatal_azure(status, error) or status not in RETRYABLE_STATUS:
                    fatal_error = f"Azure rejected {key}: status={status}, {error}"
                failed.append(doc)
        uploaded += successes
        if fatal_error:
            raise RuntimeError(fatal_error)
        remaining = failed
        if not remaining:
            return uploaded
        if attempt < MAX_UPLOAD_ATTEMPTS - 1:
            interruptible_wait(retry_delay(attempt))
    raise UploadRetriesExhausted(uploaded, len(remaining))


def get_dataset_iter(dataset_id: str, revision: str, start_pos: int = 0) -> Iterable[tuple[int, dict[str, Any]]]:
    next_pos = start_pos
    failures = 0
    while failures < MAX_UPLOAD_ATTEMPTS and not STOP_REQUESTED:
        try:
            dataset = load_dataset(dataset_id, split="train", streaming=True, revision=revision)
            for pos, item in enumerate(dataset):
                if STOP_REQUESTED:
                    return
                if pos < next_pos:
                    continue
                yield pos, dict(item)
                next_pos = pos + 1
            return
        except Exception as exc:
            if not hf_retryable(exc):
                raise
            failures += 1
            if failures >= MAX_UPLOAD_ATTEMPTS:
                raise
            logger.warning("HF stream interrupted at %d; reopening pinned revision: %s", next_pos, exc)
            interruptible_wait(retry_delay(failures))


@contextmanager
def shard_cache(cfg: Config, revision: str, filename: str) -> Iterable[Path]:
    cache_root = ROOT / "hf_cache" / "legal-shard-staging"
    cache_root.mkdir(parents=True, exist_ok=True)
    # Each reader owns only this fresh directory, never shared HF blobs.
    with tempfile.TemporaryDirectory(prefix="reader-", dir=cache_root) as directory:
        private = Path(directory)
        metadata = get_hf_file_metadata(hf_hub_url(cfg.dataset_id, filename, repo_type="dataset", revision=revision))
        size = metadata.size
        if size is None or size < 0:
            raise RuntimeError(f"Shard {filename} has unknown size; cannot guarantee disk safety.")
        minimum = int(cfg.min_free_gb * 1024**3)
        existing_bytes = sum(path.stat().st_size for path in cache_root.rglob("*")
                             if path.is_file() and not path.is_symlink())
        if existing_bytes:
            logger.warning("Existing private staging cache consumes %d bytes; no files will be deleted automatically.", existing_bytes)
        if shutil.disk_usage(ROOT).free < minimum + size * 2:
            raise RuntimeError(f"Insufficient disk space before downloading {filename}.")
        local = hf_hub_download(cfg.dataset_id, filename, repo_type="dataset", revision=revision, cache_dir=private)
        if not Path(local).resolve().is_relative_to(private.resolve()):
            raise RuntimeError("Downloaded shard escaped its private cache directory.")
        used = sum(path.stat().st_size for path in cache_root.rglob("*") if path.is_file() and not path.is_symlink())
        if shutil.disk_usage(ROOT).free < minimum:
            raise RuntimeError("Disk safety threshold exceeded after shard download.")
        yield Path(local)


def dataset_records(conn: sqlite3.Connection, cfg: Config, revision: str) -> Iterable[tuple[int, dict[str, Any]]]:
    saved = conn.execute("SELECT value FROM run_config WHERE key='parquet_manifest'").fetchone()
    if not saved and checkpoint_pos(conn) >= 0:
        logger.warning("Legacy offset checkpoint: retaining original streaming order; resume scans earlier records.")
        yield from get_dataset_iter(cfg.dataset_id, revision, checkpoint_pos(conn) + 1)
        return
    if cfg.dataset_id != "tmquan/thuvienphapluat-vn-vbpl":
        yield from get_dataset_iter(cfg.dataset_id, revision, checkpoint_pos(conn) + 1)
        return
    if saved:
        shards = json.loads(saved[0])
    else:
        shards = None
        for attempt in range(MAX_UPLOAD_ATTEMPTS):
            try:
                files = HfApi().list_repo_files(cfg.dataset_id, repo_type="dataset", revision=revision)
                shards = sorted(name for name in files if name.startswith("documents-") and name.endswith(".parquet"))
                break
            except Exception as exc:
                if not hf_retryable(exc) or attempt == MAX_UPLOAD_ATTEMPTS - 1:
                    raise
                interruptible_wait(retry_delay(attempt))
        if not shards:
            raise RuntimeError("Pinned dataset has no verified documents Parquet shards.")
        with conn:
            conn.execute("INSERT INTO run_config(key,value) VALUES('parquet_manifest',?)", (json.dumps(shards),))
    state_json = conn.execute("SELECT reader_state FROM checkpoint WHERE id=1").fetchone()[0]
    state = json.loads(state_json) if state_json else [0, -1]
    if (not isinstance(state, list) or len(state) != 2 or
            any(type(value) is not int for value in state) or
            not 0 <= state[0] < len(shards) or state[1] < -1):
        raise RuntimeError("Invalid shard checkpoint cursor.")
    if checkpoint_pos(conn) >= 0 and not state_json:
        raise RuntimeError("Shard checkpoint missing reader state; refusing ambiguous resume.")
    for shard_index in range(state[0], len(shards)):
        for attempt in range(MAX_UPLOAD_ATTEMPTS):
            if STOP_REQUESTED:
                return
            try:
                # Restore all three coordinates from the committed checkpoint on every attempt.
                durable_pos, durable_json = conn.execute(
                    "SELECT last_dataset_pos, reader_state FROM checkpoint WHERE id=1").fetchone()
                durable_cursor = json.loads(durable_json) if durable_json else [0, -1]
                if durable_cursor[0] > shard_index:
                    raise RuntimeError("Checkpoint moved beyond the active shard.")
                skip = durable_cursor[1] + 1 if durable_cursor[0] == shard_index else 0
                pos = durable_pos + 1
                check_capacity(conn, cfg)
                with shard_cache(cfg, revision, shards[shard_index]) as local, pq.ParquetFile(local) as parquet:
                    if skip > parquet.metadata.num_rows:
                        raise RuntimeError("Checkpoint row offset exceeds shard row count.")
                    row_base = 0
                    for group in range(parquet.num_row_groups):
                        rows = parquet.metadata.row_group(group).num_rows
                        if row_base + rows <= skip:
                            row_base += rows
                            continue
                        next_row = row_base
                        for batch in parquet.iter_batches(batch_size=PARQUET_READ_BATCH_ROWS, row_groups=[group], use_threads=False):
                            # Only a small Parquet batch is converted to Python objects at once.
                            for offset, record in enumerate(batch.to_pylist()):
                                row = next_row + offset
                                if row < skip:
                                    continue
                                if STOP_REQUESTED:
                                    return
                                record['_pipeline_cursor'] = [shard_index, row]
                                yield pos, record
                                confirmed_pos, confirmed_json = conn.execute(
                                    "SELECT last_dataset_pos, reader_state FROM checkpoint WHERE id=1").fetchone()
                                if confirmed_pos != pos or json.loads(confirmed_json or "null") != [shard_index, row]:
                                    raise RuntimeError("Record was not committed; refusing to advance the shard reader.")
                                pos += 1
                                skip = row + 1
                            next_row += batch.num_rows
                        row_base += rows
                break
            except Exception as exc:
                if not hf_retryable(exc) or attempt == MAX_UPLOAD_ATTEMPTS - 1:
                    raise
                interruptible_wait(retry_delay(attempt))


def log_stats(conn: sqlite3.Connection, stats: RuntimeStats, current_pos: int) -> None:
    counts = dict(conn.execute("SELECT status, COUNT(*) FROM documents GROUP BY status").fetchall())
    logger.info("Read this run=%d | Uploaded total=%d | Pending=%d | Retryable=%d | Blocked=%d | Completed=%d | Checkpoint=%d",
                stats.read, counts.get("UPLOADED", 0), counts.get("EMBEDDED", 0) + counts.get("PENDING", 0),
                counts.get("FAILED_RETRYABLE", 0), counts.get("BLOCKED", 0), counts.get("UPLOADED", 0), current_pos)


def flush_pending_uploads(conn: sqlite3.Connection, client: SearchClient, cfg: Config, stats: RuntimeStats) -> None:
    # A position watermark guarantees each document is attempted once per flush.
    high = conn.execute("SELECT COALESCE(MAX(dataset_pos), -1) FROM documents").fetchone()[0]
    after = -1
    while not STOP_REQUESTED:
        docs = pending_uploads(conn, cfg.upload_batch_size, after, high)
        if not docs:
            return
        after = max(conn.execute("SELECT dataset_pos FROM documents WHERE doc_id=?", (doc["id"],)).fetchone()[0] for doc in docs)
        for batch in split_by_payload_size(docs, cfg.upload_batch_size):
            try:
                stats.uploaded += upload_documents(conn, client, batch, cfg.dry_run)
            except UploadRetriesExhausted as exc:
                stats.uploaded += exc.uploaded
                raise


def ensure_capacity(conn: sqlite3.Connection, client: SearchClient, cfg: Config, stats: RuntimeStats) -> None:
    pending = conn.execute("SELECT COUNT(*) FROM documents WHERE embedding_json IS NOT NULL").fetchone()[0]
    if pending >= MAX_PENDING_VECTORS:
        flush_pending_uploads(conn, client, cfg, stats)
    check_capacity(conn, cfg)


def check_capacity(conn: sqlite3.Connection, cfg: Config) -> None:
    free = shutil.disk_usage(ROOT).free / 1024**3
    if free < cfg.min_free_gb:
        raise RuntimeError(f"Disk space below safety threshold: {free:.2f} GB; checkpoint preserved.")
    pending = conn.execute("SELECT COUNT(*) FROM documents WHERE embedding_json IS NOT NULL").fetchone()[0]
    if pending >= MAX_PENDING_VECTORS:
        raise RuntimeError(f"Upload queue reached {MAX_PENDING_VECTORS} vectors; stopping with payloads preserved.")


def run() -> int:
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    cfg = load_config()
    conn = connect_db(cfg.db_path)
    stats = RuntimeStats(started_at=time.time())
    index_client = search_client = None
    active_reader = None
    try:
        logger.info("Resolving Hugging Face dataset revision.")
        stored = conn.execute("SELECT value FROM run_config WHERE key='dataset_revision'").fetchone()
        revision = json.loads(stored[0]) if stored else get_dataset_revision(cfg.dataset_id)
        if revision == "unknown":
            raise RuntimeError("Legacy checkpoint has no pinned revision; manual compatibility verification required.")
        assert_or_store_run_config(conn, config_fingerprint(cfg, revision))
        forgotten = conn.execute("SELECT dataset_pos FROM documents WHERE status IN ('PENDING','FAILED_RETRYABLE') AND embedding_json IS NULL AND dataset_pos <= ? LIMIT 1",
                                 (checkpoint_pos(conn),)).fetchone()
        if forgotten:
            raise RuntimeError(f"Unembedded legacy record at position {forgotten[0]} is behind checkpoint; manual checkpoint recovery required.")

        index_client, search_client = make_clients(cfg)
        schema = preflight(cfg, index_client)
        flush_pending_uploads(conn, search_client, cfg, stats)
        check_capacity(conn, cfg)

        model = load_model(cfg)
        tokenizer = get_tokenizer(model)
        _ = embed_document(model, tokenizer, "Kiểm tra mô hình embedding.", cfg)

        start_pos = checkpoint_pos(conn) + 1
        processed_new = 0
        dataset_exhausted = False
        stop_reason = "shutdown"
        logger.info("Starting from dataset position %d", start_pos)
        active_reader = dataset_records(conn, cfg, revision)
        bounded_reader = islice(active_reader, cfg.limit) if cfg.limit is not None else active_reader
        for pos, record in bounded_reader:
            if STOP_REQUESTED:
                break
            if pos < start_pos:
                continue
            ensure_capacity(conn, search_client, cfg, stats)

            stats.read += 1
            if stats.read == 1:
                logger.info("Dataset fields detected: %s", ", ".join(sorted(record.keys())))
            try:
                doc_id = stable_doc_id(record, pos, cfg.test_namespace)
                existing = conn.execute("SELECT status, dataset_pos, embedding_json FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
                if existing and existing[1] != pos:
                    # The duplicate source record gets its own BLOCKED audit row.
                    # Never overwrite the original document's embedding/status.
                    raise ValueError(f"Source ID collision at positions {existing[1]} and {pos}")
            except ValueError as exc:
                block_invalid_id(conn, record, pos, cfg.test_namespace, revision, str(exc))
                stats.blocked += 1
                processed_new += 1
                continue
            if existing and (existing[0] in {"UPLOADED", "BLOCKED"} or existing[2] is not None):
                with conn:
                    set_checkpoint(conn, pos, record.get('_pipeline_cursor'))
                processed_new += 1
                continue

            try:
                text = record.get("vn_text")
                if not isinstance(text, str) or not text.strip():
                    raise ValueError("vn_text must be a non-empty string")
                embedding = embed_document(model, tokenizer, text or "", cfg)
                payload = build_document(record, pos, embedding, schema, cfg)
                payload_without_vector = dict(payload)
                payload_without_vector.pop("embedding", None)
                with conn:
                    upsert_status(
                        conn,
                        doc_id,
                        pos,
                        source_document_id(record, pos),
                        "EMBEDDED",
                        payload=payload_without_vector,
                        embedding=embedding,
                    )
                    set_checkpoint(conn, pos, record.get('_pipeline_cursor'))
                stats.embedded += 1
                processed_new += 1
                del text, embedding, payload, payload_without_vector
            except ValueError as exc:
                with conn:
                    upsert_status(conn, doc_id, pos, source_document_id(record, pos), "BLOCKED", error=str(exc))
                    set_checkpoint(conn, pos, record.get('_pipeline_cursor'))
                stats.blocked += 1
                processed_new += 1
                logger.warning("Record blocked at pos %d: %s", pos, exc)
            except (MemoryError, RuntimeError) as exc:
                if not isinstance(exc, MemoryError) and not any(term in str(exc).lower() for term in (
                    "out of memory", "can't allocate memory", "cannot allocate memory", "defaultcpuallocator")):
                    raise
                gc.collect()
                with conn:
                    upsert_status(conn, doc_id, pos, source_document_id(record, pos), "FAILED_RETRYABLE", error="MemoryError while embedding.")
                logger.exception("MemoryError at pos %d; progress was not advanced for this record.", pos)
                raise RuntimeError("Embedding allocation failed; checkpoint preserved for restart.") from exc

            if processed_new % cfg.upload_batch_size == 0:
                flush_pending_uploads(conn, search_client, cfg, stats)

            if stats.read % max(cfg.upload_batch_size, 1) == 0:
                log_stats(conn, stats, pos)
        else:
            dataset_exhausted = not STOP_REQUESTED and (cfg.limit is None or processed_new < cfg.limit)
            stop_reason = "shutdown" if STOP_REQUESTED else ("dataset_exhausted" if dataset_exhausted else "limit_reached")

        active_reader.close()
        active_reader = None

        flush_pending_uploads(conn, search_client, cfg, stats)
        log_stats(conn, stats, checkpoint_pos(conn))
        unfinished = dict(conn.execute("SELECT status, COUNT(*) FROM documents WHERE status != 'UPLOADED' GROUP BY status").fetchall())
        if unfinished:
            logger.warning("Run finished with unfinished records: %s", unfinished)
        logger.info("Stop reason=%s | Dataset exhausted=%s | Records handled this run=%d | Limit=%s",
                    stop_reason, dataset_exhausted, processed_new, cfg.limit)
        if dataset_exhausted and not unfinished and not cfg.dry_run:
            logger.info("Entire dataset processed and all checkpointed records uploaded.")
        elif not unfinished:
            logger.info("Checkpointed records handled; this does not confirm a full dataset upload (dry_run=%s).", cfg.dry_run)
        return 130 if STOP_REQUESTED else (2 if unfinished else 0)
    except InterruptedError:
        logger.warning("Stopped safely; unconfirmed payloads retained.")
        return 130
    finally:
        try:
            if active_reader is not None:
                active_reader.close()
            conn.rollback()
            log_stats(conn, stats, checkpoint_pos(conn))
        finally:
            conn.close()
            for client in (search_client, index_client):
                if client is not None:
                    try:
                        client.close()
                    except Exception:
                        logger.exception("Azure client cleanup failed.")


if __name__ == "__main__":
    try:
        raise SystemExit(run())
    except KeyboardInterrupt:
        logger.warning("Interrupted.")
        raise SystemExit(130)
    except Exception as exc:
        logger.exception("Fatal error: %s", exc)
        raise SystemExit(1)
