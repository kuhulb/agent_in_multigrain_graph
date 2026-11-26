"""Small shared file/SQLite/config/request helpers. No model is imported here."""

from __future__ import annotations

import contextlib
from collections import deque
from concurrent.futures import FIRST_COMPLETED, wait
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import time
from threading import Condition
from urllib.parse import urlsplit
import uuid

VERSION = 3
ALGORITHM = "multigrain-v3.1"
AGENT_ALGORITHM = "multigrain-agent-v3.9-required"
TOKEN_FIELDS = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cached_input_tokens",
    "uncached_input_tokens",
    "reasoning_tokens",
)
REMOTE_API_PATH = Path(__file__).resolve().parent / "private/remote_api.env"
_SECRET_VALUES = set()


def redact(value):
    if isinstance(value, dict):
        return {
            k: (
                "<redacted>" if k.endswith("API_KEY") or k.lower() == "authorization" else redact(v)
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact(x) for x in value]
    if isinstance(value, tuple):
        return [redact(x) for x in value]
    if isinstance(value, str):
        for token in sorted(_SECRET_VALUES, key=len, reverse=True):
            if token:
                value = value.replace(token, "<redacted>")
    return value


def dumps(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )


def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@contextlib.contextmanager
def atomic_file(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with open(temp, "wb") as f:
            yield f
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        temp.unlink(missing_ok=True)


def atomic_bytes(path, content):
    with atomic_file(path) as stream:
        stream.write(content)


def write_json(path, value):
    atomic_bytes(path, (dumps(value) + "\n").encode())


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_jsonl(path, rows):
    with atomic_file(path) as stream:
        for row in rows:
            stream.write(dumps(row).encode("utf-8") + b"\n")


def load_env(path, model_stage=None):
    from dotenv import dotenv_values

    path = Path(path).expanduser().resolve()
    if "private" in path.parts or path.name in {"api_keys.env", "remote_api.env"}:
        raise ValueError("Choose a public experiment configuration")
    values = dict(dotenv_values(path, interpolate=False))
    model_path = Path(values.get("MODEL_CONFIG_FILE") or "models.env").expanduser()
    if not model_path.is_absolute():
        model_path = path.parent / model_path
    model_path = model_path.resolve()
    if "private" in model_path.parts or model_path.name in {"api_keys.env", "remote_api.env"}:
        raise ValueError("Choose a public model configuration")
    if model_path.exists():
        models = dict(dotenv_values(model_path, interpolate=False))
        overlap = set(models) & set(values)
        if overlap:
            raise ValueError(
                "Duplicate settings between experiment.env and models.env: "
                + ",".join(sorted(overlap))
            )
        values.update(models)
    # Ignore the retired public credential path. It cannot redirect the private profile.
    values.pop("API_KEYS_FILE", None)
    if any(
        k.startswith(("AGENT_API_", "GRAPH_API_"))
        or k.endswith(("API_KEY", "API_ALLOWED_BASE_URL"))
        for k in values
    ):
        raise ValueError("Remote API settings are forbidden in public configuration")
    for prefix in ("GRAPH", "AGENT"):
        mode = values.get(prefix + "_MODEL_MODE", "local")
        if mode not in {"local", "api"}:
            raise ValueError(f"{prefix}_MODEL_MODE must be local or api")
        base = values.get(prefix + "_LLM_BASE_URL")
        if base:
            url = urlsplit(base)
            if (
                url.scheme not in {"http", "https"}
                or url.hostname not in {"localhost", "127.0.0.1", "::1"}
                or url.username
                or url.password
                or url.query
                or url.fragment
            ):
                raise ValueError(
                    "Remote destinations and credentials are forbidden in public local model URLs"
                )
    values["_base"] = str(path.parent)
    values["_env"] = str(path)
    values = {k: (v if v is not None else "") for k, v in values.items()}
    if model_stage and values.get(model_stage + "_MODEL_MODE", "local") == "api":
        load_remote_model(values, model_stage)
    return values


def load_remote_model(c, prefix):
    from dotenv import dotenv_values

    profile_path = REMOTE_API_PATH
    if not profile_path.is_file():
        raise ConfigFailure("Create private/remote_api.env from remote_api.env.example")
    if profile_path.is_symlink() or profile_path.stat().st_mode & 0o077:
        raise ConfigFailure("private/remote_api.env must be a regular private file (chmod 600)")
    profile = dict(dotenv_values(profile_path, interpolate=False))
    section = prefix + "_API_"
    private = {k[len(section) :]: v for k, v in profile.items() if k.startswith(section) and v}
    required = ["BASE_URL", "MODEL", "KEY"]
    if prefix == "AGENT":
        required.append("CONTEXT_WINDOW_TOKENS")
    for field in required:
        if field not in private:
            raise ConfigFailure(f"Missing {section}{field} in private/remote_api.env")
    try:
        url = urlsplit(private["BASE_URL"])
        _ = url.port
    except ValueError:
        raise ConfigFailure("Invalid HTTPS Base URL in private/remote_api.env") from None
    if (
        url.scheme != "https"
        or not url.hostname
        or url.username
        or url.password
        or url.query
        or url.fragment
        or url.path.rstrip("/").endswith("/chat/completions")
    ):
        raise ConfigFailure(
            "Remote API needs a clean HTTPS Base URL without the /chat/completions suffix"
        )
    adapter = private.get("ADAPTER", "deepseek")
    if adapter not in {"deepseek", "openai_compatible"}:
        raise ConfigFailure("Remote adapter must be deepseek or openai_compatible")
    choice = private.get("TOOL_CHOICE", "auto")
    if choice not in {"auto", "required"}:
        raise ConfigFailure("Remote tool choice must be auto or required")
    if private.get("ENABLE_THINKING", "false") not in {"true", "false"}:
        raise ConfigFailure("ENABLE_THINKING must be true or false")
    thinking = private.get("ENABLE_THINKING", "false") == "true"
    control = private.get(
        "THINKING_CONTROL", "deepseek" if adapter == "deepseek" else "non_thinking_model"
    )
    if control not in {"deepseek", "non_thinking_model", "reasoning_effort_none"}:
        raise ConfigFailure("Unsupported remote thinking control")
    if thinking and (adapter != "deepseek" or control != "deepseek" or choice != "auto"):
        raise ConfigFailure("Remote thinking requires the DeepSeek adapter and auto tool choice")
    _SECRET_VALUES.update(
        (private["KEY"], private["MODEL"], private["BASE_URL"].rstrip("/"), url.hostname)
    )
    c["_remote"] = {
        "stage": prefix,
        "adapter": adapter,
        "base_url": private["BASE_URL"].rstrip("/"),
        "key": private["KEY"],
    }
    c["_api_tool_choice"] = choice
    c["_api_rpm"] = integer(private, "RPM", 0)
    c["_api_tpm"] = integer(private, "TPM", 0)
    overrides = {
        "LLM_MODEL": "MODEL",
        "MODEL_REVISION": "REVISION",
        "MAX_OUTPUT_TOKENS": "MAX_OUTPUT_TOKENS",
        "TEMPERATURE": "TEMPERATURE",
        "REGENERATE_TEMPERATURE": "REGENERATE_TEMPERATURE",
        "REQUEST_TIMEOUT_SECONDS": "REQUEST_TIMEOUT_SECONDS",
        "MAX_HTTP_ATTEMPTS": "MAX_HTTP_ATTEMPTS",
    }
    if prefix == "AGENT":
        overrides.update(
            {
                "CONTEXT_WINDOW_TOKENS": "CONTEXT_WINDOW_TOKENS",
                "CONTEXT_SAFETY_TOKENS": "CONTEXT_SAFETY_TOKENS",
                "TOKENIZER_PATH": "TOKENIZER_PATH",
                "CHAT_TEMPLATE_FILE": "CHAT_TEMPLATE_FILE",
            }
        )
    defaults = {
        "MAX_OUTPUT_TOKENS": "2048" if prefix == "AGENT" else "220",
        "TEMPERATURE": "0" if prefix == "AGENT" else "0.2",
        "REGENERATE_TEMPERATURE": "0.2",
        "REQUEST_TIMEOUT_SECONDS": "240",
        "MAX_HTTP_ATTEMPTS": "5",
        "CONTEXT_SAFETY_TOKENS": "1024",
    }
    hidden = {
        prefix + "_" + name: private.get(field, defaults.get(field, ""))
        for name, field in overrides.items()
    }
    for field in ("TOKENIZER_PATH", "CHAT_TEMPLATE_FILE"):
        value = hidden.get(prefix + "_" + field)
        if value:
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = profile_path.parent.parent / path
            hidden[prefix + "_" + field] = str(path)
            _SECRET_VALUES.add(str(path))
    hidden.update(
        {prefix + "_ENABLE_THINKING": str(thinking).lower(), prefix + "_THINKING_CONTROL": control}
    )
    c.update(hidden)
    c["_hidden_model_keys"] = set(hidden) | {prefix + "_LLM_BASE_URL"}
    c["_remote_identity"] = digest(
        {
            "protocol": "chat-completions-v1",
            "endpoint": c["_remote"]["base_url"],
            "adapter": adapter,
            "tool_choice": choice,
            "settings": {
                k: v
                for k, v in hidden.items()
                if not k.endswith(("TIMEOUT_SECONDS", "MAX_HTTP_ATTEMPTS"))
            },
        }
    )


def integer(c, key, default=None, minimum=0):
    value = c.get(key, "")
    if value == "":
        if default is None:
            raise ValueError(f"{key} is required")
        return default
    result = int(value)
    if str(result) != str(value).strip() or result < minimum:
        raise ValueError(f"Invalid {key}: {value}")
    return result


def boolean(c, key, default=False):
    value = c.get(key, str(default).lower())
    if value not in ("true", "false"):
        raise ValueError(f"{key}: expected true/false")
    return value == "true"


def csv_values(c, key, default=""):
    values = [s.strip() for s in c.get(key, default).split(",") if s.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {key}")
    return values


def config_path(c, key, default=""):
    value = c.get(key, "") or default
    if not value:
        raise ValueError(f"{key} is required")
    p = Path(value).expanduser()
    # Preserve venv interpreter symlinks: resolving bin/python changes sys.prefix
    # to the base interpreter and silently drops the venv's dependencies.
    return Path(os.path.abspath(p if p.is_absolute() else Path(c["_base"]) / p))


def stage_fingerprint(c, stage):
    # Runtime-only edits may repair a deployment without changing research conditions.
    runtime_suffixes = (
        "API_KEY",
        "API_ALLOWED_BASE_URL",
        "BASE_URL",
        "TIMEOUT_SECONDS",
        "MAX_HTTP_ATTEMPTS",
        "CONCURRENCY",
        "BATCH_SIZE",
        "WORKERS",
    )
    prefixes = ("GRAPH_", "EMBED_") if stage == "build" else ("AGENT_",)
    selected = {
        k: v
        for k, v in c.items()
        if k.startswith(prefixes)
        and not k.endswith(runtime_suffixes)
        and not (k.endswith("_MODEL_MODE") and v == "local")
        and not (k == "AGENT_TARGET_PLAN" and not v)
        and k not in {"AGENT_MAX_CENTER_QUERIES", "AGENT_MAX_DISTINCT_COMMUNITIES", "AGENT_SHOW_SIMILARITY"}
        and not (
            c.get("AGENT_TARGET_PLAN")
            and k in {"AGENT_TARGET_SPLIT", "AGENT_TARGET_LIMIT", "AGENT_TARGET_SEED"}
        )
        and k not in c.get("_hidden_model_keys", ())
    }
    if c.get("_remote", {}).get("stage") == ("GRAPH" if stage == "build" else "AGENT"):
        selected["remote_profile"] = c["_remote_identity"]
    for k in ("DATASET",) if stage == "build" else ():
        selected[k] = c.get(k, "")
    selected["algorithm"] = ALGORITHM if stage == "build" else AGENT_ALGORITHM
    return digest(selected)


def public_config(c):
    return redact(
        {
            k: v
            for k, v in c.items()
            if not k.startswith("_") and k not in c.get("_hidden_model_keys", ())
        }
    )


def record_launch(folder, c, stage):
    import sys

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    entry = {
        "stage": stage,
        "time": time.time(),
        "pid": os.getpid(),
        "python": sys.version,
        "config": public_config(c),
    }
    with open(folder / "launches.jsonl", "a", encoding="utf-8") as stream:
        stream.write(dumps(redact(entry)) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


@contextlib.contextmanager
def process_lock(folder):
    import fcntl

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / ".process.lock", "a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise StageBusy(f"Another process owns {folder}") from e
        yield


class StageBusy(RuntimeError):
    pass


class Store:
    def __init__(self, path, initialize=True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30)
        self.db.row_factory = sqlite3.Row
        if initialize:
            self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        if initialize:
            self.db.executescript("""
          CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS tasks(key TEXT PRIMARY KEY,status TEXT NOT NULL,data TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS requests(id TEXT PRIMARY KEY,owner TEXT NOT NULL,step INTEGER NOT NULL,
            generation INTEGER NOT NULL,status TEXT NOT NULL,payload TEXT NOT NULL,response TEXT,usage TEXT,
            error TEXT,latency REAL,rejected INTEGER NOT NULL DEFAULT 0,
            recovery_index INTEGER NOT NULL DEFAULT 0, transport_index INTEGER NOT NULL DEFAULT 0);
          CREATE INDEX IF NOT EXISTS request_lookup ON requests(owner,step,generation);
          CREATE TABLE IF NOT EXISTS events(key TEXT PRIMARY KEY,data TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS failures(id TEXT PRIMARY KEY,data TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS artifacts(key TEXT PRIMARY KEY,data TEXT NOT NULL);
        """)
            self.db.commit()
        self.request_metadata = any(
            r["name"] == "recovery_index" for r in self.db.execute("PRAGMA table_info(requests)")
        )

    @contextlib.contextmanager
    def tx(self):
        with self.db:
            # Acquire the write lock before checking state, avoiding WAL read-to-write races.
            self.db.execute("BEGIN IMMEDIATE")
            yield self.db
            n = int(self.meta("commit_seq", 0)) + 1
            self.db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", ("commit_seq", dumps(n)))

    def meta(self, key, default=None):
        row = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_meta(self, key, value):
        with self.tx() as db:
            db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (key, dumps(value)))

    def pin(self, key, value):
        old = self.meta(key)
        if old is not None and old != value:
            raise ValueError(f"Identity conflict for {key}; use a new stage ID")
        if old is None:
            self.set_meta(key, value)

    def task(self, key):
        row = self.db.execute("SELECT data FROM tasks WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def save_task(self, key, value):
        with self.tx() as db:
            db.execute(
                "INSERT OR REPLACE INTO tasks VALUES(?,?,?)", (key, value["status"], dumps(value))
            )

    def task_statuses(self):
        return dict(self.db.execute("SELECT key,status FROM tasks ORDER BY rowid"))

    def artifact(self, key):
        r = self.db.execute("SELECT data FROM artifacts WHERE key=?", (key,)).fetchone()
        return json.loads(r[0]) if r else None

    def save_artifact(self, key, value):
        with self.tx() as db:
            db.execute("INSERT OR REPLACE INTO artifacts VALUES(?,?)", (key, dumps(value)))

    def failure(self, **data):
        key = uuid.uuid4().hex
        data.update(failure_id=key, time=time.time())
        data = redact(data)
        with self.tx() as db:
            db.execute("INSERT INTO failures VALUES(?,?)", (key, dumps(data)))
        return key

    def commit_step(self, key, task, event):
        operation = f"{key}:{event['step_id']}"
        with self.tx() as db:
            if db.execute("SELECT 1 FROM events WHERE key=?", (operation,)).fetchone():
                return False
            db.execute("INSERT INTO events VALUES(?,?)", (operation, dumps(event)))
            db.execute(
                "INSERT OR REPLACE INTO tasks VALUES(?,?,?)", (key, task["status"], dumps(task))
            )
        return True

    def recover_sent(self):
        rows = self.db.execute(
            "SELECT id,owner,step,generation FROM requests WHERE status='sent'"
        ).fetchall()
        for r in rows:
            with self.tx() as db:
                db.execute(
                    "UPDATE requests SET status='outcome_unknown',error=? WHERE id=?",
                    ("Interrupted before response persisted", r["id"]),
                )
            self.failure(
                owner=r["owner"],
                step_id=r["step"],
                generation_index=r["generation"],
                attempt_id=r["id"],
                code="OUTCOME_UNKNOWN",
                exposed=False,
            )

    def export(self, folder):
        folder = Path(folder)

        def request_rows():
            for row in self.db.execute("SELECT * FROM requests ORDER BY rowid"):
                entry = dict(row)
                for key in ("payload", "response", "usage"):
                    entry[key] = json.loads(entry[key]) if entry[key] else None
                if "transport_index" in entry:
                    entry["request_type"] = (
                        "transport_retry" if entry["transport_index"]
                        else "protocol_repair" if entry["generation"] else "decision"
                    )
                    if entry["owner"].startswith("capability:"):
                        entry["request_type"] = "capability_" + entry["request_type"]
                entry["usage_missing"] = not any(
                    (entry["usage"] or {}).get(key) is not None for key in TOKEN_FIELDS
                )
                yield entry

        write_jsonl(folder / "requests.jsonl", request_rows())
        for table, name in (("failures", "failures.jsonl"), ("events", "trajectory.jsonl")):
            rows = self.db.execute(f"SELECT data FROM {table} ORDER BY rowid")
            write_jsonl(folder / name, (json.loads(row["data"]) for row in rows))
        write_jsonl(
            folder / "results.jsonl",
            (
                result_for_task(self.db, row["key"], json.loads(row["data"]))
                for row in self.db.execute("SELECT key,data FROM tasks ORDER BY rowid")
            ),
        )


def result_for_task(db, key, task, include_trace=True):
    result = dict(task, task_key=key, usage=usage_for(db, key))
    state = task.get("state", {})
    for field in (
        "messages",
        "coarse_checkpoint",
        "first_fine_request_round",
        "first_new_fine_round",
        "fine_access_reason",
        "coarse_evidence_delivered",
    ):
        if field in state:
            result[field] = state[field]
    counts = dict(state.get("counts", {}))
    rows = db.execute(
        """SELECT step,generation,rejected,latency,
                  json_array_length(response,'$.choices[0].message.tool_calls') AS tool_calls
           FROM requests WHERE owner=? AND status <> 'prepared'""", (key,)
    ).fetchall()
    counts.update(
        request_attempts=len(rows),
        model_generations=len({(r["step"], r["generation"]) for r in rows}),
        rolled_back_generations=len({(r["step"], r["generation"]) for r in rows if r["rejected"]}),
        distinct_communities=len(state.get("opened_communities", [])),
        submit_calls=int(task["status"] == "submitted"),
        distinct_fine_centers=len(state.get("queried_centers", {}).get("fine", [])),
        distinct_coarse_centers=len(state.get("queried_centers", {}).get("coarse", [])),
    )
    if "sample_runs" in task or "invalid_queries" in counts:
        generations = {(r["step"], r["generation"]) for r in rows}
        coarse_centers = set(state.get("queried_centers", {}).get("coarse", []))
        counts.update(
            api_attempts=len(rows),
            api_retries=len(rows) - len(generations),
            repair_attempts=sum(generation > 0 for _, generation in generations),
            tool_calls=sum(r["tool_calls"] or 0 for r in rows),
            logical_queries=counts.get("graph_tool_calls", 0),
            sample_recoveries=max(0, task.get("sample_runs", 0) - 1),
            distinct_accessed_coarse_nodes=len(coarse_centers.union(state.get("opened_communities", []))),
            distinct_accessed_fine_nodes=counts["distinct_fine_centers"],
        )
        latencies = [r["latency"] for r in rows if r["latency"] is not None]
        result["reported_request_latency_seconds"] = sum(latencies) if latencies else None
    result["counts"] = counts
    if task["status"] == "submitted" and "evidence_ids" in task and "first_seen_step" in state:
        refs = task["evidence_ids"]
        home = state.get("home_id")
        result["citations"] = {
            "target": f"F:{task['target_id']}" in refs,
            "initial_community": home is not None and f"C:{home}" in refs,
            "retrieved": any(state["first_seen_step"].get(r, 0) > 0 for r in refs),
        }
    if not include_trace:
        for field in ("state", "active", "messages"):
            result.pop(field, None)
    return result


def usage_for(db, owner=None, *, capability_only=False):
    sql = """SELECT status,usage,rejected,
                    json_extract(response, '$.choices[0].finish_reason') AS finish_reason,
                    json_type(response, '$.choices[0].message') AS message_type,
                    json_extract(response, '$.choices[0].message.content') AS content,
                    json_type(response, '$.choices[0].message.content') AS content_type
             FROM requests WHERE status <> 'prepared'"""
    if capability_only:
        rows = db.execute(sql + " AND owner LIKE 'capability:%'").fetchall()
    elif owner is not None:
        rows = db.execute(sql + " AND owner=?", (owner,)).fetchall()
    else:
        rows = db.execute(sql).fetchall()
    sums = dict.fromkeys(TOKEN_FIELDS, 0)
    known = dict.fromkeys(TOKEN_FIELDS, 0)
    paired_input = paired_cache = cache_known = cache_positive = 0
    usage_missing = usage_invalid = 0
    rejected_attempts = rejected_total_known = rejected_total = 0
    content_observed = content_nonempty = content_chars = content_tokens_known = content_tokens = 0
    max_output = None
    for row in rows:
        u = json.loads(row["usage"]) if row["usage"] else {}
        # Earlier databases did not normalize total/reasoning and could drop partial token data.
        if u and "total_tokens" not in u:
            legacy = normalize_usage(u.get("raw"))
            legacy.update({k: u[k] for k in TOKEN_FIELDS if u.get(k) is not None})
            if legacy["total_tokens"] is None and all(
                legacy[k] is not None for k in ("input_tokens", "output_tokens")
            ):
                legacy["total_tokens"] = legacy["input_tokens"] + legacy["output_tokens"]
            u = legacy
        content_observed += row["message_type"] == "object"
        if row["content_type"] == "text" and row["content"]:
            content_nonempty += 1
            content_chars += len(row["content"])
            raw_usage = u.get("raw")
            if not isinstance(raw_usage, dict):
                raw_usage = {}
            details = raw_usage.get("completion_tokens_details")
            plain_tokens = details.get("content_tokens") if isinstance(details, dict) else None
            if plain_tokens is None:
                plain_tokens = raw_usage.get("content_tokens")
            if (
                type(plain_tokens) is int and plain_tokens >= 0
                and (u.get("output_tokens") is None or plain_tokens <= u["output_tokens"])
            ):
                content_tokens_known += 1
                content_tokens += plain_tokens
        usage_missing += not any(u.get(k) is not None for k in TOKEN_FIELDS)
        usage_invalid += bool(u.get("issues"))
        if row["rejected"]:
            rejected_attempts += 1
            if u.get("total_tokens") is not None:
                rejected_total_known += 1
                rejected_total += u["total_tokens"]
        if u.get("input_tokens") is not None and u.get("cached_input_tokens") is not None:
            paired_input += u["input_tokens"]
            paired_cache += u["cached_input_tokens"]
            cache_known += 1
            cache_positive += u["cached_input_tokens"] > 0
        for k in TOKEN_FIELDS:
            if u.get(k) is not None:
                sums[k] += u[k]
                known[k] += 1
        if u.get("output_tokens") is not None:
            max_output = max(max_output or 0, u["output_tokens"])
    return {
        **{k: sums[k] if known[k] else None for k in TOKEN_FIELDS},
        "attempts": len(rows),
        "known_counts": known,
        "complete": {k: bool(rows) and known[k] == len(rows) for k in TOKEN_FIELDS},
        "outcome_unknown": sum(r["status"] in {"outcome_unknown", "sent"} for r in rows),
        "pending_requests": sum(r["status"] == "sent" for r in rows),
        "usage_missing_requests": usage_missing,
        "usage_invalid_requests": usage_invalid,
        "rejected_attempts": rejected_attempts,
        "rejected_total_tokens_known_requests": rejected_total_known,
        "known_rejected_total_tokens": rejected_total
        if rejected_total_known or not rejected_attempts else None,
        "finish_reason_known_requests": sum(r["finish_reason"] is not None for r in rows),
        "finish_length_requests": sum(r["finish_reason"] == "length" for r in rows),
        "content_observed_requests": content_observed,
        "content_nonempty_requests": content_nonempty,
        "content_nonempty_ratio": content_nonempty / content_observed if content_observed else None,
        "content_chars": content_chars,
        "content_tokens_known_requests": content_tokens_known,
        "known_content_tokens": content_tokens if content_tokens_known else None,
        "max_output_tokens": max_output,
        "cache_pair_input_tokens": paired_input,
        "cache_pair_cached_tokens": paired_cache,
        "cache_known_requests": cache_known,
        "cache_positive_requests": cache_positive,
    }


def normalize_usage(raw):
    issues = []

    def fields(value, name):
        if isinstance(value, dict):
            return value
        if value is not None:
            issues.append(f"Invalid usage object: {name}")
        return {}

    def count(obj, key):
        value = obj.get(key)
        if value is None or (type(value) is int and value >= 0):
            return value
        issues.append(f"Invalid token count: {key}")
        return None

    def first(*values):
        return next((v for v in values if v is not None), None)

    u = fields(raw, "usage")
    prompt = fields(u.get("prompt_tokens_details"), "prompt_tokens_details")
    completion = fields(u.get("completion_tokens_details"), "completion_tokens_details")
    i = first(count(u, "prompt_tokens"), count(u, "input_tokens"))
    o = first(count(u, "completion_tokens"), count(u, "output_tokens"))
    total = count(u, "total_tokens")
    cache = count(prompt, "cached_tokens")
    hit = count(u, "prompt_cache_hit_tokens")
    miss = count(u, "prompt_cache_miss_tokens")
    reasoning = first(count(completion, "reasoning_tokens"), count(u, "reasoning_tokens"))
    if total is not None and any(t is not None and t > total for t in (i, o)):
        issues.append("Total tokens less than a reported component")
        total = None
    if i is not None and o is not None:
        if total is not None and total != i + o:
            issues.append("Total tokens inconsistent with reported input and output")
        total = i + o
    if cache is not None and hit is not None and cache != hit:
        issues.append("Conflicting provider cache-hit counts")
        cache = None
    else:
        cache = first(cache, hit)
    if i is None and cache is not None and miss is not None:
        i = cache + miss
        if total is not None and (i > total or (o is not None and i + o != total)):
            issues.append("Cache-derived input tokens inconsistent with total")
            i = None
        elif total is None and o is not None:
            total = i + o
    if i is not None:
        if cache is not None and cache > i:
            issues.append("Cache-hit tokens exceed input tokens")
            cache = None
        if miss is not None and miss > i:
            issues.append("Cache-miss tokens exceed input tokens")
            miss = None
        if cache is not None and miss is not None and cache + miss != i:
            issues.append("Cache-hit and cache-miss tokens do not sum to input tokens")
            cache = miss = None
        elif cache is not None:
            miss = i - cache
        elif miss is not None:
            cache = i - miss
    if o is not None and reasoning is not None and reasoning > o:
        issues.append("Reasoning tokens exceed output tokens")
        reasoning = None
    return {
        "input_tokens": i,
        "output_tokens": o,
        "total_tokens": total,
        "cached_input_tokens": cache,
        "uncached_input_tokens": miss,
        "reasoning_tokens": reasoning,
        "raw": raw,
        "issues": issues,
    }


class RetryLater(Exception):
    pass


class ConfigFailure(Exception):
    def __init__(self, message):
        super().__init__(redact(str(message)))


class ContextFailure(ConfigFailure):
    pass


class ApiRateLimiter:
    """Shared rolling quota and 429 cool-down for one model stage in this process."""

    def __init__(self, rpm=0, tpm=0):
        self.rpm, self.tpm = rpm, tpm
        self.history = deque()
        self.tokens = 0
        self.cooldown_until = 0.0
        self.condition = Condition()

    def acquire(self, tokens):
        if self.tpm and tokens > self.tpm:
            raise ConfigFailure("Estimated request size exceeds the configured minute token quota")
        with self.condition:
            while True:
                now = time.monotonic()
                while self.history and self.history[0][0] <= now - 60:
                    self.tokens -= self.history.popleft()[1]
                delay = self.cooldown_until - now
                if (self.rpm and len(self.history) >= self.rpm) or (
                    self.tpm and self.tokens + tokens > self.tpm
                ):
                    delay = max(delay, self.history[0][0] + 60 - now)
                if delay <= 0:
                    if self.rpm or self.tpm:
                        self.history.append((now, tokens))
                        self.tokens += tokens
                    return
                self.condition.wait(delay)

    def pause(self, seconds):
        with self.condition:
            self.cooldown_until = max(self.cooldown_until, time.monotonic() + seconds)
            self.condition.notify_all()


def worker_count(c, prefix):
    if c.get(prefix + "_MODEL_MODE", "local") != "api":
        return 1
    return integer(c, prefix + "_MODEL_WORKERS", 6, 1)


@contextlib.contextmanager
def api_runtime(c, prefix):
    if c.get(prefix + "_MODEL_MODE", "local") != "api":
        yield c
        return
    import httpx

    with httpx.Client(
        trust_env=False, limits=httpx.Limits(max_connections=worker_count(c, prefix))
    ) as client:
        yield dict(
            c,
            _http_client=client,
            _api_limiter=ApiRateLimiter(c.get("_api_rpm", 0), c.get("_api_tpm", 0)),
        )


def bounded_jobs(executor, items, work, workers):
    """At most workers in flight. Yield results as they finish."""
    todo = iter(items)
    inflight = set()
    for _ in range(workers):
        try:
            item = next(todo)
        except StopIteration:
            break
        inflight.add(executor.submit(work, item))
    while inflight:
        done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
        for future in done:
            yield future.result()
            try:
                item = next(todo)
            except StopIteration:
                continue
            inflight.add(executor.submit(work, item))


def retry_delay(response, fallback, limit):
    from email.utils import parsedate_to_datetime

    value = response.headers.get("Retry-After")
    if value:
        try:
            seconds = float(value)
        except ValueError:
            try:
                seconds = parsedate_to_datetime(value).timestamp() - time.time()
            except (ValueError, TypeError, OverflowError):
                seconds = fallback
        if math.isfinite(seconds):
            return min(max(0, seconds), limit)
    return fallback


def model_connection(c, prefix):
    if c.get(prefix + "_MODEL_MODE", "local") == "api":
        remote = c.get("_remote", {})
        if remote.get("stage") != prefix:
            raise ConfigFailure(f"{prefix} private API profile was not loaded")
        return remote["base_url"], {"Authorization": "Bearer " + remote["key"]}
    return c[prefix + "_LLM_BASE_URL"].rstrip("/"), {}


def configure_agent_low_thinking(c):
    remote = c.get("_remote", {})
    if (
        c.get("AGENT_MODEL_MODE") != "api" or remote.get("stage") != "AGENT"
        or remote.get("adapter") != "deepseek" or c.get("AGENT_THINKING_CONTROL") != "deepseek"
        or c.get("_api_tool_choice") != "auto"
    ):
        raise ConfigFailure("Low Thinking requires the DeepSeek API adapter and auto tool choice")
    c["AGENT_ENABLE_THINKING"] = "true"
    c["AGENT_REASONING_EFFORT"] = "low"


def thinking_params(c, prefix):
    enabled = boolean(c, prefix + "_ENABLE_THINKING")
    control = c.get(prefix + "_THINKING_CONTROL", "chat_template")
    if control == "chat_template":
        return {"chat_template_kwargs": {"enable_thinking": enabled}}
    if control == "deepseek":
        params = {"thinking": {"type": "enabled" if enabled else "disabled"}}
        if prefix == "AGENT" and enabled and c.get("AGENT_REASONING_EFFORT") == "low":
            params["reasoning_effort"] = "low"
        return params
    if control == "non_thinking_model" and not enabled:
        return {}
    if control == "reasoning_effort_none" and not enabled:
        return {"reasoning_effort": "none"}
    raise ConfigFailure(f"Unsupported {prefix} thinking control")


def verify_deployment(store, c, prefix):
    import httpx

    base, headers = model_connection(c, prefix)
    if headers:
        identity = {"mode": "api", "profile_sha256": c["_remote_identity"]}
        store.pin(prefix + "_deployment", identity)
        return identity  # Relays can expose completions without a model-list endpoint.
    for attempt in range(integer(c, prefix + "_MAX_HTTP_ATTEMPTS", 5, 1)):
        try:
            response = httpx.get(
                base + "/models",
                timeout=20,
                headers=headers,
                follow_redirects=False,
                trust_env=not bool(headers),
            )
            response.raise_for_status()
            break
        except httpx.HTTPError as e:
            store.failure(
                owner="capability:" + prefix,
                code="DEPLOYMENT_UNAVAILABLE",
                message=str(e),
                exposed=False,
            )
            if attempt + 1 == integer(c, prefix + "_MAX_HTTP_ATTEMPTS", 5, 1):
                raise RetryLater("Deployment metadata unavailable") from e
            time.sleep(min(2**attempt, integer(c, "RETRY_MAX_WAIT_SECONDS", 60, 1)))
    models = response.json()["data"]
    model = next((x for x in models if x["id"] == c[prefix + "_LLM_MODEL"]), None)
    if model is None:
        raise ConfigFailure("Configured model is not served")
    if model.get("revision") and model["revision"] != c[prefix + "_MODEL_REVISION"]:
        raise ConfigFailure("Model weight revision mismatch")
    if model.get("thinking") is True and not boolean(c, prefix + "_ENABLE_THINKING"):
        raise ConfigFailure("Thinking is enabled by deployment")
    if (
        prefix == "AGENT"
        and model.get("context_window")
        and model["context_window"] != integer(c, "AGENT_CONTEXT_WINDOW_TOKENS", minimum=1)
    ):
        raise ConfigFailure("Declared context does not match deployment")
    identity = {
        k: model.get(k) for k in ("id", "revision", "engine", "dtype", "thinking", "context_window")
    }
    store.pin(prefix + "_deployment", identity)
    return identity


def request_model(store, c, prefix, owner, step, generation, payload):
    """Persist replayable request/response with private profile fields removed."""
    import httpx

    base, headers = model_connection(c, prefix)
    remote = bool(headers)
    safe_payload = redact({**payload, "model": "<remote>"} if remote else payload)
    saved = store.db.execute(
        "SELECT * FROM requests WHERE owner=? AND step=? AND generation=? AND status='response_saved' AND rejected=0 ORDER BY rowid DESC LIMIT 1",
        (owner, step, generation),
    ).fetchone()
    if saved:
        if json.loads(saved["payload"]) != safe_payload:
            raise ValueError("Saved request identity changed")
        raw = json.loads(saved["response"])
        check_thinking_response(c, prefix, raw)
        return raw, saved["id"]
    url = base + "/chat/completions"
    limiter = c.get("_api_limiter")
    reserve = len(dumps(payload).encode("utf-8")) + (payload.get("max_tokens") or 0)
    previous_attempts = store.db.execute(
        """SELECT count(*) FROM requests
           WHERE owner=? AND step=? AND generation=? AND status <> 'prepared'""",
        (owner, step, generation),
    ).fetchone()[0]
    for attempt in range(integer(c, prefix + "_MAX_HTTP_ATTEMPTS", 5, 1)):
        if limiter is not None:
            limiter.acquire(reserve)
        aid = uuid.uuid4().hex
        with store.tx() as db:
            fields = (aid, owner, step, generation, "prepared", dumps(safe_payload))
            if store.request_metadata:
                db.execute(
                    """INSERT INTO requests(id,owner,step,generation,status,payload,recovery_index,
                       transport_index) VALUES(?,?,?,?,?,?,?,?)""",
                    (*fields, c.get("_sample_recovery", 0), previous_attempts + attempt),
                )
            else:
                db.execute(
                    "INSERT INTO requests(id,owner,step,generation,status,payload) VALUES(?,?,?,?,?,?)",
                    fields,
                )
        with store.tx() as db:
            db.execute("UPDATE requests SET status='sent' WHERE id=?", (aid,))
        started = time.monotonic()
        wait = min(2**attempt, integer(c, "RETRY_MAX_WAIT_SECONDS", 60, 1))
        http_status = None
        try:
            client = c.get("_http_client")
            post = client.post if client is not None else httpx.post
            extra = {} if client is not None else {"trust_env": not remote}
            response = post(
                url,
                json=payload,
                headers=headers,
                timeout=integer(c, prefix + "_REQUEST_TIMEOUT_SECONDS", 120, 1),
                follow_redirects=False,
                **extra,
            )
            if not response.is_success:
                diagnostic = {"http_status": response.status_code}
                try:
                    error_body = response.json()
                except ValueError:
                    error_body = None
                raw_usage = error_body.get("usage") if isinstance(error_body, dict) else None
                usage = normalize_usage(redact(raw_usage)) if raw_usage is not None else None
                if not remote:
                    diagnostic.update(
                        body=response.text,
                        request_id=response.headers.get("x-request-id"),
                        retry_after=response.headers.get("Retry-After"),
                    )
                with store.tx() as db:
                    db.execute(
                        "UPDATE requests SET response=?,usage=? WHERE id=?",
                        (dumps(redact(diagnostic)), dumps(usage) if usage is not None else None, aid),
                    )
            if response.status_code in (408, 429) or (
                response.status_code >= 500 and response.status_code not in (501, 505)
            ):
                http_status = response.status_code
                wait = retry_delay(response, wait, integer(c, "RETRY_MAX_WAIT_SECONDS", 60, 1))
                if limiter is not None and http_status == 429:
                    limiter.pause(wait)
                detail = "" if remote else ": " + response.text[:300]
                raise httpx.TransportError(f"Temporary HTTP {response.status_code}{detail}")
            if not response.is_success:
                detail = "" if remote else ": " + response.text[:500]
                if response.status_code in (400, 413):
                    try:
                        provider_error = response.json().get("error", {})
                        code = (
                            provider_error.get("code") if isinstance(provider_error, dict) else None
                        )
                    except (ValueError, AttributeError):
                        code = None
                    if response.status_code == 413 or code in (
                        "context_length_exceeded",
                        "max_context_length_exceeded",
                    ):
                        raise ContextFailure("Provider rejected the request size")
                raise ConfigFailure(f"Chat Completions HTTP {response.status_code}{detail}")
            raw = response.json()
            if remote and isinstance(raw, dict):
                if "model" in raw:
                    raw["model"] = "<remote>"
                raw.pop("deployment", None)
            raw = redact(raw)
            usage = normalize_usage(raw.get("usage") if isinstance(raw, dict) else None)
            with store.tx() as db:
                db.execute(
                    "UPDATE requests SET status='response_saved',response=?,usage=?,latency=? WHERE id=?",
                    (dumps(raw), dumps(usage), time.monotonic() - started, aid),
                )
            if usage["issues"]:
                store.failure(
                    owner=owner,
                    attempt_id=aid,
                    code="USAGE_INVALID",
                    message="; ".join(usage["issues"]),
                    exposed=False,
                )
            check_thinking_response(c, prefix, raw)
            return raw, aid
        except (httpx.HTTPError, ValueError, ConfigFailure) as e:
            attempt_status = (
                "outcome_unknown"
                if isinstance(e, (httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError))
                else "failed"
            )
            error_message = (
                f"HTTP {http_status}"
                if remote and http_status
                else (
                    type(e).__name__
                    if remote and isinstance(e, httpx.RequestError)
                    else redact(str(e))
                )
            )
            with store.tx() as db:
                db.execute(
                    "UPDATE requests SET status=?,error=?,latency=? WHERE id=?",
                    (attempt_status, error_message, time.monotonic() - started, aid),
                )
            store.failure(
                owner=owner,
                step_id=step,
                generation_index=generation,
                attempt_id=aid,
                code="RATE_LIMITED" if http_status == 429 else type(e).__name__,
                message=error_message,
                exposed=False,
            )
            if isinstance(e, ConfigFailure):
                raise
            if attempt + 1 < integer(c, prefix + "_MAX_HTTP_ATTEMPTS", 5, 1):
                time.sleep(max(0, wait))
    raise RetryLater(f"{owner}: request cycle exhausted")


def check_thinking_response(c, prefix, raw):
    if c.get("_remote", {}).get("stage") != prefix or boolean(c, prefix + "_ENABLE_THINKING"):
        return
    if not isinstance(raw, dict):
        return
    choices = raw.get("choices") or []
    reasoning = any((ch.get("message") or {}).get("reasoning_content") for ch in choices)
    reasoning_tokens = ((raw.get("usage") or {}).get("completion_tokens_details") or {}).get(
        "reasoning_tokens"
    ) or 0
    if reasoning or reasoning_tokens:
        raise ConfigFailure(
            "Relay returned reasoning while thinking is disabled; check the private model and control setting"
        )
