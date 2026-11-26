"""Online node classification. Reads a finished graph; never builds one or evaluates."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import copy
import csv
import json
from pathlib import Path
import random
import re
import traceback
import sqlite3
import subprocess
import time
from threading import Lock

from experiment_io import (
    AGENT_ALGORITHM,
    VERSION,
    ConfigFailure,
    ContextFailure,
    RetryLater,
    StageBusy,
    Store,
    api_runtime,
    bounded_jobs,
    config_path,
    csv_values,
    digest,
    dumps,
    file_hash,
    integer,
    load_env,
    process_lock,
    public_config,
    record_launch,
    read_json,
    request_model,
    stage_fingerprint,
    thinking_params,
    write_json,
    worker_count,
)

MODES = {"target_only", "fine_only", "coarse_only", "multi"}
TARGET_SPLITS = ("train", "val", "test")
MODE_STRATEGIES = {
    "multi": "若当前证据充足，则直接提交预测结果。若证据不足，优先在粗粒度图上寻找证据。若粗粒度证据仍然不足，可在原始图上进一步寻找证据。",
    "fine_only": "若当前证据充足，则直接提交预测结果。若证据不足，仅在原始图上寻找证据。",
    "coarse_only": "若当前证据充足，则直接提交预测结果。若证据不足，仅查询粗粒度图上的邻居，不获取原始节点证据。",
    "target_only": "仅依据目标节点文本进行分类，并提交预测结果。",
}


def system_prompt(classes, mode):
    labels = "\n".join(f"{row['class_id']}: {row['name']}" for row in classes)
    return (
        "你的任务是根据已有证据，从候选类别中选择目标节点所属的类别。\n\n"
        f"候选类别：\n{labels}\n\n{MODE_STRATEGIES[mode]}"
    )


FORCE_SUBMIT = "请立刻根据已有信息提交"


class CallError(ValueError):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def tool_schemas(classes, c=None):
    c = c or {}
    k_schema = {
        "type": "integer",
        "minimum": integer(c, "AGENT_MIN_K", 1, 1),
        "maximum": integer(c, "AGENT_MAX_K", 10, 1),
    }
    prediction = {"type": "string", "enum": [r["class_id"] for r in classes], "description": "预测的类别ID。"}

    def batch(field, description, k_description):
        return {
            "queries": {
                "type": "array",
                "description": "查询列表。",
                "minItems": 1,
                "maxItems": integer(c, "AGENT_MAX_SEARCH_CALLS_PER_ROUND", 4, 1),
                "items": {
                    "type": "object",
                    "properties": {
                        field: {"type": "string", "description": description},
                        "k": {**k_schema, "description": k_description},
                    },
                    "required": [field, "k"],
                    "additionalProperties": False,
                },
            }
        }

    schemas = []
    for name, desc, props in [
        (
            "search_1hop",
            "批量查询已见节点的1-hop邻居。F:表示原始节点，C:表示超节点。优先返回未见节点，再按与目标节点的相似度排序。"
            "Neighbors记录邻接关系，w为粗图边权，remaining为剩余候选数，Evidence为新增证据，"
            "sim为与目标节点的相似度，degree为所在图的邻居数，Budget为剩余查询预算。",
            batch(
                "center", "本轮开始前已见的原始节点或超节点ID；同一来源同类操作每轮仅查一次，后续轮可续取。",
                "最多返回的邻居数。根据证据需求和节点度数选择。在同一张图上，通常距搜索起点越远，k越小。",
            ),
        ),
        (
            "expand_community",
            "批量获取已见超节点包含的未见原始节点，按与目标节点的相似度降序返回。"
            "Members记录成员关系，remaining为剩余候选数，Evidence为新增证据，"
            "sim为与目标节点的相似度，degree为原始图的邻居数，Budget为剩余查询预算。",
            batch(
                "community", "本轮开始前已见的超节点ID；同一来源同类操作每轮仅查一次，后续轮可续取。",
                "最多返回的原始节点数，根据证据需求和超节点大小合理选择。",
            ),
        ),
        (
            "submit",
            "提交目标节点的预测类别并结束搜索。同轮图查询不执行。",
            {
                "prediction": prediction,
                "evidence_ids": {
                    "type": "array", "items": {"type": "string"}, "maxItems": 20,
                    "description": "可选，支持预测的已见节点ID。",
                },
            },
        ),
    ]:
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": {
                        "type": "object",
                        "properties": props,
                        "required": ["prediction"] if name == "submit" else list(props),
                        "additionalProperties": False,
                    },
                },
            }
        )
    return schemas


def load_search_data(folder):
    import numpy as np

    folder = Path(folder)
    manifest = read_json(folder / "build_manifest.json")
    if manifest.get("status") != "ready" or manifest.get("schema_version") != VERSION:
        raise ValueError(
            "A finished schema-v3 graph build is required; run build_coarse_graph.py separately"
        )
    for artifact in manifest["artifacts"]:
        p = (folder / artifact["path"]).resolve()
        if not p.is_relative_to(folder.resolve()) or file_hash(p) != artifact["sha256"]:
            raise ValueError(f"Artifact invalid: {artifact['path']}")
    raw = read_json(folder / "artifacts/raw_graph.json")
    coarse = read_json(folder / "artifacts/coarse_graph.json")
    if (
        coarse["graph_sha256"] != raw["graph_sha256"]
        or manifest["source_graph_sha256"] != raw["graph_sha256"]
    ):
        raise ValueError("Graph identity mismatch")
    n = len(raw["papers"])
    k = len(coarse["communities"])
    if len(coarse["node_to_coarse"]) != n or any(
        type(x) is not int or not 0 <= x < k for x in coarse["node_to_coarse"]
    ):
        raise ValueError("Invalid coarse mapping")
    coarse_adj = [{} for _ in range(k)]
    for e in coarse["coarse_edges"]:
        coarse_adj[e["u"]][e["v"]] = e["weight"]
        coarse_adj[e["v"]][e["u"]] = e["weight"]
    members = [[] for _ in range(k)]
    for node, community in enumerate(coarse["node_to_coarse"]):
        members[community].append(node)
    meta = read_json(folder / "embeddings/embedding_meta.json")
    vectors = {}
    for kind, count in [("fine", n), ("coarse", k)]:
        parts = sorted([x for x in meta["files"] if x["kind"] == kind], key=lambda x: x["start"])
        cursor = 0
        arrays = []
        for part in parts:
            if part["start"] != cursor:
                raise ValueError("Noncontiguous embedding rows")
            arr = np.load(folder / part["path"], allow_pickle=False)
            if list(arr.shape) != part["shape"] or not np.isfinite(arr).all():
                raise ValueError("Invalid vector block")
            arrays.append(arr)
            cursor = part["end"]
        if cursor != count:
            raise ValueError("Incomplete vectors")
        vectors[kind] = np.concatenate(arrays)
        if not np.allclose(np.linalg.norm(vectors[kind], axis=1), 1, atol=1e-4):
            raise ValueError("Unnormalized vectors")
    from build_coarse_graph import fine_text

    return {
        "raw": raw,
        "coarse": coarse,
        "manifest": manifest,
        "manifest_hash": digest(manifest),
        "fine_adj": raw["fine_adj"],
        "coarse_adj": coarse_adj,
        "members": members,
        "fine_texts": [fine_text(p) for p in raw["papers"]],
        "coarse_texts": [
            x["summary"] + "\nKeywords: " + ", ".join(x["keywords"]) for x in coarse["communities"]
        ],
        "vectors": vectors,
        "mapping": coarse["node_to_coarse"],
    }


def load_evaluation_labels(c, raw):
    data_dir = config_path(c, "DATA_DIR")
    classes_file = config_path(c, "AGENT_CLASSES_FILE", str(data_dir / "classes.json"))
    target_file = config_path(c, "AGENT_TARGETS_FILE", str(data_dir / "targets.tsv"))
    classes = read_json(classes_file)
    classes = sorted(classes, key=lambda x: x["class_id"])
    ids = [r["class_id"] for r in classes]
    if not ids or len(ids) != len(set(ids)) or any(not isinstance(x, str) or not x for x in ids):
        raise ValueError("Invalid classes")
    rows = []
    seen = set()
    lookup = raw["paper_id_to_fine"]
    with target_file.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if (
                r["paper_id"] in seen
                or r["paper_id"] not in lookup
                or r["class_id"] not in ids
                or r["split"] not in TARGET_SPLITS
            ):
                raise ValueError("Invalid/duplicate label row")
            seen.add(r["paper_id"])
            rows.append(
                {
                    "fine_id": lookup[r["paper_id"]],
                    "paper_id": r["paper_id"],
                    "truth": r["class_id"],
                    "split": r["split"],
                }
            )
    return (
        classes,
        rows,
        {"file_sha256": file_hash(target_file), "classes_sha256": file_hash(classes_file)},
    )


def select_plan_targets(plan, rows, dataset, graph_hash, sources):
    if (
        not isinstance(plan, dict)
        or plan.get("schema_version") != 1
        or plan.get("dataset") != dataset
        or plan.get("source_graph_sha256") != graph_hash
        or plan.get("label_source") != sources
    ):
        raise ValueError("Target plan belongs to another graph or label/split version")
    paper_ids = plan.get("paper_ids")
    split = plan.get("split")
    if (
        split not in TARGET_SPLITS
        or not isinstance(paper_ids, list)
        or not paper_ids
        or any(not isinstance(paper_id, str) for paper_id in paper_ids)
        or len(paper_ids) != len(set(paper_ids))
        or plan.get("size") != len(paper_ids)
    ):
        raise ValueError("Invalid target plan split or node list")
    eligible = {row["paper_id"]: row for row in rows if row["split"] == split}
    if not set(paper_ids) <= eligible.keys():
        raise ValueError("Target plan includes a node outside its declared split")
    return [eligible[paper_id] for paper_id in paper_ids]


def freeze_inputs(c, data, folder, store):
    existing = folder / "evaluation_inputs.json"
    plan = read_json(config_path(c, "AGENT_TARGET_PLAN")) if c.get("AGENT_TARGET_PLAN") else None
    if existing.exists() and store.meta("evaluation_input_sha256"):
        fixed = read_json(existing)
        if digest(fixed) != store.meta("evaluation_input_sha256"):
            raise ValueError("Evaluation input changed")
        if plan is not None and fixed["selection"].get("target_plan_sha256") != digest(plan):
            raise ValueError("Target plan changed; choose a new experiment ID")
        return fixed
    if c["DATASET"] != data["coarse"]["dataset"]:
        raise ValueError("Dataset mismatch")
    raw = data["raw"]
    classes, rows, sources = load_evaluation_labels(c, raw)
    split = c.get("AGENT_TARGET_SPLIT", "val")
    if plan is not None:
        targets = select_plan_targets(plan, rows, c["DATASET"], raw["graph_sha256"], sources)
        selection = {
            "target_split": plan["split"],
            "target_limit": len(targets),
            "target_seed": plan.get("seed"),
            "target_plan_sha256": digest(plan),
            "sampling": plan.get("sampling"),
        }
    else:
        if split not in TARGET_SPLITS:
            raise ValueError("Target split must be train, val or test")
        targets = sorted([r for r in rows if r["split"] == split], key=lambda x: x["fine_id"])
        seed = integer(c, "AGENT_TARGET_SEED", 234)
        random.Random(seed).shuffle(targets)
        if c.get("AGENT_TARGET_LIMIT"):
            targets = targets[: integer(c, "AGENT_TARGET_LIMIT", minimum=1)]
        selection = {
            "target_split": split,
            "target_limit": c.get("AGENT_TARGET_LIMIT", ""),
            "target_seed": seed,
        }
    if not targets:
        raise ValueError("No targets selected")
    fixed = {
        "schema_version": VERSION,
        "source_graph_sha256": raw["graph_sha256"],
        "classes": classes,
        "targets": targets,
        "split_source": sources if plan is not None else {"file_sha256": sources["file_sha256"]},
        "selection": selection,
    }
    if existing.exists() and read_json(existing) != fixed:
        raise ValueError("Unregistered evaluation snapshot conflict")
    if plan is not None:
        store.pin("target_plan_sha256", digest(plan))
    write_json(existing, fixed)
    store.pin("evaluation_input_sha256", digest(fixed))
    return fixed


def ref(level, node):
    return ("F:" if level == "fine" else "C:") + str(node)


def parse_ref(value, data):
    if not isinstance(value, str) or not re.fullmatch(r"[FC]:(0|[1-9][0-9]*)", value):
        raise CallError("INVALID_ARGUMENT", "Invalid node reference")
    level = "fine" if value.startswith("F:") else "coarse"
    digits = value[2:]
    count = len(data[level + "_texts"])
    if len(digits) > len(str(count)) or (n := int(digits)) >= count:
        raise CallError("NODE_NOT_FOUND", "Node does not exist")
    return level, n


def evidence(data, state, level, node):
    if level == "fine":
        structure = {
            "degree": len(data["fine_adj"][node]),
            "same_community_as_target": data["mapping"][node] == data["mapping"][state["target_id"]]
            if state["mode"] in ("multi", "coarse_only")
            else None,
        }
        source = "original_title_abstract"
    else:
        row = data["coarse"]["communities"][node]
        structure = {
            "size": row["size"],
            "degree": len(data["coarse_adj"][node]),
            **{k: row["stats"][k] for k in ("density", "ext_deg_frac")},
            "summary_source": row["summary_source"],
            "summary_incomplete": row.get("summary_incomplete", False),
        }
        source = row["summary_source"]
    return {
        "ref": ref(level, node),
        "structure": structure,
        "text": data[level + "_texts"][node],
        "text_source": source,
    }


def remaining(state, c):
    return {
        "graph_tool_calls": max(
            0, integer(c, "AGENT_MAX_GRAPH_TOOL_CALLS", 6) - state["counts"]["graph_tool_calls"]
        ),
        "model_rounds": max(
            0, integer(c, "AGENT_MAX_MODEL_ROUNDS", 14, 1) - state["counts"]["model_rounds"]
        ),
        "context_tokens_available": state.get("context_tokens_available"),
    }


def source_progress(state, level, node, expand):
    if expand:
        return state["community_sent"], str(node)
    return state["neighbor_sent"], f"{level}:{node}"


def source_nodes(data, state, level, node, expand=False):
    values = data["members"][node] if expand else data[level + "_adj"][node]
    progress, key = source_progress(state, level, node, expand)
    sent = set(progress.get(key, ()))
    seen = set(state["seen_fine" if expand or level == "fine" else "seen_coarse"])
    return [n for n in values if n not in sent and (not expand or n not in seen)]


def allowed(data, state, c, decision=False):
    tools = ["submit"]
    levels = []
    rem = remaining(state, c)
    if (
        state.get("force_submit_next")
        or rem["model_rounds"] <= (0 if decision else 1)
        or not rem["graph_tool_calls"]
    ):
        return tools, levels
    mode = state["mode"]
    count_seen = len(state["seen_fine"]) + len(state["seen_coarse"])

    def available(kind, level, node, expand=False):
        key = f"{kind}:{level}:{node}"
        progress, source_key = source_progress(state, level, node, expand)
        return state["blocked"].get(key) != count_seen and (
            source_key not in progress or bool(source_nodes(data, state, level, node, expand))
        )

    if mode in ("coarse_only", "multi") and any(
        available("search", "coarse", n) for n in state["seen_coarse"]
    ):
        levels.append("coarse")
    if mode in ("fine_only", "multi") and any(
        available("search", "fine", n) for n in state["seen_fine"]
    ):
        levels.append("fine")
    if levels:
        tools.insert(0, "search_1hop")
    if mode == "multi" and any(
        available("expand", "coarse", n, True) for n in state["seen_coarse"]
    ):
        tools.insert(0, "expand_community")
    return tools, levels


def state_view(data, s, c, new_refs=None):
    tools, levels = allowed(data, s, c)
    result = {
        "mode": s["mode"],
        "allowed_tools": tools,
        "allowed_search_levels": levels,
        "remaining": remaining(s, c),
        "coarse_search_attempted": s["coarse_search_attempted"],
        "coarse_evidence_delivered": s["coarse_evidence_delivered"],
        "max_queries_this_round": min(
            integer(c, "AGENT_MAX_SEARCH_CALLS_PER_ROUND", 4, 1),
            remaining(s, c)["graph_tool_calls"],
        ),
        "context_blocked_sources": sorted(
            k for k, v in s["blocked"].items() if v == len(s["seen_fine"]) + len(s["seen_coarse"])
        ),
    }
    if tools == ["submit"]:
        result.pop("context_blocked_sources", None)
    result["visible_refs" if new_refs is None else "new_refs"] = (
        new_refs
        if new_refs is not None
        else sorted(
            [ref("fine", n) for n in s["seen_fine"]] + [ref("coarse", n) for n in s["seen_coarse"]]
        )
    )
    return result


def make_sample_state(data, target, mode, classes, c, counter=None):
    home = data["mapping"][target] if mode in ("coarse_only", "multi") else None
    s = {
        "mode": mode,
        "target_id": target,
        "home_id": home,
        "seen_fine": [target],
        "seen_coarse": [home] if home is not None else [],
        "neighbor_sent": {},
        "community_sent": {},
        "opened_communities": [],
        "queried_centers": {"fine": [], "coarse": []},
        "first_seen_step": {ref("fine", target): 0},
        "coarse_search_attempted": False,
        "coarse_evidence_delivered": False,
        "first_fine_request_round": None,
        "first_new_fine_round": None,
        "blocked": {},
        "force_submit_next": False,
        "counts": {
            "tool_ok_rounds": 0,
            "invalid_queries": 0,
            "empty_queries": 0,
            "skipped_queries": 0,
            "tool_partial_rounds": 0,
            "tool_failed_rounds": 0,
            "rounds_with_new_evidence": 0,
            "model_rounds": 0,
            "graph_tool_calls": 0,
            "center_queries": 0,
            "coarse_center_queries": 0,
            "fine_center_queries": 0,
            "community_calls": 0,
            "retrieved_chars": 0,
            "new_fine_nodes": 0,
            "target_chars": len(data["fine_texts"][target]),
        },
        "messages": [],
        "status": "running",
    }
    target_part = {"ref": ref("fine", target), "text": data["fine_texts"][target]}
    target_structure = (
        evidence(data, s, "fine", target)["structure"] if mode != "target_only" else None
    )
    community = evidence(data, s, "coarse", home) if home is not None else None
    if home is not None:
        s["first_seen_step"][ref("coarse", home)] = 0
        community["similarity_to_target"] = round(target_similarity(data, s, "coarse", home), 4)
    s["initial_evidence"] = {
        "target": {**target_part, "structure": target_structure}, "community": community,
    }
    s["counts"]["initial_evidence_chars"] = len(target_part["text"]) + (
        len(community["text"]) if community else 0
    )
    header = f"Target {target_part['ref']}"
    if target_structure:
        header = text_fields(header, {
            "degree": target_structure["degree"],
            "same_community": target_structure.get("same_community_as_target"),
        })
    blocks = [header + "\n" + evidence_text(target_part["text"])]
    if community:
        attrs = community["structure"]
        blocks.append(text_fields(f"Community {community['ref']}", {
            "degree": attrs["degree"], "size": attrs["size"], "density": attrs["density"],
            "ext_deg_frac": attrs["ext_deg_frac"], "sim": community["similarity_to_target"],
            "summary_incomplete": True if attrs.get("summary_incomplete") else None,
        }) + "\n" + evidence_text(community["text"], summary=True))
    if mode != "target_only":
        blocks.append(f"Budget: {remaining(s, c)['graph_tool_calls']}")
        if allowed(data, s, c)[0] == ["submit"]:
            blocks.append("Submit only")
    s["messages"] = [
        {"role": "system", "content": system_prompt(classes, mode)},
        {"role": "user", "content": "\n\n".join(blocks)},
    ]
    if counter is not None:
        s["context_tokens_available"] = context_input_limit(c) - counter(s["messages"])
    return s


def parse_calls(message):
    values = message.get("tool_calls")
    if not isinstance(values, list) or not values:
        raise CallError("NO_TOOL_CALLS", "Use a function")
    calls = []
    interpretable = False
    for i, item in enumerate(values):
        f = item.get("function", {}) if isinstance(item, dict) else {}
        f = f if isinstance(f, dict) else {}
        original_id = item.get("id") if isinstance(item, dict) else None
        name = f.get("name")
        valid_name = isinstance(name, str) and bool(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name))
        interpretable |= valid_name or (isinstance(original_id, str) and bool(original_id))
        supplied = f.get("arguments")
        if isinstance(supplied, dict):
            supplied = dumps(supplied)  # Some relays decode native JSON arguments.
        calls.append(
            {
                "id": original_id,
                "name": name if valid_name else "invalid_tool",
                "arguments": supplied,
                "index": i,
            }
        )
    if not interpretable:
        raise CallError("INVALID_TOOL_MESSAGE", "Tool calls cannot be paired with native responses")
    return calls


def arguments(call):
    try:
        value = json.loads(call["arguments"])
    except (ValueError, TypeError) as e:
        raise CallError("INVALID_ARGUMENT", "Arguments must be valid JSON") from e
    if not isinstance(value, dict):
        raise CallError("INVALID_ARGUMENT", "Arguments must be an object")
    return value


def validate_submit(args, start, classes):
    if set(args) - {"prediction", "evidence_ids", "reason"}:
        raise CallError("INVALID_SUBMIT", "Invalid submit fields")
    prediction = args.get("prediction")
    if prediction not in [x["class_id"] for x in classes]:
        raise CallError("INVALID_SUBMIT", "Submit requires a valid category ID")
    refs = args.get("evidence_ids", [])
    if not isinstance(refs, list):
        refs = [refs]
        valid, invalid = [], refs
    else:
        seen = start["first_seen_step"]
        valid, invalid = [], []
        for node in refs:
            if isinstance(node, str) and node in seen and node not in valid and len(valid) < 20:
                valid.append(node)
            else:
                invalid.append(node)
    return {"prediction": prediction, "evidence_ids": valid}, invalid


def target_similarity(data, state, level, node):
    return float(data["vectors"][level][node] @ data["vectors"]["fine"][state["target_id"]])


def result_shell(call):
    return {
        "ok": False,
        "function": call["name"],
        "by_source": [],
        "new_evidence": [],
        "already_read": [],
        "spent": {"graph_tool_calls": 0, "center_queries": 0, "evidence_chars": 0},
        "state_delta": None,
        "error": None,
    }


def source_result(source, status, *, message=None, total=None, has_more=False, items=(), executed=False):
    return {
        "source": source,
        "status": status,
        "message": message,
        "total_count": total,
        "returned_count": len(items),
        "items": list(items),
        "has_more": has_more,
        "executed": executed,
    }


def query(data, state, start, call, args, c, step, used, limit=None, excluded=()):
    expand = call["name"] == "expand_community"
    field = "community" if expand else "center"
    src = args.get(field) if isinstance(args, dict) else None
    src = src if isinstance(src, str) else None
    if not isinstance(args, dict) or set(args) != {field, "k"}:
        raise CallError("invalid_args", f"Expected {field} and k")
    k = args["k"]
    if type(k) is not int or not integer(c, "AGENT_MIN_K", 1, 1) <= k <= integer(c, "AGENT_MAX_K", 10, 1):
        raise CallError("invalid_args", "k is outside the configured range")
    if src is None or not re.fullmatch(r"[FC]:(0|[1-9][0-9]*)", src):
        raise CallError("invalid_center", "Center must be a visible node reference")
    level = "fine" if src.startswith("F:") else "coarse"
    if expand and level != "coarse":
        raise CallError("invalid_center", "Expansion requires a community reference")
    mode = start["mode"]
    if (
        mode == "target_only"
        or (mode == "fine_only" and (expand or level != "fine"))
        or (mode == "coarse_only" and (expand or level != "coarse"))
    ):
        raise CallError("mode_forbidden", "Operation is unavailable in this mode")
    if src not in start["first_seen_step"]:
        raise CallError("invalid_center", "Center must be visible at round start")
    _, node = parse_ref(src, data)
    key = (call["name"], src)
    if key in used:
        raise CallError("duplicate", "This operation already used this source in the round")
    used.add(key)

    rows = data["members"][node] if expand else data[level + "_adj"][node]
    progress, progress_key = source_progress(state, level, node, expand)
    candidates = source_nodes(data, state, level, node, expand)
    if not candidates and progress_key in progress:
        return source_result(src, "exhausted", total=len(rows)), [], []
    if not remaining(state, c)["graph_tool_calls"]:
        raise CallError("budget_exhausted", "No logical query budget remains")

    result_level = "fine" if expand else level
    visible = state["seen_fine" if result_level == "fine" else "seen_coarse"]
    seen = set(visible)
    similarity = {n: target_similarity(data, state, result_level, n) for n in candidates}

    def rank(n):
        weight = rows[n] if level == "coarse" and not expand else 0
        return n in seen, -similarity[n], -weight, n

    ranked = sorted(candidates, key=rank)
    selected = [n for n in ranked if ref(result_level, n) not in excluded][
        :k if limit is None else min(k, limit)
    ]
    status = "partial" if len(selected) < min(k, len(candidates)) else "ok"
    source_key = f"{'expand' if expand else 'search'}:{level}:{node}"
    if candidates and not selected:
        state["blocked"][source_key] = len(state["first_seen_step"])
        return source_result(src, "context_limited", total=len(rows), has_more=True), [], []
    state["blocked"].pop(source_key, None)
    if not candidates:
        status = "empty"

    new, known, items = [], [], []
    for n in selected:
        node_ref = ref(result_level, n)
        item = {"ref": node_ref, "similarity_to_target": similarity[n]}
        if not expand:
            item["weight"] = rows[n] if level == "coarse" else 1
        items.append(item)
        if n in seen:
            known.append({
                "ref": node_ref, "status": "already_read",
                "first_seen_step": state["first_seen_step"][node_ref],
            })
            continue
        entry = evidence(data, state, result_level, n)
        new.append(entry)
        visible.append(n)
        state["first_seen_step"][node_ref] = step
        state["counts"]["retrieved_chars"] += len(entry["text"])
    visible.sort()
    progress[progress_key] = sorted(set(progress.get(progress_key, [])) | set(selected))
    counts = state["counts"]
    counts["graph_tool_calls"] += 1
    if expand:
        counts["community_calls"] += 1
        if node not in state["opened_communities"]:
            state["opened_communities"].append(node)
    else:
        counts["center_queries"] += 1
        counts[level + "_center_queries"] += 1
        if node not in state["queried_centers"][level]:
            state["queried_centers"][level].append(node)
        if level == "coarse":
            state["coarse_search_attempted"] = True
            state["coarse_evidence_delivered"] |= bool(new)
    if result_level == "fine":
        state["first_fine_request_round"] = state["first_fine_request_round"] or step
        counts["new_fine_nodes"] += len(new)
        if new:
            state["first_new_fine_round"] = state["first_new_fine_round"] or step
    return source_result(
        src, status, total=len(rows), has_more=bool(source_nodes(data, state, level, node, expand)),
        items=items, executed=True,
    ), new, known


def normalize_calls(calls, start):
    history = {x["id"] for m in start["messages"] for x in m.get("tool_calls", [])}
    safe_id = re.compile(r"[A-Za-z0-9_-]{1,128}")
    reserved = history | {x["id"] for x in calls if isinstance(x["id"], str)}
    used = set(history)
    mapping = []
    for call in calls:
        old = call["id"]
        if not isinstance(old, str) or not safe_id.fullmatch(old) or old in used:
            new = "call_" + digest([start["target_id"], start["counts"]["model_rounds"], call["index"]])[:22]
            while new in reserved:
                new += "x"
            call["id"] = new
            reserved.add(new)
            mapping.append({"index": call["index"], "raw": old, "normalized": new})
        used.add(call["id"])
    return mapping


def history_arguments(call):
    try:
        arguments(call)
    except CallError:
        # Keep the original malformed output in requests, and serialize an object for chat replay.
        return "{}"
    return call["arguments"]


def assistant_message(raw_message, calls):
    content = raw_message.get("content")
    message = {
        "role": "assistant",
        "content": content if isinstance(content, str) else None,
        "tool_calls": [
            {
                "id": x["id"],
                "type": "function",
                "function": {
                    "name": x["name"] if isinstance(x["name"], str) else "invalid_tool",
                    "arguments": history_arguments(x),
                },
            }
            for x in calls
        ],
    }
    if isinstance(raw_message.get("reasoning_content"), str):
        message["reasoning_content"] = raw_message["reasoning_content"]
    return message


def compact_tool_result(result):
    """Project an execution result onto the fields the model can use."""
    if "by_source" not in result:
        return result

    output = {}
    if result.get("status"):
        output["status"] = result["status"]
        return output
    if result.get("error"):
        output["error"] = result["error"]

    rows = result["by_source"]
    group_key = {"search_1hop": "neighbors", "expand_community": "members"}.get(result["function"])
    if group_key:
        groups = output[group_key] = {}
        errors = []
        resolved = {row["source"] for row in rows if row["total_count"] is not None}
        for index, row in enumerate(rows):
            source = row["source"]
            status = row["status"]
            if (
                not isinstance(source, str) or not re.fullmatch(r"[FC]:(0|[1-9][0-9]*)", source)
                or source in groups or (source in resolved and row["total_count"] is None)
            ):
                error = {"index": index, "status": status}
                if source is not None:
                    error["ref"] = source
                if row.get("message"):
                    error["message"] = row["message"]
                errors.append(error)
                continue
            items = row["items"]
            entry = {"nodes": [item["ref"] for item in items]}
            if "remaining" in row:
                entry["remaining"] = row["remaining"]
            weights = {
                item["ref"]: item["weight"] for item in items
                if "weight" in item and (source.startswith("C:") or item["weight"] != 1)
            }
            if weights:
                entry["weights"] = weights
            if status != "ok":
                entry["status"] = status
            if row.get("message"):
                entry["message"] = row["message"]
            groups[source] = entry
        if errors:
            output["errors"] = errors

        similarities = {
            item["ref"]: item["similarity_to_target"] for row in rows for item in row["items"]
        }
        evidence_by_ref = output["evidence"] = {}
        for evidence in result["new_evidence"]:
            node_ref, structure = evidence["ref"], evidence["structure"]
            entry = {
                "sim": round(similarities[node_ref], 3),
                "text": evidence["text"],
                **{k: structure[k] for k in ("degree", "size", "density", "ext_deg_frac") if k in structure},
            }
            if structure.get("same_community_as_target") is not None:
                entry["same_community"] = structure["same_community_as_target"]
            if structure.get("summary_incomplete"):
                entry["summary_incomplete"] = True
            if structure.get("summary_source") not in (None, "llm"):
                entry["summary_source"] = structure["summary_source"]
            evidence_by_ref[node_ref] = entry

    if "budget" in result:
        output["budget"] = result["budget"]
    if (result.get("state_delta") or {}).get("allowed_tools") == ["submit"]:
        output["submit_only"] = True
    return output


def text_fields(header, fields):
    return " | ".join([header] + [
        f"{key}={str(value).lower() if isinstance(value, bool) else value}"
        for key, value in fields.items() if value is not None
    ])


def evidence_text(text, *, summary=False):
    if summary and not text.startswith("Summary:"):
        text = "Summary: " + text
    # Escape only lines that could be mistaken for a section or a source header.
    return re.sub(
        r"(?m)^(?=\\|(?:Neighbors|Members|Evidence|Budget|Errors?|Status|Accepted)\s*:|Submit only\b|"
        r"(?:Target |Community )?[FC]:[0-9]+\b)",
        r"\\",
        text,
    )


def render_tool_result(result):
    compact = compact_tool_result(result)
    blocks = []
    for key, title in (("neighbors", "Neighbors"), ("members", "Members")):
        if key not in compact:
            continue
        lines = [f"{title}:"]
        for source, entry in compact[key].items():
            weights = entry.get("weights", {})
            nodes = ", ".join(f"{node}(w={weights[node]})" if node in weights else node for node in entry["nodes"])
            parts = [text_fields(f"{source} -> {nodes or 'none'}", {"remaining": entry.get("remaining")})]
            if "status" in entry:
                status = "partial: context limited" if entry["status"] == "partial" else entry["status"]
                parts.append(status + (f": {entry['message']}" if entry.get("message") else ""))
            lines.append(" | ".join(parts))
        blocks.append("\n".join(lines))
    if compact.get("errors"):
        lines = ["Errors:"]
        for error in compact["errors"]:
            source = " ".join(error.get("ref", "").split())
            message = f": {error['message']}" if error.get("message") else ""
            lines.append(f"query {error['index']} {source}".rstrip() + f" | {error['status']}{message}")
        blocks.append("\n".join(lines))
    if (err := compact.get("error")) is not None:
        blocks.append(f"Error: {err['code']}: {err['message']}")
    if compact.get("evidence"):
        evidence_blocks = []
        for node, item in compact["evidence"].items():
            fields = {"sim": f"{item['sim']:.3f}", **{
                k: v for k, v in item.items() if k not in ("sim", "text")
            }}
            evidence_blocks.append(text_fields(node, fields) + "\n" + evidence_text(
                item["text"], summary=node.startswith("C:"),
            ))
        blocks.append("Evidence:\n" + "\n\n".join(evidence_blocks))
    if "budget" in compact:
        blocks.append(f"Budget: {compact['budget']}")
    if compact.get("submit_only"):
        blocks.append("Submit only")
    if compact.get("status"):
        blocks.append(f"Status: {compact['status']}")
    if compact.get("accepted"):
        blocks.append("Accepted")
    return "\n\n".join(blocks) if blocks else "OK"


def tool_messages(calls, results):
    return [
        {"role": "tool", "tool_call_id": call["id"], "content": render_tool_result(result)}
        for call, result in zip(calls, results)
    ]


def graph_queries(call):
    args = arguments(call)
    if set(args) != {"queries"} or not isinstance(args["queries"], list) or not args["queries"]:
        raise CallError("invalid_args", "queries must be a nonempty array")
    return args["queries"]


def context_input_limit(c):
    return (
        integer(c, "AGENT_CONTEXT_WINDOW_TOKENS", minimum=1)
        - integer(c, "AGENT_MAX_OUTPUT_TOKENS", 2048, 1)
        - integer(c, "AGENT_CONTEXT_SAFETY_TOKENS", 256)
    )


def pack_tool_batch(data, start, assistant, calls, c, counter, messages=None):
    """Fit paired tool replies by replaying only the requested sources on fresh state."""
    messages = start["messages"] if messages is None else messages
    step = start["counts"]["model_rounds"]
    per_round = integer(c, "AGENT_MAX_SEARCH_CALLS_PER_ROUND", 4, 1)
    limit = context_input_limit(c)
    msg = assistant_message(assistant, calls)
    terminal = {"role": "user", "content": FORCE_SUBMIT}
    parsed = []
    for call in calls:
        try:
            if call["name"] not in ("search_1hop", "expand_community"):
                raise CallError("unknown_tool", "Unknown graph tool")
            parsed.append((graph_queries(call), None))
        except CallError as error:
            parsed.append(([], {"code": error.code, "message": str(error)}))
    caps, omitted = {}, {}
    while True:
        state = copy.deepcopy(start)
        results = []
        used = set()
        position = 0
        delivered = []
        for call_index, (call, (queries, error)) in enumerate(zip(calls, parsed)):
            result = result_shell(call)
            result["error"] = error
            for item_index, args in enumerate(queries):
                position += 1
                key = (call_index, item_index)
                try:
                    if position > per_round:
                        raise CallError("round_limit", "Round subquery limit reached")
                    row, new, known = query(
                        data, state, start, call, args, c, step, used, caps.get(key), omitted.get(key, ()),
                    )
                except CallError as invalid:
                    raw_ref = args.get("center", args.get("community")) if isinstance(args, dict) else None
                    # Invalid IDs are already available in the raw attempt log.
                    raw_ref = raw_ref if isinstance(raw_ref, str) and len(raw_ref) < 40 else None
                    row = source_result(raw_ref, invalid.code, message=str(invalid))
                    new, known = [], []
                result["by_source"].append(row)
                result["new_evidence"].extend(new)
                result["already_read"].extend(known)
                if row["items"]:
                    delivered.append((key, row, new))
                if row["executed"]:
                    result["spent"]["graph_tool_calls"] += 1
                    result["spent"]["center_queries"] += call["name"] == "search_1hop"
                    result["spent"]["evidence_chars"] += sum(len(e["text"]) for e in new)
                counts = state["counts"]
                if row["status"] in ("invalid_center", "invalid_args", "mode_forbidden", "duplicate"):
                    counts["invalid_queries"] += 1
                elif row["status"] in ("budget_exhausted", "round_limit", "context_limited"):
                    counts["skipped_queries"] += 1
                elif row["status"] == "exhausted" or (row["executed"] and not new):
                    counts["empty_queries"] += 1
            result["ok"] = any(r["executed"] or r["status"] == "exhausted" for r in result["by_source"])
            results.append(result)
        budget = remaining(state, c)["graph_tool_calls"]
        for call, result in zip(calls, results):
            expand = call["name"] == "expand_community"
            result["budget"] = budget
            for row in result["by_source"]:
                if row["total_count"] is not None:
                    level, node = parse_ref(row["source"], data)
                    row["remaining"] = len(source_nodes(data, state, level, node, expand))
                    row["has_more"] = bool(row["remaining"])

        new_refs = sorted(state["first_seen_step"].keys() - start["first_seen_step"].keys())
        results[-1]["state_delta"] = state_view(data, state, c, new_refs)
        history = messages + [msg] + tool_messages(calls, results)
        room = limit - counter(history + [terminal])
        if room >= 0:
            state["context_tokens_available"] = room
            results[-1]["state_delta"]["remaining"]["context_tokens_available"] = room
            state["messages"] = history
            return state, results
        if not delivered:
            return None, results
        key, last, new = delivered[-1]
        count = last["returned_count"]
        # An individually oversized first hit should not starve a later short hit.
        if count == 1 and new:
            minimal = [result_shell(call) for call in calls]
            minimal[key[0]]["new_evidence"] = new
            minimal[key[0]]["by_source"] = [last]
            minimal[key[0]]["budget"] = budget
            if counter(messages + [msg] + tool_messages(calls, minimal) + [terminal]) > limit:
                omitted.setdefault(key, set()).add(new[0]["ref"])
                continue
        caps[key] = count - 1


class TokenCounter:
    def __init__(self, c, tools):
        self.tools = tools
        self.enable_thinking = c.get("AGENT_ENABLE_THINKING") == "true"
        self.lock = Lock()
        self.exact = bool(c.get("AGENT_TOKENIZER_PATH")) or not c.get("_remote")
        if not self.exact:
            self.identity = digest({"method": "utf8-envelope-estimate-v1", "tools": tools})
            return
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(config_path(c, "AGENT_TOKENIZER_PATH")), local_files_only=True
        )
        if c.get("AGENT_CHAT_TEMPLATE_FILE"):
            self.tokenizer.chat_template = config_path(c, "AGENT_CHAT_TEMPLATE_FILE").read_text()
        model_dir = config_path(c, "AGENT_TOKENIZER_PATH")
        hashes = {
            name: file_hash(model_dir / name)
            for name in (
                "tokenizer.json",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "vocab.json",
                "merges.txt",
            )
            if (model_dir / name).exists()
        }
        self.identity = digest(
            {
                "template": self.tokenizer.chat_template,
                "tools": tools,
                "revision": c["AGENT_MODEL_REVISION"],
                "tokenizer_files": hashes,
            }
        )

    def __call__(self, messages):
        if not self.exact:
            return len(dumps({"messages": messages, "tools": self.tools}).encode("utf-8"))
        with self.lock:
            return len(
                self.tokenizer.apply_chat_template(
                    messages,
                    tools=self.tools,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )
            )


def agent_payload(c, messages, tools, *, generation=0, seed=None, probe=False):
    remote = c.get("AGENT_MODEL_MODE", "local") == "api"
    choice = c.get("_api_tool_choice", "auto") if remote else "required"
    thinking = thinking_params(c, "AGENT")
    payload = {
        "model": c["AGENT_LLM_MODEL"],
        "messages": messages,
        "tools": tools,
        "tool_choice": choice,
        "stream": False,
        "max_tokens": integer(c, "AGENT_MAX_OUTPUT_TOKENS", 2048, 1),
        **thinking,
    }
    if thinking.get("thinking", {}).get("type") != "enabled":
        payload["temperature"] = (
            0
            if probe
            else float(c["AGENT_REGENERATE_TEMPERATURE"] if generation else c["AGENT_TEMPERATURE"])
        )
    if not remote:
        payload["parallel_tool_calls"] = True
        if seed is not None:
            payload["seed"] = seed
    if probe:
        payload["max_tokens"] = min(payload["max_tokens"], 512)
    return payload


def verify_tool_capability(store, c, classes, tools, counter):
    """Check that a single native call carries multiple queries on either backbone."""
    from experiment_io import verify_deployment

    identity = verify_deployment(store, c, "AGENT")
    capability_hash = digest([identity, counter.identity, "batched-tool-probe-v3"])
    if store.meta("capability_verified") == capability_hash:
        return
    owner = "capability:agent:" + capability_hash[:16]
    class_id = classes[0]["class_id"]
    k = integer(c, "AGENT_MIN_K", 1, 1)
    size = min(2, integer(c, "AGENT_MAX_SEARCH_CALLS_PER_ROUND", 4, 1))
    searches = [{"center": x, "k": k} for x in ("C:0", "F:1")[:size]]
    expansions = [{"community": x, "k": k} for x in ("C:0", "C:1")[:size]]
    stages = [
        ("search_1hop", searches, f"只调用一次 search_1hop，参数为 {dumps({'queries': searches})}。"),
        (
            "expand_community", expansions,
            f"只调用一次 expand_community，参数为 {dumps({'queries': expansions})}。",
        ),
        ("submit", None, f"只调用 submit，prediction={class_id}。"),
    ]
    messages = [{"role": "system", "content": "验证原生工具协议。根据当前用户指令调用函数。"}]
    call_ids = set()
    for step, (name, expected, instruction) in enumerate(stages, 1):
        messages = messages + [{"role": "user", "content": instruction}]
        payload = agent_payload(c, messages, tools, probe=True)
        raw, aid = request_model(store, c, "AGENT", owner, step, 0, payload)
        try:
            message = raw["choices"][0]["message"]
            calls = parse_calls(message)
            if len(calls) != 1 or calls[0]["name"] != name:
                raise ValueError("Expected one native tool call")
            call = calls[0]
            if not isinstance(call["id"], str) or not call["id"] or call["id"] in call_ids:
                raise ValueError("Invalid native tool ID")
            call_ids.add(call["id"])
            if expected is not None:
                queries = graph_queries(call)
                if queries != expected or any(type(item["k"]) is not int for item in queries):
                    raise ValueError("Unexpected batched arguments")
            else:
                validate_submit(arguments(call), {"first_seen_step": {}}, classes)
            actual = (raw.get("usage") or {}).get("prompt_tokens")
            margin = integer(c, "AGENT_CONTEXT_SAFETY_TOKENS", 256)
            if actual is None and not c.get("_remote"):
                raise ValueError("Local server did not report prompt token usage")
            if actual is not None:
                difference = actual - counter(messages)
                if (abs(difference) if counter.exact else difference) > margin:
                    raise ValueError("Tokenizer or prompt token count mismatch")
            messages = messages + [assistant_message(message, calls)] + tool_messages(calls, [{"ok": True}])
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as error:
            store.failure(owner=owner, step_id=step, attempt_id=aid, code="TOOL_PROBE_FAILED", exposed=False)
            with store.tx() as db:
                db.execute("UPDATE requests SET rejected=1 WHERE id=?", (aid,))
            raise ConfigFailure(f"Batched tool probe failed at round {step}") from error
    store.set_meta("remote_tool_capabilities", {
        "batched_queries_verified": size > 1,
        "queries_per_call_verified": size,
        "token_count": "local_tokenizer" if counter.exact else "utf8_envelope_estimate",
    })
    store.set_meta("capability_verified", capability_hash)


def skipped_results(calls, state, status, selected=None):
    results = []
    for call in calls:
        if call is selected:
            results.append({"accepted": True})
            continue
        result = result_shell(call)
        result["status"] = status
        if call["name"] in ("search_1hop", "expand_community"):
            try:
                queries = graph_queries(call)
            except CallError:
                queries = []
            field = "center" if call["name"] == "search_1hop" else "community"
            for args in queries:
                source = args.get(field) if isinstance(args, dict) else None
                result["by_source"].append(source_result(source, status))
            state["counts"]["skipped_queries"] += len(queries)
        results.append(result)
    return results


def handle_round(store, key, task, data, c, classes, tools, counter):
    started = time.monotonic()
    if task.get("active") is None:
        state = task["state"]
        state["counts"]["model_rounds"] += 1
        task["active"] = {
            "start": copy.deepcopy(state), "generation": 0, "step": state["counts"]["model_rounds"],
        }
        store.save_task(key, task)
    active = task["active"]
    start = active["start"]
    step, generation = active["step"], active["generation"]
    permitted, _ = allowed(data, start, c, decision=True)
    forced = permitted == ["submit"]
    messages = list(start["messages"])
    if forced:
        messages.append({"role": "user", "content": FORCE_SUBMIT})
    if generation:
        messages += active["repair_suffix"]
    if counter(messages) > context_input_limit(c):
        task.update(status="context_failure", failure_code="INPUT_TOO_LARGE", state=start, active=None)
        task["elapsed_seconds"] = task.get("elapsed_seconds", 0) + time.monotonic() - started
        store.commit_step(key, task, {
            "step_id": step, "generation_index": generation, "event_status": "context_failure",
            "code": "INPUT_TOO_LARGE", "calls": [], "results": [],
            "remaining_before": remaining(start, c), "remaining_after": remaining(start, c),
        })
        return

    seed = int(digest([integer(c, "AGENT_SEED", 234), task["repeat_id"], task["target_id"],
                       step, generation])[:8], 16) % 2147483647
    payload = agent_payload(c, messages, tools, generation=generation, seed=seed)
    runtime = {**c, "_sample_recovery": max(0, task.get("sample_runs", 1) - 1)}
    raw, aid = request_model(store, runtime, "AGENT", key, step, generation, payload)
    message, calls, mapping = None, [], []
    try:
        try:
            message = raw["choices"][0]["message"]
            calls = parse_calls(message)
        except (KeyError, TypeError, IndexError, AttributeError) as error:
            raise CallError("PROTOCOL_INVALID", "Model message could not be read") from error
        mapping = normalize_calls(calls, {**start, "messages": messages})
        submits = [call for call in calls if call["name"] == "submit"]
        selected = None
        ignored_submit_fields = []
        for call in submits:
            try:
                submitted_args = arguments(call)
                args, invalid_refs = validate_submit(submitted_args, start, classes)
            except CallError:
                continue
            ignored_submit_fields = ["reason"] if "reason" in submitted_args else []
            selected = call
            break
        if selected:
            state = copy.deepcopy(start)
            results = skipped_results(calls, state, "skipped_after_submit", selected)
            state["messages"] = messages + [assistant_message(message, calls)] + tool_messages(calls, results)
            rem = remaining(start, c)
            if not forced:
                trigger = "model"
            elif start["mode"] == "target_only":
                trigger = "target_only"
            elif start.get("force_submit_next"):
                trigger = "context_replan"
            elif not rem["model_rounds"]:
                trigger = "round_limit"
            elif not rem["graph_tool_calls"]:
                trigger = "graph_budget"
            else:
                trigger = "no_available_query"
            task.update(status="submitted", **args, reason=None, submit_trigger=trigger,
                        submit_after_failed_tool_round=start.get("last_event_status") == "tool_failed")
            event_status = "submitted"
        else:
            if submits:
                raise CallError("INVALID_SUBMIT", "Submit requires a valid category ID")
            if forced:
                raise CallError("FINAL_ROUND_REQUIRES_SUBMIT", "Expected a submit call")
            state, results = pack_tool_batch(data, start, message, calls, c, counter, messages)
            invalid_refs = []
            if state is None:
                state = copy.deepcopy(start)
                state["force_submit_next"] = True
                state["messages"] = messages
                for result in results:
                    for row in result["by_source"]:
                        if row["executed"]:
                            row.update(status="context_limited", executed=False, items=[], returned_count=0)
                        if row["status"] in ("invalid_center", "invalid_args", "mode_forbidden", "duplicate"):
                            state["counts"]["invalid_queries"] += 1
                        elif row["status"] != "exhausted":
                            state["counts"]["skipped_queries"] += 1
                    result.update(
                        new_evidence=[], already_read=[], state_delta=None, ok=False,
                        spent=result_shell({"name": result["function"]})["spent"],
                        budget=remaining(state, c)["graph_tool_calls"],
                    )
                event_status = "context_replan"
            else:
                rows = [row for result in results for row in result["by_source"]]
                ok = sum(row["status"] in ("ok", "empty", "exhausted") for row in rows)
                failed = len(rows) - ok + sum(result["error"] is not None for result in results)
                partial = any(row["status"] == "partial" for row in rows)
                event_status = (
                    "tool_partial" if partial or (ok and failed) else "tool_failed" if failed else "tool_ok"
                )
                state["counts"][event_status + "_rounds"] += 1
                state["counts"]["rounds_with_new_evidence"] += any(r["new_evidence"] for r in results)

        has_feedback = any(
            r.get("error") or any(
                row["status"] not in ("ok", "empty", "exhausted") for row in r.get("by_source", [])
            ) for r in results
        )
        if mapping or invalid_refs or has_feedback or event_status == "context_replan":
            store.failure(
                owner=key, step_id=step, generation_index=generation, attempt_id=aid,
                code="TOOL_FEEDBACK", id_mapping=mapping, invalid_evidence_ids=invalid_refs,
                feedback=[
                    {"call_index": call["index"], "error": r.get("error"), "by_source": r.get("by_source", [])}
                    for call, r in zip(calls, results)
                ], exposed=event_status not in ("context_replan", "submitted"),
            )
        state["status"] = task["status"]
        state["last_event_status"] = event_status
        task.update(state=state, active=None)
        task["elapsed_seconds"] = task.get("elapsed_seconds", 0) + time.monotonic() - started
        store.commit_step(key, task, {
            "step_id": step, "generation_index": generation, "accepted_attempt_id": aid,
            "event_status": event_status, "calls": calls, "results": results,
            "id_mapping": mapping, "invalid_evidence_ids": invalid_refs,
            "ignored_submit_fields": ignored_submit_fields,
            "new_evidence_count": sum(len(r.get("new_evidence", [])) for r in results),
            "remaining_before": remaining(start, c), "remaining_after": remaining(state, c),
            "submit_trigger": task.get("submit_trigger"),
        })
    except CallError as error:
        suffix, skipped = [], []
        if calls:
            skipped = skipped_results(calls, start, "skipped_during_protocol_repair")
            suffix = [assistant_message(message, calls)] + tool_messages(calls, skipped)
        elif isinstance(message, dict) and isinstance(message.get("content"), str) and message["content"].strip():
            assistant = {"role": "assistant", "content": message["content"]}
            if isinstance(message.get("reasoning_content"), str):
                assistant["reasoning_content"] = message["reasoning_content"]
            suffix.append(assistant)
        suffix.append({"role": "user", "content": f"请按工具协议重新输出。{error}"})
        store.failure(
            owner=key, step_id=step, generation_index=generation, attempt_id=aid, code=error.code,
            message=str(error), response=raw, repair_suffix=suffix if not generation else None,
            skipped_calls=skipped, id_mapping=mapping, exposed=bool(suffix[:-1]) and not generation,
        )
        task["elapsed_seconds"] = task.get("elapsed_seconds", 0) + time.monotonic() - started
        if not generation:
            active.update(generation=1, repair_suffix=suffix)
            with store.tx() as db:
                db.execute("UPDATE requests SET rejected=1 WHERE id=?", (aid,))
                db.execute("UPDATE tasks SET data=? WHERE key=?", (dumps(task), key))
        else:
            start["messages"] = messages + suffix[:-1]
            task.update(status="protocol_failure", failure_code=error.code, prediction=None,
                        state=start, active=None)
            store.commit_step(key, task, {
                "step_id": step, "generation_index": generation, "event_status": "invalid_call",
                "code": error.code, "attempt_id": aid, "calls": calls, "results": skipped,
                "remaining_before": remaining(start, c), "remaining_after": remaining(start, c),
            })


def run_sample_turns(store, key, data, c, classes, tools, counter):
    task = store.task(key)
    task.pop("next_retry_at", None)
    task["status"] = "running"
    task["sample_runs"] = task.get("sample_runs", 0) + 1
    store.save_task(key, task)
    turn_started = time.monotonic()
    try:
        while task["status"] == "running":
            turn_started = time.monotonic()
            handle_round(store, key, task, data, c, classes, tools, counter)
        return task["status"], None
    except Exception as error:
        # In-memory state can be ahead of the last successful SQLite commit.
        task = store.task(key)
        if task["status"] == "submitted":
            return "submitted", None
        if isinstance(error, (RetryLater, OSError, sqlite3.OperationalError)):
            status = "retry_pending"
            task["next_retry_at"] = time.time() + integer(c, "RETRY_COOLDOWN_SECONDS", 60)
            code = "TEMPORARY_API" if isinstance(error, RetryLater) else "LOCAL_IO"
        elif isinstance(error, ContextFailure):
            status, code = "context_failure", "CONTEXT_TOO_LARGE"
        elif isinstance(error, ConfigFailure):
            status, code = "config_failure", "CONFIG_FAILURE"
        else:
            status, code = "data_failure", "LOCAL_DATA_OR_IMPLEMENTATION_FAILURE"
        from experiment_io import redact

        task.update(status=status, prediction=None, failure_code=redact(str(error)))
        task["elapsed_seconds"] = task.get("elapsed_seconds", 0) + time.monotonic() - turn_started
        store.save_task(key, task)
        store.failure(
            owner=key,
            code=code,
            message=str(error),
            traceback=traceback.format_exc(),
            exposed=False,
        )
        return status, task.get("next_retry_at")


def run_tasks(store, plan, data, c, classes, tools, counter):
    workers = worker_count(c, "AGENT")
    retry_limit = integer(c, "MAX_RECOVERY_CYCLES_PER_LAUNCH", 3, 1)
    eligible = {"pending", "running", "retry_pending", "config_failure"}
    statuses = store.task_statuses()
    queue = {key: 0.0 for key in plan if statuses[key] in eligible}
    cycles = {}
    for key in queue:
        if statuses[key] == "retry_pending":
            queue[key] = store.task(key).get("next_retry_at", 0)

    def work(key):
        connection = Store(store.path, initialize=False)
        with closing(connection.db):
            return key, *run_sample_turns(connection, key, data, c, classes, tools, counter)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        while queue:
            now = time.time()
            ready = [key for key, timestamp in queue.items() if timestamp <= now]
            if not ready:
                time.sleep(max(0, min(5, min(queue.values()) - now)))
                continue
            for key in ready:
                del queue[key]
            stop_dispatch = False

            def pending():
                for key in ready:
                    if stop_dispatch:
                        break
                    yield key

            for key, status, retry_at in bounded_jobs(executor, pending(), work, workers):
                print("TASK", key, status, flush=True)
                if status == "retry_pending":
                    cycles[key] = cycles.get(key, 0) + 1
                    if cycles[key] < retry_limit:
                        queue[key] = retry_at
                if status == "config_failure" and c.get("_remote"):
                    # Authentication/protocol settings are shared across the pending samples.
                    stop_dispatch = True
            if stop_dispatch:
                break


def run(c):
    folder = config_path(c, "EXPERIMENT_ROOT") / c["EXPERIMENT_ID"]
    with process_lock(folder):
        store = Store(folder / "experiment.sqlite")
        store.pin("config", stage_fingerprint(c, "agent"))
        record_launch(folder, c, "agent")
        store.recover_sent()
        data = load_search_data(config_path(c, "AGENT_GRAPH_BUILD_DIR"))
        store.pin("graph_manifest", data["manifest_hash"])
        fixed = freeze_inputs(c, data, folder, store)
        classes = fixed["classes"]
        tools = tool_schemas(classes, c)
        if (
            store.meta("prompt_schema") is None
            and store.db.execute(
                "SELECT count(*) FROM requests WHERE owner NOT LIKE 'capability:%'"
            ).fetchone()[0]
        ):
            raise ValueError(
                "Legacy experiment has requests without a pinned prompt identity; use a new experiment ID"
            )
        prompts = {m: system_prompt(classes, m) for m in sorted(MODES)}
        store.pin("prompt_schema", digest({"system": prompts, "tools": tools, "submit": FORCE_SUBMIT}))
        if store.meta("source_commit") is None:
            try:
                revision = subprocess.run(
                    ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                    capture_output=True, check=True, text=True, timeout=2,
                ).stdout.strip()
            except (OSError, subprocess.SubprocessError):
                revision = "unavailable"
            store.pin("source_commit", revision)
        counter = TokenCounter(c, tools)
        store.pin("tokenizer_template", counter.identity)
        modes = csv_values(c, "AGENT_MODES")
        repeats = [int(x) for x in csv_values(c, "AGENT_REPEAT_IDS", "0")]
        if not modes or not set(modes) <= MODES or not repeats:
            raise ValueError("Invalid modes/repeats")
        plan = []
        known_tasks = store.task_statuses()
        for rp, repeat in enumerate(repeats):
            for tp, target in enumerate(fixed["targets"]):
                offset = (rp + tp) % len(modes)
                order = modes[offset:] + modes[:offset]
                for mode in order:
                    key = f"{repeat}:{mode}:{target['fine_id']}"
                    plan.append(key)
                    if key not in known_tasks:
                        state = make_sample_state(
                            data, target["fine_id"], mode, classes, c, counter
                        )
                        store.save_task(
                            key,
                            {
                                "repeat_id": repeat,
                                "mode": mode,
                                "target_id": target["fine_id"],
                                "paper_id": target["paper_id"],
                                "status": "pending",
                                "prediction": None,
                                "failure_code": None,
                                "state": state,
                                "active": None,
                            },
                        )
        store.pin("plan", plan)
        store.pin("repeat_ids", repeats)
        store.pin("modes", modes)
        write_json(
            folder / "manifest.json",
            {
                "schema_version": VERSION,
                "experiment_id": c["EXPERIMENT_ID"],
                "algorithm": AGENT_ALGORITHM,
                "git_commit_sha": store.meta("source_commit"),
                "tool_schema_sha256": digest(tools),
                "system_prompt_sha256_by_mode": {m: digest(text) for m, text in prompts.items()},
                "config_hash": store.meta("config"),
                "graph_manifest_sha256": data["manifest_hash"],
                "evaluation_input_sha256": digest(fixed),
            },
        )
        write_json(
            folder / "run_config.json",
            dict(
                public_config(c),
                planned_targets=[x["fine_id"] for x in fixed["targets"]],
                modes=modes,
                repeat_ids=repeats,
                plan=plan,
            ),
        )
        with api_runtime(c, "AGENT") as runtime:
            verify_tool_capability(store, runtime, classes, tools, counter)
            manifest = read_json(folder / "manifest.json")
            manifest["deployment"] = store.meta("AGENT_deployment")
            manifest["prompt_schema_sha256"] = store.meta("prompt_schema")
            write_json(folder / "manifest.json", manifest)
            run_tasks(store, plan, data, runtime, classes, tools, counter)
        store.export(folder / "exports")
        incomplete = sum(
            status not in ("submitted", "protocol_failure", "context_failure")
            for status in store.task_statuses().values()
        )
        print("INFERENCE_FINISHED", folder, "incomplete=", incomplete, flush=True)
        return 2 if incomplete else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="experiment.env")
    parser.add_argument("--target-split", choices=TARGET_SPLITS)
    parser.add_argument("--target-limit", type=int)
    parser.add_argument("--target-seed", type=int)
    parser.add_argument("--target-plan", help="Use an existing evaluation node plan")
    parser.add_argument("--experiment-id")
    args = parser.parse_args()
    c = load_env(args.env, model_stage="AGENT")
    if args.experiment_id:
        c["EXPERIMENT_ID"] = args.experiment_id
    if args.target_plan:
        c["AGENT_TARGET_PLAN"] = args.target_plan
    if c.get("AGENT_TARGET_PLAN") and any(
        value is not None for value in (args.target_split, args.target_limit, args.target_seed)
    ):
        parser.error("A target plan cannot be combined with explicit split, limit, or seed flags")
    for name in ("split", "limit", "seed"):
        value = getattr(args, "target_" + name)
        if value is not None:
            c["AGENT_TARGET_" + name.upper()] = str(value)
    for cycle in range(integer(c, "MAX_RECOVERY_CYCLES_PER_LAUNCH", 3, 1)):
        try:
            raise SystemExit(run(c))
        except RetryLater:
            if cycle + 1 < integer(c, "MAX_RECOVERY_CYCLES_PER_LAUNCH", 3, 1):
                time.sleep(integer(c, "RETRY_COOLDOWN_SECONDS", 60))
        except StageBusy as error:
            raise SystemExit(str(error)) from None
        except Exception as e:
            folder = config_path(c, "EXPERIMENT_ROOT") / c["EXPERIMENT_ID"]
            with process_lock(folder):
                store = Store(folder / "experiment.sqlite")
                store.failure(
                    owner="startup",
                    code=type(e).__name__,
                    message=str(e),
                    traceback=traceback.format_exc(),
                    exposed=False,
                )
                store.export(folder / "exports")
            raise
    raise SystemExit(2)


if __name__ == "__main__":
    main()
