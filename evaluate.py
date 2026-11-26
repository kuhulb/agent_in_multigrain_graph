"""Read-only, model-free evaluation. evaluate_rows also works on tiny in-memory inputs."""

from __future__ import annotations

import argparse
import csv
from contextlib import closing
import io
import json
from pathlib import Path
import sqlite3

from experiment_io import (
    TOKEN_FIELDS,
    atomic_bytes,
    config_path,
    csv_values,
    digest,
    load_env,
    read_json,
    result_for_task,
    usage_for,
    write_json,
    write_jsonl,
)

STATUSES = {
    "pending",
    "running",
    "retry_pending",
    "config_failure",
    "data_failure",
    "context_failure",
    "protocol_failure",
    "submitted",
}
METHOD_TERMINAL = {"submitted", "protocol_failure"}
EVAL_VERSION = 6


def token_summary(usages):
    totals = {}
    for field in TOKEN_FIELDS:
        values = [u[field] for u in usages if u.get(field) is not None]
        complete = [
            u[field] for u in usages if u.get("complete", {}).get(field) and u.get(field) is not None
        ]
        totals["known_" + field] = sum(values) if values else None
        totals[field + "_known_requests"] = sum(
            u.get("known_counts", {}).get(field, 0) for u in usages
        )
        totals[field + "_complete_n"] = len(complete)
        totals[field + "_complete_mean"] = sum(complete) / len(complete) if complete else None
    for field in (
        "outcome_unknown",
        "pending_requests",
        "usage_missing_requests",
        "usage_invalid_requests",
        "rejected_attempts",
        "rejected_total_tokens_known_requests",
        "finish_reason_known_requests",
        "finish_length_requests",
        "content_observed_requests",
        "content_nonempty_requests",
        "content_chars",
        "content_tokens_known_requests",
        "cache_pair_input_tokens",
        "cache_pair_cached_tokens",
        "cache_known_requests",
        "cache_positive_requests",
    ):
        totals[field] = sum(u.get(field, 0) for u in usages)
    attempts = sum(u.get("attempts", 0) for u in usages)
    paired_input = totals["cache_pair_input_tokens"]
    paired_cache = totals["cache_pair_cached_tokens"]
    cache_known = totals["cache_known_requests"]
    rejected_totals = [
        u["known_rejected_total_tokens"] for u in usages
        if u.get("known_rejected_total_tokens") is not None and u.get("rejected_attempts")
    ]
    content_token_counts = [u["known_content_tokens"] for u in usages if u.get("known_content_tokens") is not None]
    totals.update(
        request_attempts=attempts,
        input_usage_complete_n=totals["input_tokens_complete_n"],
        known_rejected_total_tokens=sum(rejected_totals)
        if rejected_totals else None if totals["rejected_attempts"] else 0,
        max_output_tokens=max(
            (u["max_output_tokens"] for u in usages if u.get("max_output_tokens") is not None),
            default=None,
        ),
        content_nonempty_ratio=totals["content_nonempty_requests"] / totals["content_observed_requests"]
        if totals["content_observed_requests"] else None,
        known_content_tokens=sum(content_token_counts) if content_token_counts else None,
        cache_pair_uncached_tokens=paired_input - paired_cache,
        cache_coverage=cache_known / attempts if attempts else None,
        cache_token_ratio=paired_cache / paired_input if paired_input else None,
        cache_hit_request_ratio=totals["cache_positive_requests"] / cache_known
        if cache_known else None,
    )
    return totals


def metrics(rows, truth, classes):
    if not rows:
        return {"n": 0, "accuracy": None, "macro_f1": None}
    correct = sum(
        r["status"] == "submitted" and r["prediction"] == truth[r["target_id"]] for r in rows
    )
    f1 = []
    for label in classes:
        tp = fp = fn = 0
        for row in rows:
            actual = truth[row["target_id"]]
            pred = row["prediction"] if row["status"] == "submitted" else object()
            tp += actual == label and pred == label
            fp += actual != label and pred == label
            fn += actual == label and pred != label
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
    return {"n": len(rows), "accuracy": correct / len(rows), "macro_f1": sum(f1) / len(f1)}


def evaluate_rows(results, label_data, options):
    targets = options["planned_targets"]
    modes = options["modes"]
    repeats = options["repeat_ids"]
    if any(len(x) != len(set(x)) or not x for x in (targets, modes, repeats)):
        raise ValueError("Nonempty unique plan fields required")
    classes = [x["class_id"] for x in label_data["classes"]]
    if not classes or len(classes) != len(set(classes)):
        raise ValueError("Duplicate/empty classes")
    truth = {}
    for row in label_data["targets"]:
        target = row["fine_id"]
        if type(target) is not int or target < 0 or target in truth:
            raise ValueError("Duplicate or invalid evaluation target ID")
        truth[target] = row["truth"]
    class_set = set(classes)
    planned = set(targets)
    if not planned <= truth.keys() or any(v not in class_set for v in truth.values()):
        raise ValueError("Invalid label plan")
    group_keys = {(r, m) for r in repeats for m in modes}
    indexed = {}
    for row in results:
        key = (row["repeat_id"], row["mode"], row["target_id"])
        if key in indexed:
            raise ValueError("Duplicate task key")
        if row["status"] not in STATUSES:
            raise ValueError("Unknown status")
        if row["status"] == "submitted":
            if row.get("prediction") not in classes:
                raise ValueError("Invalid prediction")
        elif row.get("prediction") is not None:
            raise ValueError("Failed/pending task contains a prediction")
        if key[:2] not in group_keys or key[2] not in planned:
            raise ValueError("Unexpected result outside requested plan")
        indexed[key] = row
    normalized = []
    summary = []
    paired = []
    for repeat in repeats:
        for mode in modes:
            for target in targets:
                row = indexed.get(
                    (repeat, mode, target),
                    {
                        "repeat_id": repeat,
                        "mode": mode,
                        "target_id": target,
                        "status": "missing",
                        "prediction": None,
                    },
                )
                normalized.append(
                    dict(
                        row,
                        truth=truth[target],
                        correct=row["prediction"] == truth[target]
                        if row["status"] == "submitted"
                        else None,
                    )
                )
    lookup = {(r["repeat_id"], r["mode"], r["target_id"]): r for r in normalized}
    for repeat in repeats:
        common = [
            t
            for t in targets
            if all(lookup[repeat, m, t]["status"] in METHOD_TERMINAL for m in modes)
        ]
        for mode in modes:
            rows = [lookup[repeat, mode, t] for t in targets]
            scored = metrics([lookup[repeat, mode, t] for t in common], truth, classes)
            planned_scores = metrics(rows, truth, classes)
            summary.append(
                {
                    "repeat_id": repeat,
                    "mode": mode,
                    "planned_n": len(targets),
                    "submitted": sum(r["status"] == "submitted" for r in rows),
                    "submission_rate": sum(r["status"] == "submitted" for r in rows) / len(targets),
                    "end_to_end_accuracy": sum(
                        r["status"] == "submitted" and r["prediction"] == truth[r["target_id"]]
                        for r in rows
                    )
                    / len(targets),
                    "end_to_end_macro_f1": planned_scores["macro_f1"],
                    "failure_rate": sum(
                        r["status"] in {"protocol_failure", "context_failure", "config_failure", "data_failure"}
                        for r in rows
                    ) / len(targets),
                    "common_n": len(common),
                    "common_accuracy": scored["accuracy"],
                    "common_macro_f1": scored["macro_f1"],
                    **{
                        "status_" + s: sum(r["status"] == s for r in rows)
                        for s in sorted(STATUSES | {"missing"})
                    },
                    **token_summary([r.get("usage", {}) for r in rows]),
                }
            )
            for field in (
                "graph_tool_calls",
                "coarse_center_queries",
                "model_rounds",
                "new_fine_nodes",
                "community_calls",
                "retrieved_chars",
                "initial_evidence_chars",
                "distinct_accessed_coarse_nodes",
                "distinct_accessed_fine_nodes",
                "request_attempts",
            ):
                values = [
                    r.get("counts", {}).get(field)
                    for r in rows
                    if r.get("counts", {}).get(field) is not None
                ]
                summary[-1][field + "_mean"] = sum(values) / len(values) if values else None
            for field in (
                "tool_ok_rounds", "tool_partial_rounds", "tool_failed_rounds", "rounds_with_new_evidence",
                "model_rounds", "api_attempts", "api_retries", "repair_attempts", "logical_queries",
                "tool_calls", "invalid_queries", "empty_queries", "skipped_queries", "sample_recoveries",
            ):
                values = [r["counts"][field] for r in rows if field in r.get("counts", {})]
                summary[-1][field] = sum(values) if values else None
            elapsed = [r["elapsed_seconds"] for r in rows if "elapsed_seconds" in r]
            summary[-1]["sample_elapsed_seconds_sum"] = sum(elapsed) if elapsed else None
            summary[-1]["sample_elapsed_measured_n"] = len(elapsed)
            submitted = [r for r in rows if r["status"] == "submitted"]
            citations = [r["citations"] for r in submitted if "citations" in r]
            summary[-1]["citation_measured_n"] = len(citations)
            for field in ("target", "initial_community", "retrieved"):
                summary[-1]["cites_" + field + "_n"] = sum(c[field] for c in citations)
            triggers = [r["submit_trigger"] for r in submitted if "submit_trigger" in r]
            summary[-1]["submit_trigger_measured_n"] = len(triggers)
            for trigger in (
                "model", "target_only", "context_replan", "round_limit", "graph_budget",
                "no_available_query",
            ):
                summary[-1]["submit_trigger_" + trigger + "_n"] = triggers.count(trigger)
            failed_predecessors = [
                r["submit_after_failed_tool_round"] for r in submitted
                if "submit_after_failed_tool_round" in r
            ]
            summary[-1]["submit_after_failed_tool_round_n"] = (
                sum(failed_predecessors) if failed_predecessors else None
            )
            if mode in ("coarse_only", "multi"):
                measured = [
                    r["coarse_evidence_delivered"] for r in submitted if "coarse_evidence_delivered" in r
                ]
                summary[-1]["coarse_evidence_measured_n"] = len(measured)
                summary[-1]["coarse_evidence_delivered_n"] = sum(measured)
            # Only submitted rows with the trace field present can qualify as early stops.
            if mode == "multi":
                for source, label in (
                    ("first_fine_request_round", "no_fine_request"),
                    ("first_new_fine_round", "no_new_fine_evidence"),
                ):
                    measured = [r for r in submitted if source in r]
                    stopped = [r for r in measured if r[source] is None]
                    summary[-1].update(
                        {
                            label + "_n": len(stopped),
                            label + "_measured_n": len(measured),
                            label + "_rate": len(stopped) / len(measured) if measured else None,
                        }
                    )
                checkpoints = [
                    r
                    for r in submitted
                    if (r.get("coarse_checkpoint") or {}).get("prediction") in class_set
                ]
                if any("coarse_checkpoint" in r for r in rows):
                    summary[-1].update(
                        coarse_checkpoint_n=len(checkpoints),
                        wrong_to_correct_n=sum(
                            r["coarse_checkpoint"]["prediction"] != r["truth"] and r["correct"]
                            for r in checkpoints
                        ),
                        correct_to_wrong_n=sum(
                            r["coarse_checkpoint"]["prediction"] == r["truth"] and not r["correct"]
                            for r in checkpoints
                        ),
                    )
        if {"multi", "fine_only"} <= set(modes):
            pair_targets = [
                t
                for t in targets
                if all(
                    lookup[repeat, m, t]["status"] in METHOD_TERMINAL
                    for m in ("multi", "fine_only")
                )
            ]
            pair = {
                "repeat_id": repeat,
                "left": "multi",
                "right": "fine_only",
                "method_common_n": len(pair_targets),
            }
            for mode in ("multi", "fine_only"):
                pair[mode + "_accuracy"] = metrics(
                    [lookup[repeat, mode, t] for t in pair_targets], truth, classes
                )["accuracy"]
            for field in TOKEN_FIELDS:
                matched = [
                    t
                    for t in targets
                    if all(
                        lookup[repeat, m, t]["status"] == "submitted"
                        and lookup[repeat, m, t]
                        .get("usage", {})
                        .get("complete", {})
                        .get(field, False)
                        and lookup[repeat, m, t]["usage"].get(field) is not None
                        for m in ("multi", "fine_only")
                    )
                ]
                differences = [
                    lookup[repeat, "multi", t]["usage"][field]
                    - lookup[repeat, "fine_only", t]["usage"][field]
                    for t in matched
                ]
                pair[field + "_paired_n"] = len(matched)
                pair[field + "_mean_difference"] = (
                    sum(differences) / len(differences) if differences else None
                )
            paired.append(pair)
    return {"normalized_rows": normalized, "summary_rows": summary, "paired_rows": paired}


def csv_output(path, rows):
    if not rows:
        atomic_bytes(path, b"")
        return
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=sorted({k for r in rows for k in r}))
    writer.writeheader()
    writer.writerows(rows)
    atomic_bytes(path, buffer.getvalue().encode("utf-8-sig"))


def evaluate(folder, options):
    folder = Path(folder)
    manifest = read_json(folder / "manifest.json")
    labels = read_json(folder / "evaluation_inputs.json")
    labels_hash = digest(labels)
    if labels_hash != manifest["evaluation_input_sha256"]:
        raise ValueError("Evaluation snapshot hash mismatch")
    uri = (folder / "experiment.sqlite").resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri, uri=True, timeout=30)) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")
        meta = {r["key"]: json.loads(r["value"]) for r in db.execute("SELECT * FROM meta")}
        if meta.get("evaluation_input_sha256") != labels_hash or meta.get("config") != manifest.get(
            "config_hash"
        ):
            raise ValueError(
                "Database, evaluation labels and manifest do not describe the same run"
            )
        modes = options.get("modes") if options.get("modes") is not None else meta["modes"]
        repeats = (
            options.get("repeat_ids")
            if options.get("repeat_ids") is not None
            else meta["repeat_ids"]
        )
        if not set(modes) <= set(meta["modes"]) or not set(repeats) <= set(meta["repeat_ids"]):
            raise ValueError("Unknown modes/repeats")
        targets = [x["fine_id"] for x in labels["targets"]]
        if options.get("targets") is not None:
            chosen_targets = set(options["targets"])
            if len(chosen_targets) != len(options["targets"]) or not chosen_targets <= set(targets):
                raise ValueError("Invalid target filter")
            targets = [t for t in targets if t in chosen_targets]
        if options.get("limit") is not None:
            if type(options["limit"]) is not int or options["limit"] <= 0:
                raise ValueError("limit must be a positive integer")
            targets = targets[: options["limit"]]
        records = []
        capability_usage = usage_for(db, capability_only=True)
        for repeat in repeats:
            for mode in modes:
                for target in targets:
                    key = f"{repeat}:{mode}:{target}"
                    row = db.execute("SELECT status,data FROM tasks WHERE key=?", (key,)).fetchone()
                    if row is None:
                        continue
                    task = json.loads(row["data"])
                    if (task["repeat_id"], task["mode"], task["target_id"], task["status"]) != (
                        repeat,
                        mode,
                        target,
                        row["status"],
                    ):
                        raise ValueError(f"Task key/status mismatch for {key}")
                    records.append(result_for_task(db, key, task, include_trace=False))
    chosen = {"planned_targets": targets, "modes": modes, "repeat_ids": repeats}
    result = evaluate_rows(records, labels, chosen)
    selection = digest(chosen)[:16]
    snapshot = digest(
        {
            "manifest": manifest,
            "commit_seq": meta["commit_seq"],
            "records": records,
            "capability_usage": capability_usage,
            "evaluation_version": EVAL_VERSION,
        }
    )[:16]
    output = folder / "evaluations" / selection / snapshot
    write_jsonl(output / "selected_results.jsonl", result["normalized_rows"])
    csv_output(output / "summary.csv", result["summary_rows"])
    csv_output(output / "paired_summary.csv", result["paired_rows"])
    write_json(output / "capability_usage.json", capability_usage)
    write_json(
        output / "eval_config.json",
        dict(
            chosen,
            snapshot_hash=snapshot,
            commit_seq=meta["commit_seq"],
            evaluation_input_sha256=labels_hash,
            evaluation_version=EVAL_VERSION,
        ),
    )
    print("EVALUATION", output)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="experiment.env")
    p.add_argument("--experiment-dir")
    p.add_argument("--limit", type=int)
    p.add_argument("--modes", nargs="+")
    p.add_argument("--repeat-ids", type=int, nargs="+")
    p.add_argument("--targets", type=int, nargs="+")
    args = p.parse_args()
    c = load_env(args.env)
    if args.experiment_dir:
        c["EVAL_EXPERIMENT_DIR"] = args.experiment_dir
    folder = (
        config_path(c, "EVAL_EXPERIMENT_DIR")
        if c.get("EVAL_EXPERIMENT_DIR")
        else config_path(c, "EXPERIMENT_ROOT") / c["EXPERIMENT_ID"]
    )
    options = {
        "limit": args.limit
        if args.limit is not None
        else int(c["EVAL_LIMIT"])
        if c.get("EVAL_LIMIT")
        else None,
        "modes": args.modes or csv_values(c, "EVAL_MODES") or None,
        "repeat_ids": args.repeat_ids
        if args.repeat_ids is not None
        else [int(x) for x in csv_values(c, "EVAL_REPEAT_IDS")] or None,
        "targets": args.targets,
    }
    evaluate(folder, options)


if __name__ == "__main__":
    main()
