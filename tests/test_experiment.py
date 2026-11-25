import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main
from build_coarse_graph import chunk_ranges, load_raw_graph, make_coarse, summarize_pack
from evaluate import evaluate_rows
from experiment_io import Store, request_model, usage_for, write_jsonl


@pytest.fixture
def config():
    return {
        "AGENT_MAX_GRAPH_TOOL_CALLS": "6",
        "AGENT_MAX_CENTER_QUERIES": "6",
        "AGENT_MAX_DISTINCT_COMMUNITIES": "2",
        "AGENT_MAX_MODEL_ROUNDS": "14",
        "AGENT_MAX_SEARCH_CALLS_PER_ROUND": "4",
        "AGENT_MIN_K": "1",
        "AGENT_MAX_K": "10",
        "AGENT_MAX_OUTPUT_TOKENS": "20",
        "AGENT_CONTEXT_SAFETY_TOKENS": "0",
        "AGENT_CONTEXT_WINDOW_TOKENS": "10000",
        "AGENT_LLM_MODEL": "test",
        "AGENT_TEMPERATURE": "0",
        "AGENT_REGENERATE_TEMPERATURE": ".2",
        "AGENT_SEED": "234",
        "AGENT_ENABLE_THINKING": "false",
        "AGENT_THINKING_CONTROL": "non_thinking_model",
    }


@pytest.fixture
def data():
    edges = [[0, 1], [1, 2], [1, 4], [2, 3], [3, 4], [4, 5]]
    adj = [[] for _ in range(6)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    raw = {
        "papers": [{"title": f"Title {i}", "abstract": f"Abstract {i}"} for i in range(6)],
        "paper_ids": [str(i) for i in range(6)],
        "fine_edges": edges,
        "fine_adj": adj,
        "graph_sha256": "test",
    }
    partition = {
        "communities": [
            {"members": group, "original_stats": {"level": "Core"}}
            for group in [[0, 1], [2, 3], [4, 5]]
        ],
        "source_sha256": "test",
        "weight_convention": "unit",
    }
    coarse = make_coarse(raw, partition, "toy")
    for row in coarse["communities"]:
        row.update(
            summary="summary " + str(row["id"]),
            keywords=[],
            summary_source="llm",
            support_fine_ids=[],
        )
    return {
        "raw": raw,
        "coarse": coarse,
        "fine_adj": adj,
        "coarse_adj": [{1: 1, 2: 1}, {0: 1, 2: 1}, {0: 1, 1: 1}],
        "members": [[0, 1], [2, 3], [4, 5]],
        "mapping": [0, 0, 1, 1, 2, 2],
        "fine_texts": [f"complete fine text {i}" for i in range(6)],
        "coarse_texts": [f"complete coarse text {i}" for i in range(3)],
        "vectors": {
            "fine": np.array(
                [[1.0, 0], [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.6, 0.4], [0.5, 0.5]]
            ),
            "coarse": np.array([[1.0, 0], [0.9, 0.1], [0.8, 0.2]]),
        },
    }


CLASSES = [{"class_id": "a", "name": "A"}, {"class_id": "b", "name": "B"}]


def test_stratified_plan_has_exact_size_coverage_and_fixed_order():
    from scripts.prepare_eval_plan import sample_targets

    rows = [
        {"paper_id": f"p{i}", "truth": "a" if i < 90 else "b", "split": "test"} for i in range(100)
    ]
    rows.extend({"paper_id": f"v{i}", "truth": "b", "split": "val"} for i in range(10))
    ids, counts, population = sample_targets(rows, "test", 20, 234)
    assert counts == {"a": 18, "b": 2} and population == 100
    assert len(set(ids)) == 20 and all(value.startswith("p") for value in ids)
    assert sample_targets(list(reversed(rows)), "test", 20, 234)[0] == ids
    assert sample_targets(rows, "test", 2, 234)[1] == {"a": 1, "b": 1}
    with pytest.raises(ValueError, match="Sample size"):
        sample_targets(rows, "test", 101, 234)


def test_fixed_plan_validation_and_resume(tmp_path):
    from experiment_io import write_json

    cache = tmp_path / "experiment"
    store = Store(cache / "experiment.sqlite")
    write_json(tmp_path / "classes.json", CLASSES)
    (tmp_path / "targets.tsv").write_text(
        "paper_id\tclass_id\tsplit\np0\ta\ttrain\np1\ta\ttest\np2\tb\ttest\n"
    )
    c = {
        "_base": str(tmp_path),
        "DATA_DIR": str(tmp_path),
        "DATASET": "toy",
        "AGENT_TARGET_PLAN": "plan.json",
    }
    raw = {"graph_sha256": "graph-v1", "paper_id_to_fine": {"p0": 0, "p1": 1, "p2": 2}}
    data = {"raw": raw, "coarse": {"dataset": "toy"}}
    _, rows, sources = main.load_evaluation_labels(c, raw)
    plan = {
        "schema_version": 1,
        "dataset": "toy",
        "source_graph_sha256": "graph-v1",
        "label_source": sources,
        "split": "test",
        "size": 2,
        "seed": 234,
        "paper_ids": ["p2", "p1"],
    }
    with pytest.raises(ValueError, match="outside"):
        main.select_plan_targets(
            dict(plan, paper_ids=["p1", "p0"]), rows, "toy", "graph-v1", sources
        )
    with pytest.raises(ValueError, match="label/split version"):
        main.select_plan_targets(plan, rows, "toy", "graph-v1", {})
    write_json(tmp_path / "plan.json", plan)
    fixed = main.freeze_inputs(c, data, cache, store)
    assert [row["fine_id"] for row in fixed["targets"]] == [2, 1]
    assert main.freeze_inputs(c, data, cache, store) == fixed
    write_json(tmp_path / "plan.json", dict(plan, paper_ids=["p1", "p2"]))
    with pytest.raises(ValueError, match="Target plan changed"):
        main.freeze_inputs(c, data, cache, store)


def call(name, args, id="call_1", index=0):
    return {"name": name, "arguments": json.dumps(args), "id": id, "index": index}


def search(center, k=1):
    return {"center": center, "k": k}


def test_graph_ids_edges_and_labels_are_separate(tmp_path):
    write_jsonl(
        tmp_path / "papers.jsonl",
        [
            {"paper_id": "002", "title": "b", "abstract": ""},
            {"paper_id": "001", "title": "a", "abstract": ""},
            {"paper_id": "010", "title": "c", "abstract": ""},
        ],
    )
    (tmp_path / "citations.tsv").write_text("source_id\ttarget_id\n002\t001\n001\t002\n001\t001\n")
    raw = load_raw_graph(tmp_path)
    assert (
        raw["paper_ids"] == ["001", "002", "010"]
        and raw["fine_edges"] == [[0, 1]]
        and raw["fine_adj"][2] == []
    )
    assert "truth" not in raw


def test_packing_skips_truly_empty_original_text(data):
    data["raw"]["papers"][0] = {"title": "", "abstract": ""}
    packed, supports, _ = summarize_pack(data["raw"], [0, 1])
    assert supports == [1] and "[Node 0]" in packed


@pytest.mark.parametrize("length", [1, 8189, 8190, 8191, 16380, 30000])
def test_embedding_windows_cover_tail(length):
    blocks = list(chunk_ranges(length, 8190, 128))
    assert sum(x[2] for x in blocks) == length and blocks[-1][1] == length
    assert all(b - a + 2 <= 8192 for a, b, _ in blocks)


def test_fine_only_has_no_community_signal(data, config):
    s = main.make_sample_state(data, 0, "fine_only", CLASSES, config)
    assert main.evidence(data, s, "fine", 1)["structure"]["same_community_as_target"] is None
    with pytest.raises(main.CallError):
        main.query(
            data,
            s,
            copy.deepcopy(s),
            call("search_1hop", {}),
            {"center": "F:0", "k": 1, "coarse_checkpoint": {"prediction": "a"}},
            config,
            1,
        )


def test_same_parameters_are_continuation(data, config):
    s = main.make_sample_state(data, 0, "multi", CLASSES, config)
    start = copy.deepcopy(s)
    c = call("search_1hop", {})
    args = search("C:0")
    first = main.query(data, s, start, c, args, config, 1)[0]
    second = main.query(data, s, start, c, args, config, 1)[0]
    assert first["new_evidence"][0]["ref"] == "C:1" and second["new_evidence"][0]["ref"] == "C:2"
    assert first["new_evidence"][0]["structure"]["similarity_to_target"] == 0.9
    assert first["by_source"][0]["new_edges"][0]["weight"] == 1
    assert s["counts"]["graph_tool_calls"] == 2


def test_target_is_seen_and_known_edge_does_not_take_new_k(data, config):
    s = main.make_sample_state(data, 0, "fine_only", CLASSES, config)
    s["seen_fine"].append(1)
    s["first_seen_step"]["F:1"] = 1
    result = main.query(
        data,
        s,
        copy.deepcopy(s),
        call("search_1hop", {}),
        search("F:1"),
        config,
        2,
    )[0]
    assert result["new_evidence"][0]["ref"] == "F:2" and result["already_read"][0]["ref"] == "F:0"


def test_coarse_gate_is_round_start_based(data, config):
    s = main.make_sample_state(data, 0, "multi", CLASSES, config)
    s["counts"]["model_rounds"] = 1
    calls = [
        call("search_1hop", search("C:0")),
        call("search_1hop", search("F:0"), id="call_2", index=1),
        call("search_1hop", search("C:0", k=0), id="call_3", index=2),
    ]
    with pytest.raises(main.CallError) as raised:
        main.pack_tool_batch(data, s, {}, calls, config, lambda _: 1, True)
    assert [issue["message"] for issue in raised.value.issues] == [
        "Read coarse evidence before fine query", "Invalid k"
    ]
    assert s["seen_coarse"] == [0] and s["counts"]["graph_tool_calls"] == 0
    coarse, _, _ = main.pack_tool_batch(data, s, {}, calls[:1], config, lambda _: 1, True)
    assert coarse["fine_access_reason"] == "coarse_evidence"
    coarse["counts"]["model_rounds"] = 2
    fine, results, _ = main.pack_tool_batch(
        data, coarse, {}, calls[1:2], {**config, "AGENT_SHOW_SIMILARITY": "false"},
        lambda _: 1, True,
    )
    assert fine["first_fine_request_round"] == 2
    assert results[0]["new_evidence"][0]["ref"] == "F:1"
    assert "similarity_to_target" not in results[0]["new_evidence"][0]["structure"]


def test_id_repair_never_falsely_claims_next_page_read(data, config):
    s = main.make_sample_state(data, 0, "multi", CLASSES, config)
    calls = [
        call("search_1hop", search("C:0")),
        call("search_1hop", search("C:0"), index=1),
    ]
    with pytest.raises(main.CallError):
        main.pack_tool_batch(data, s, {}, copy.deepcopy(calls), config, lambda _: 1, True)
    final, results, debug = main.pack_tool_batch(
        data, s, {}, calls, config, lambda _: 1, False
    )
    assert final["counts"]["graph_tool_calls"] == 1
    assert (
        results[1]["error"]["code"] == "DUPLICATE_CALL_ID"
        and not results[1]["already_read"]
        and debug["id_mapping"]
    )


def test_expand_consumes_shared_budget(data, config):
    s = main.make_sample_state(data, 0, "multi", CLASSES, config)
    s["fine_access_reason"] = "coarse_evidence"
    args = {
        "community": "C:0",
        "k": 1,
    }
    result = main.query(
        data, s, copy.deepcopy(s), call("expand_community", {}), args, config, 1
    )[0]
    assert (
        result["spent"]["graph_tool_calls"] == 1
        and result["spent"]["center_queries"] == 0
        and s["first_fine_request_round"] == 1
        and "coarse_checkpoint" not in s
    )


def test_capacity_rollback_does_not_pollute_seen(data, config):
    s = main.make_sample_state(data, 0, "fine_only", CLASSES, config)
    calls = [call("search_1hop", search("F:0"))]
    # Every tool response is too big, but the earlier original history remains usable.
    counter = lambda messages: 100000 if any(m["role"] == "tool" for m in messages) else 1
    final, _, debug = main.pack_tool_batch(data, s, {}, calls, config, counter, True)
    assert final is None and debug["code"] == "CONTEXT_REPLAN" and s["seen_fine"] == [0]


def test_mixed_valid_submit_precedes_bad_query_id(data, config, tmp_path, monkeypatch):
    config["AGENT_MODEL_MODE"] = "api"
    s = main.make_sample_state(data, 0, "target_only", CLASSES, config)
    original = copy.deepcopy(s["messages"])
    response = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"id": None, "function": {"name": "search_1hop", "arguments": "BROKEN"}},
                        {
                            "id": None,
                            "function": {
                                "name": "submit",
                                "arguments": json.dumps(
                                    {
                                        "prediction": "a",
                                        "reason": "target evidence",
                                        "evidence_ids": ["F:0"],
                                    }
                                ),
                            },
                        },
                    ]
                }
            }
        ]
    }
    def fake(*args):
        assert args[-1]["messages"] == original + [
            {"role": "user", "content": "请立刻根据已有信息提交"}
        ]
        return response, "attempt"

    monkeypatch.setattr(main, "request_model", fake)
    task = {
        "status": "running",
        "state": s,
        "active": None,
        "repeat_id": 0,
        "target_id": 0,
        "prediction": None,
    }
    store = Store(tmp_path / "experiment.sqlite")
    main.handle_round(
        store, "sample", task, data, config, CLASSES, main.tool_schemas(CLASSES), lambda _: 1
    )
    assert task["status"] == "submitted" and task["state"]["counts"]["graph_tool_calls"] == 0
    assert json.loads(task["state"]["messages"][-2]["content"])["status"] == "SKIPPED_AFTER_SUBMIT"


def test_missing_tasks_do_not_enter_method_intersection():
    labels = {"classes": CLASSES, "targets": [{"fine_id": x, "truth": "a"} for x in range(3)]}
    rows = [
        {"repeat_id": 0, "mode": "multi", "target_id": 0, "status": "submitted", "prediction": "a"},
        {"repeat_id": 0, "mode": "multi", "target_id": 1, "status": "submitted", "prediction": "b"},
        {
            "repeat_id": 0,
            "mode": "fine_only",
            "target_id": 0,
            "status": "submitted",
            "prediction": "a",
        },
        {
            "repeat_id": 0,
            "mode": "fine_only",
            "target_id": 1,
            "status": "retry_pending",
            "prediction": None,
        },
        {
            "repeat_id": 0,
            "mode": "fine_only",
            "target_id": 2,
            "status": "submitted",
            "prediction": "a",
        },
    ]
    options = {"planned_targets": [0, 1, 2], "modes": ["multi", "fine_only"], "repeat_ids": [0]}
    result = evaluate_rows(rows, labels, options)
    assert [x["end_to_end_accuracy"] for x in result["summary_rows"]] == [1 / 3, 2 / 3]
    assert all(x["common_n"] == 1 for x in result["summary_rows"])
    with pytest.raises(ValueError):
        evaluate_rows(rows + [rows[0]], labels, options)


def test_saved_response_reused_without_new_tokens(tmp_path, monkeypatch):
    import httpx

    store = Store(tmp_path / "store.sqlite")
    calls = []

    def post(*args, **kwargs):
        calls.append(1)
        return httpx.Response(
            200,
            json={
                "choices": [],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 3,
                    "prompt_tokens_details": {"cached_tokens": 4},
                },
            },
        )

    monkeypatch.setattr(httpx, "post", post)
    c = {"AGENT_LLM_BASE_URL": "http://localhost/v1", "AGENT_MAX_HTTP_ATTEMPTS": "1"}
    first = request_model(store, c, "AGENT", "task", 1, 0, {"model": "x"})
    second = request_model(store, c, "AGENT", "task", 1, 0, {"model": "x"})
    assert first == second and len(calls) == 1
    usage = usage_for(store.db, "task")
    assert (
        usage["input_tokens"] == 12
        and usage["uncached_input_tokens"] == 8
        and usage["total_tokens"] == 15
        and usage["attempts"] == 1
    )


def test_overlarge_first_candidate_does_not_starve_smaller_neighbor(data, config):
    s = main.make_sample_state(data, 0, "fine_only", CLASSES, config)
    s["seen_fine"].append(1)
    s["first_seen_step"]["F:1"] = 1
    s["counts"]["model_rounds"] = 1
    counter = (
        lambda messages: 100000
        if any(
            "complete fine text 2" in m.get("content", "") for m in messages if m["role"] == "tool"
        )
        else 1
    )
    final, results, _ = main.pack_tool_batch(
        data, s, {}, [call("search_1hop", search("F:1"))], config, counter, True
    )
    assert (
        final is not None
        and results[0]["new_evidence"][0]["ref"] == "F:4"
        and 2 not in final["seen_fine"]
    )


@pytest.mark.parametrize("remote", [False, True])
def test_round_thirteen_can_query_and_fourteen_must_submit(
    data, config, tmp_path, monkeypatch, remote
):
    if remote:
        config["AGENT_MODEL_MODE"] = "api"
    s = main.make_sample_state(data, 0, "fine_only", CLASSES, config)
    s["counts"]["model_rounds"] = 12
    payloads = []
    tools = main.tool_schemas(CLASSES)

    def fake(store, c, prefix, key, step, generation, payload):
        payloads.append(payload)
        query = step == 13 or (remote and generation == 0)
        args = (
            search("F:0")
            if query
            else {"prediction": "a", "reason": "existing evidence", "evidence_ids": ["F:0"]}
        )
        return {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": f"c{step}",
                                "function": {
                                    "name": "search_1hop" if query else "submit",
                                    "arguments": json.dumps(args),
                                },
                            }
                        ]
                    }
                }
            ]
        }, str(step)

    monkeypatch.setattr(main, "request_model", fake)
    task = {
        "status": "running",
        "state": s,
        "active": None,
        "repeat_id": 0,
        "target_id": 0,
        "prediction": None,
    }
    store = Store(tmp_path / "test.sqlite")
    main.handle_round(store, "t", task, data, config, CLASSES, tools, lambda _: 1)
    assert task["status"] == "running" and task["state"]["counts"]["graph_tool_calls"] == 1
    main.handle_round(store, "t", task, data, config, CLASSES, tools, lambda _: 1)
    if remote:
        assert task["status"] == "running" and task["active"]["generation"] == 1
        assert store.db.execute("SELECT data FROM failures").fetchone() is not None
        main.handle_round(store, "t", task, data, config, CLASSES, tools, lambda _: 1)
        assert [p["tool_choice"] for p in payloads] == ["auto"] * 3
        assert payloads[1]["messages"] == payloads[2]["messages"][:-1]
        assert "上次未收到有效提交" in payloads[2]["messages"][-1]["content"]
        assert payloads[2]["messages"][-1] not in task["state"]["messages"]
    else:
        assert [p["tool_choice"] for p in payloads] == ["required"] * 2
    assert all(p["tools"] == tools for p in payloads)
    assert payloads[-1]["messages"][: len(payloads[0]["messages"])] == payloads[0]["messages"]
    assert payloads[-1]["messages"][-2 if remote else -1] == {
        "role": "user", "content": "请立刻根据已有信息提交"
    }
    assert task["submit_trigger"] == "round_limit"
    assert task["status"] == "submitted" and task["state"]["counts"]["graph_tool_calls"] == 1


def test_uncommitted_kill_rolls_back_and_operation_is_idempotent(tmp_path):
    db = tmp_path / "store.sqlite"
    Store(db)
    code = "import sqlite3,os; db=sqlite3.connect(__import__('sys').argv[1]); db.execute('BEGIN'); db.execute(\"INSERT INTO meta VALUES('uncommitted','1')\"); os._exit(7)"
    assert subprocess.run([sys.executable, "-c", code, str(db)]).returncode == 7
    store = Store(db)
    assert store.meta("uncommitted") is None
    task = {"status": "submitted", "prediction": "a"}
    assert store.commit_step("t", task, {"step_id": 1})
    assert not store.commit_step("t", dict(task, prediction="b"), {"step_id": 1})
    assert store.task("t")["prediction"] == "a"


def test_repeat_invalid_coarse_center_keeps_retry_feedback_out_of_trace(
    data, config, tmp_path, monkeypatch
):
    payloads = []
    store = Store(tmp_path / "audit.sqlite")
    key = "0:coarse_only:0"
    task = {
        "status": "running", "mode": "coarse_only", "target_id": 0, "repeat_id": 0,
        "state": main.make_sample_state(data, 0, "coarse_only", CLASSES, config),
    }
    tools = main.tool_schemas(CLASSES, config)

    def fake(store, c, prefix, key, step, generation, payload):
        payloads.append(payload)
        item = (
            call("search_1hop", search("F:0"), id="wrong_center") if step == 1
            else call("submit", {"prediction": "a", "reason": "done", "evidence_ids": ["F:0"]})
        )
        message = main.assistant_message({}, [item])
        return {"choices": [{"message": message}]}, f"{step}-{generation}"

    monkeypatch.setattr(main, "request_model", fake)
    main.handle_round(store, key, task, data, config, CLASSES, tools, lambda _: 1)
    task = store.task(key)
    assert task["active"]["generation"] == 1
    main.handle_round(store, key, task, data, config, CLASSES, tools, lambda _: 1)
    assert payloads[0]["messages"] == payloads[1]["messages"][:-1]
    assert "C:0" in payloads[1]["messages"][-1]["content"]
    state = task["state"]
    assert state["counts"]["graph_tool_calls"] == 0 and state["counts"]["tool_failed_rounds"] == 1
    assert payloads[1]["messages"][-1] not in state["messages"]
    event = json.loads(store.db.execute("SELECT data FROM events").fetchone()[0])
    assert event["event_status"] == "tool_failed" and event["new_evidence_count"] == 0
    main.handle_round(store, key, task, data, config, CLASSES, tools, lambda _: 1)
    assert task["status"] == "submitted" and task["submit_after_failed_tool_round"]


def test_evaluation_import_never_initializes_torch():
    result = subprocess.run(
        [sys.executable, "-c", "import evaluate,sys; assert 'torch' not in sys.modules"],
        cwd=Path(__file__).resolve().parents[1],
    )
    assert result.returncode == 0


@pytest.fixture
def remote_config(tmp_path, monkeypatch):
    import experiment_io

    public = tmp_path / "experiment.env"
    public.write_text("AGENT_MODEL_MODE=api\nGRAPH_MODEL_MODE=local\n")
    (tmp_path / "models.env").write_text(
        "AGENT_LLM_MODEL=test-local-qwen\nGRAPH_LLM_MODEL=test-local-qwen\n"
    )
    profile = tmp_path / "private" / "remote_api.env"
    profile.parent.mkdir()
    profile.write_text(
        "AGENT_API_BASE_URL=https://relay.fixture.invalid/v1\n"
        "AGENT_API_MODEL=fixture-private-deepseek\n"
        "AGENT_API_KEY=FAKE_TEST_SENTINEL_123456\n"
        "AGENT_API_CONTEXT_WINDOW_TOKENS=32768\n"
    )
    profile.chmod(0o600)
    monkeypatch.setattr(experiment_io, "REMOTE_API_PATH", profile)
    return public, profile


def test_evaluation_and_local_stage_never_open_remote_profile(remote_config, monkeypatch):
    import dotenv
    from experiment_io import load_env

    public, profile = remote_config
    parse = dotenv.dotenv_values

    def guarded(path, **kwargs):
        assert path != profile
        return parse(path, **kwargs)

    monkeypatch.setattr(dotenv, "dotenv_values", guarded)
    assert "_remote" not in load_env(public)
    assert "_remote" not in load_env(public, model_stage="GRAPH")


def test_private_fields_stay_out_of_requests_responses_and_snapshots(
    remote_config, tmp_path, monkeypatch
):
    import httpx
    from experiment_io import load_env, public_config, request_model, usage_for

    public, _ = remote_config
    config = load_env(public, model_stage="AGENT")
    model = config["AGENT_LLM_MODEL"]
    endpoint = config["_remote"]["base_url"]
    key = config["_remote"]["key"]
    requests = []

    def post(url, **kwargs):
        requests.append(url)
        assert url == endpoint + "/chat/completions"
        assert kwargs["headers"]["Authorization"] == "Bearer " + key
        assert kwargs["json"]["model"] == model
        assert kwargs["follow_redirects"] is False
        return httpx.Response(
            200,
            json={
                "model": model,
                "choices": [],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "prompt_cache_hit_tokens": 3,
                },
                "echo": f"{model} {endpoint} {key}",
            },
        )

    monkeypatch.setattr(httpx, "post", post)
    store = Store(tmp_path / "remote.sqlite")
    payload = {"model": model, "messages": []}
    first = request_model(store, config, "AGENT", "sample", 1, 0, payload)
    assert request_model(store, config, "AGENT", "sample", 1, 0, payload) == first
    assert len(requests) == 1
    saved = str(tuple(store.db.execute("SELECT payload,response FROM requests").fetchone()))
    assert all(
        secret not in saved + repr(public_config(config)) for secret in (model, endpoint, key)
    )
    assert usage_for(store.db, "sample")["cached_input_tokens"] == 3


def test_readable_configuration_rejects_api_key_field(tmp_path):
    from experiment_io import load_env

    (tmp_path / "experiment.env").write_text("AGENT_LLM_API_KEY=FAKE_TEST_SENTINEL_000000\n")
    with pytest.raises(ValueError, match="forbidden"):
        load_env(tmp_path / "experiment.env")


def test_config_interpreter_path_preserves_venv_symlink(tmp_path):
    from experiment_io import config_path

    executable = tmp_path / "env/bin/python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(sys.executable)
    assert config_path({"_base": str(tmp_path), "PYTHON": "env/bin/python"}, "PYTHON") == executable


def test_changing_secret_does_not_change_experiment_identity(remote_config):
    from experiment_io import load_env, stage_fingerprint

    public, profile = remote_config
    original = profile.read_text()
    a = load_env(public, model_stage="AGENT")
    profile.write_text(original.replace("FAKE_TEST_SENTINEL_123456", "FAKE_ROTATED_KEY"))
    b = load_env(public, model_stage="AGENT")
    assert stage_fingerprint(a, "agent") == stage_fingerprint(b, "agent")
    profile.write_text(original.replace("fixture-private-deepseek", "fixture-other-model"))
    c = load_env(public, model_stage="AGENT")
    assert stage_fingerprint(a, "agent") != stage_fingerprint(c, "agent") and stage_fingerprint(
        a, "build"
    ) == stage_fingerprint(c, "build")


def test_cached_and_unknown_cost_denominators_stay_separate():
    labels = {"classes": CLASSES, "targets": [{"fine_id": 0, "truth": "a"}]}
    rows = [
        {
            "repeat_id": 0,
            "mode": "multi",
            "target_id": 0,
            "status": "submitted",
            "prediction": "a",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 2,
                "total_tokens": 102,
                "complete": {
                    "input_tokens": True,
                    "output_tokens": True,
                    "total_tokens": True,
                    "cached_input_tokens": False,
                },
            },
        },
        {
            "repeat_id": 0,
            "mode": "fine_only",
            "target_id": 0,
            "status": "submitted",
            "prediction": "a",
            "usage": {
                "input_tokens": 200,
                "output_tokens": 3,
                "total_tokens": 203,
                "complete": {
                    "input_tokens": True,
                    "output_tokens": True,
                    "total_tokens": True,
                    "cached_input_tokens": False,
                },
            },
        },
    ]
    result = evaluate_rows(
        rows, labels, {"planned_targets": [0], "modes": ["multi", "fine_only"], "repeat_ids": [0]}
    )
    assert (
        result["paired_rows"][0]["input_tokens_paired_n"] == 1
        and result["paired_rows"][0]["cached_input_tokens_paired_n"] == 0
    )
    assert result["summary_rows"][0]["known_cached_input_tokens"] is None
    assert result["paired_rows"][0]["total_tokens_mean_difference"] == -101


def test_same_source_batch_progress_is_final_and_placeholders_are_grounded(data, config):
    s = main.make_sample_state(data, 0, "multi", CLASSES, config)
    calls = [
        call("search_1hop", search("C:0")),
        call("search_1hop", search("C:0"), id="call_2", index=1),
    ]
    final, results, _ = main.pack_tool_batch(data, s, {}, calls, config, lambda _: 1, True)
    assert final["seen_coarse"] == [0, 1, 2]
    assert all(
        not r["by_source"][0]["has_more"] and not r["by_source"][0]["has_more_new"] for r in results
    )


def test_readable_model_url_cannot_embed_credentials(tmp_path):
    from experiment_io import load_env

    (tmp_path / "experiment.env").write_text(
        "AGENT_LLM_BASE_URL=https://invalid.example/v1?key=FAKE_SENTINEL\n"
    )
    with pytest.raises(ValueError, match="forbidden"):
        load_env(tmp_path / "experiment.env")


def test_public_loopback_change_does_not_change_remote_dispatch(remote_config):
    from experiment_io import load_env, model_connection

    public, _ = remote_config
    public.write_text(public.read_text() + "AGENT_LLM_BASE_URL=http://127.0.0.1:9999/v1\n")
    config = load_env(public, model_stage="AGENT")
    assert model_connection(config, "AGENT")[0] == "https://relay.fixture.invalid/v1"


def test_public_model_config_cannot_point_to_private_file(tmp_path):
    from experiment_io import load_env

    path = tmp_path / "experiment.env"
    path.write_text("MODEL_CONFIG_FILE=private/remote_api.env\n")
    with pytest.raises(ValueError, match="public model configuration"):
        load_env(path)


@pytest.mark.parametrize("thinking,query_count", [(False, 1), (True, 2)])
def test_deepseek_relay_tool_round_trip(
    remote_config, tmp_path, monkeypatch, thinking, query_count
):
    import httpx
    from experiment_io import load_env

    public, profile = remote_config
    if thinking:
        profile.write_text(profile.read_text() + "AGENT_API_ENABLE_THINKING=true\n")
    config = load_env(public, model_stage="AGENT")
    tool_schemas = main.tool_schemas(CLASSES, config)
    counter = main.TokenCounter(config, tool_schemas)
    store = Store(tmp_path / "probe.sqlite")
    payloads = []

    def reject_models_lookup(*args, **kwargs):
        pytest.fail("API mode should probe completions rather than require /models")

    def post(url, **kwargs):
        payload = kwargs["json"]
        payloads.append(payload)
        assert url.endswith("/chat/completions")
        assert "parallel_tool_calls" not in payload and "seed" not in payload
        assert payload["thinking"]["type"] == ("enabled" if thinking else "disabled")
        assert ("temperature" in payload) != thinking
        assert payload["tool_choice"] == "auto"
        if len(payloads) == 1:
            calls = [
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": "search_1hop",
                        "arguments": json.dumps(search("C:0" if i == 0 else "F:1")),
                    },
                }
                for i in range(query_count)
            ]
            message = {"role": "assistant", "content": None, "tool_calls": calls}
            if thinking:
                message["reasoning_content"] = "synthetic reasoning marker"
        else:
            history = payload["messages"]
            assert [msg["role"] for msg in history] == ["system", "user", "assistant"] + [
                "tool"
            ] * query_count + ["user"]
            assert [msg["tool_call_id"] for msg in history[3:-1]] == [
                f"call_{i}" for i in range(query_count)
            ]
            assert all(isinstance(msg["content"], str) for msg in history[3:-1])
            if thinking:
                assert history[2]["reasoning_content"] == "synthetic reasoning marker"
            assert main.FORCE_SUBMIT == history[-1]["content"]
            assert payload["tools"] == payloads[0]["tools"]
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "answer",
                        "type": "function",
                        "function": {
                            "name": "submit",
                            "arguments": json.dumps(
                                {"prediction": "demo", "reason": "probe", "evidence_ids": []}
                            ),
                        },
                    }
                ],
            }
        return httpx.Response(200, json={"choices": [{"message": message}]})

    monkeypatch.setattr(httpx, "get", reject_models_lookup)
    monkeypatch.setattr(httpx, "post", post)
    main.verify_tool_capability(store, config, CLASSES, tool_schemas, counter)
    assert len(payloads) == 2
    assert store.meta("remote_tool_capabilities")["multiple_calls_observed"] == (query_count == 2)


def test_deepseek_flags_ignored_thinking_toggle(remote_config):
    from experiment_io import ConfigFailure, check_thinking_response, load_env

    public, _ = remote_config
    config = load_env(public, model_stage="AGENT")
    with pytest.raises(ConfigFailure, match="thinking is disabled"):
        check_thinking_response(
            config, "AGENT", {"choices": [{"message": {"reasoning_content": "example"}}]}
        )


def test_eval_rejects_duplicate_truth_even_if_prediction_matches_last_row():
    labels = {
        "classes": CLASSES,
        "targets": [{"fine_id": 0, "truth": "a"}, {"fine_id": 0, "truth": "b"}],
    }
    options = {"planned_targets": [0], "modes": ["multi"], "repeat_ids": [0]}
    with pytest.raises(ValueError, match="Duplicate"):
        evaluate_rows([], labels, options)


def test_selective_evaluation_uses_consistent_read_snapshot(tmp_path, monkeypatch):
    import evaluate as evaluation
    from experiment_io import digest, write_json

    labels = {"classes": CLASSES, "targets": [{"fine_id": t, "truth": "a"} for t in (0, 1, 2)]}
    store = Store(tmp_path / "experiment.sqlite")
    store.pin("config", "test-config-fingerprint")
    store.pin("evaluation_input_sha256", digest(labels))
    store.pin("modes", ["multi", "fine_only"])
    store.pin("repeat_ids", [0])
    write_json(
        tmp_path / "manifest.json",
        {"config_hash": "test-config-fingerprint", "evaluation_input_sha256": digest(labels)},
    )
    write_json(tmp_path / "evaluation_inputs.json", labels)
    for mode in ("multi", "fine_only"):
        store.save_task(
            f"0:{mode}:0",
            {
                "repeat_id": 0,
                "mode": mode,
                "target_id": 0,
                "status": "running" if mode == "fine_only" else "submitted",
                "prediction": None if mode == "fine_only" else "a",
                "state": {
                    "messages": [{"role": "system", "content": "not duplicated in eval files"}]
                },
            },
        )
    original = evaluation.result_for_task
    seen = []

    def trigger_writer(db, key, task, **kwargs):
        seen.append(key)
        if task["mode"] == "multi":
            updated = store.task("0:fine_only:0")
            updated.update(status="submitted", prediction="a")
            store.save_task("0:fine_only:0", updated)
        return original(db, key, task, **kwargs)

    monkeypatch.setattr(evaluation, "result_for_task", trigger_writer)
    result = evaluation.evaluate(tmp_path, {"limit": 1})
    assert seen == ["0:multi:0", "0:fine_only:0"]
    assert result["normalized_rows"][1]["status"] == "running"
    assert result["summary_rows"][1]["end_to_end_accuracy"] == 0
    assert result["summary_rows"][1]["input_tokens_complete_n"] == 0
    assert all("messages" not in row for row in result["normalized_rows"])
    assert (
        evaluation.evaluate(tmp_path, {"limit": 1})["summary_rows"][1]["end_to_end_accuracy"] == 1
    )


def test_remote_retry_preserves_unknown_and_reports_cache_token_coverage(
    remote_config, tmp_path, monkeypatch
):
    import httpx
    from experiment_io import ApiRateLimiter, load_env
    from evaluate import token_summary

    public, profile = remote_config
    profile.write_text(
        profile.read_text() + "\nAGENT_API_PRICE_INPUT_CNY_PER_MTOK=legacy-ignored\n"
    )
    c = load_env(public, model_stage="AGENT")
    c["_api_limiter"] = ApiRateLimiter(rpm=2500, tpm=5000000)
    responses = iter(
        [
            httpx.Response(429, headers={"Retry-After": "0"}),
            httpx.Response(
                200,
                json={
                    "choices": [{"finish_reason": "length"}],
                    "usage": {
                        "prompt_tokens": 1000,
                        "prompt_cache_hit_tokens": 800,
                        "prompt_cache_miss_tokens": 200,
                        "completion_tokens": 100,
                        "completion_tokens_details": {"reasoning_tokens": 0},
                    },
                },
            ),
        ]
    )
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: next(responses))
    store = Store(tmp_path / "usage.sqlite")
    request_model(store, c, "AGENT", "task", 1, 0, {"model": c["AGENT_LLM_MODEL"]})
    usage = usage_for(store.db, "task")
    stats = token_summary([usage])
    assert stats["known_total_tokens"] == 1100
    assert stats["known_cached_input_tokens"] == 800
    assert stats["known_uncached_input_tokens"] == 200
    assert stats["known_reasoning_tokens"] == 0
    assert stats["total_tokens_complete_n"] == 0
    assert stats["total_tokens_known_requests"] == 1 and stats["request_attempts"] == 2
    assert stats["usage_missing_requests"] == stats["finish_length_requests"] == 1
    assert stats["cache_coverage"] == 0.5
    assert stats["cache_pair_input_tokens"] == 1000
    assert stats["cache_token_ratio"] == 0.8 and stats["cache_hit_request_ratio"] == 1
    assert (
        store.db.execute("SELECT error FROM requests WHERE status=?", ("failed",)).fetchone()[0]
        == "HTTP 429"
    )


def test_api_scheduler_concurrent_samples_preserve_tool_order_and_resume(
    data, config, tmp_path, monkeypatch
):
    import re
    from threading import Barrier
    import httpx
    from experiment_io import api_runtime

    config.update(AGENT_MODEL_MODE="api", AGENT_MODEL_WORKERS="2", AGENT_MAX_MODEL_ROUNDS="2")
    config["_remote"] = {
        "stage": "AGENT",
        "base_url": "https://fake-concurrency.invalid/v1",
        "key": "fixture",
    }
    store = Store(tmp_path / "parallel.sqlite")
    plan = []
    for target in (0, 2):
        key = f"0:fine_only:{target}"
        plan.append(key)
        store.save_task(
            key,
            {
                "status": "pending",
                "repeat_id": 0,
                "target_id": target,
                "mode": "fine_only",
                "prediction": None,
                "active": None,
                "state": main.make_sample_state(data, target, "fine_only", CLASSES, config),
            },
        )
    rendezvous = Barrier(2, timeout=6)

    def mock_relay(request):
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer fixture"
        history = json.loads(request.content)["messages"]
        target = int(re.search(r'"ref":"F:(\d+)"', history[1]["content"]).group(1))
        if len(history) == 2:
            rendezvous.wait()
            name = "search_1hop"
            args = search(f"F:{target}")
            call_id = f"lookup_{target}"
        else:
            assert (
                history[-2]["role"] == "tool" and history[-2]["tool_call_id"] == f"lookup_{target}"
            )
            assert history[-3]["tool_calls"][0]["id"] == f"lookup_{target}"
            name = "submit"
            args = {"prediction": "a", "reason": "done", "evidence_ids": [f"F:{target}"]}
            call_id = f"submit_{target}"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": call_id,
                                    "type": "function",
                                    "function": {"name": name, "arguments": json.dumps(args)},
                                }
                            ]
                        }
                    }
                ]
            },
        )

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(mock_relay), **kwargs),
    )
    tools = main.tool_schemas(CLASSES, config)
    with api_runtime(config, "AGENT") as runtime:
        main.run_tasks(store, plan, data, runtime, CLASSES, tools, lambda _: 1)
    assert list(store.task_statuses().values()) == ["submitted", "submitted"]
    assert store.db.execute("SELECT COUNT(*) FROM events").fetchone()[0] == 4
    assert store.db.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 4
    with api_runtime(config, "AGENT") as runtime:
        main.run_tasks(store, plan, data, runtime, CLASSES, tools, lambda _: 1)
    assert store.db.execute("SELECT COUNT(*) FROM requests").fetchone()[0] == 4


def test_duplicate_runner_cannot_acquire_stage_lock(tmp_path):
    from experiment_io import StageBusy, process_lock

    with process_lock(tmp_path):
        with pytest.raises(StageBusy, match="Another process"):
            with process_lock(tmp_path):
                pytest.fail("duplicate writer")


def test_lost_response_is_exported_as_unknown_even_after_retry(
    remote_config, tmp_path, monkeypatch
):
    import httpx
    from experiment_io import RetryLater, load_env

    public, _ = remote_config
    c = load_env(public, model_stage="AGENT")
    c["AGENT_MAX_HTTP_ATTEMPTS"] = "1"

    def timeout(*args, **kwargs):
        raise httpx.ReadTimeout("response dropped")

    monkeypatch.setattr(httpx, "post", timeout)
    store = Store(tmp_path / "unknown.sqlite")
    with pytest.raises(RetryLater):
        request_model(store, c, "AGENT", "task", 1, 0, {"model": c["AGENT_LLM_MODEL"]})
    usage = usage_for(store.db, "task")
    assert usage["outcome_unknown"] == usage["usage_missing_requests"] == 1
    assert usage["total_tokens"] is None


def test_partial_usage_and_old_database_keep_independent_token_counts(tmp_path):
    from experiment_io import dumps, normalize_usage

    raw = {"prompt_tokens": 12, "completion_tokens": 2, "prompt_cache_hit_tokens": 13}
    u = normalize_usage(raw)
    assert u["total_tokens"] == 14 and u["cached_input_tokens"] is None and u["issues"]
    assert normalize_usage({"prompt_cache_miss_tokens": 4})["uncached_input_tokens"] == 4
    assert normalize_usage({"total_tokens": 6})["total_tokens"] == 6
    store = Store(tmp_path / "old.sqlite")
    with store.tx() as db:
        db.execute(
            "INSERT INTO requests(id,owner,step,generation,status,payload,usage) "
            "VALUES('old','task',1,0,'response_saved','{}',?)",
            (dumps({"input_tokens": 5, "output_tokens": 2, "cost_cny": 0.5}),),
        )
    old = usage_for(store.db, "task")
    assert old["total_tokens"] == 7 and old["complete"]["total_tokens"]
    assert "cost_cny" not in old
