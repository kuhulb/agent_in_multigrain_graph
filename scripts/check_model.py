"""Real native-function-call, count-calibration and non-thinking deployment probe."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_io import Store, config_path, load_env, read_json, write_json
from main import TokenCounter, tool_schemas, verify_tool_capability


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="experiment.env")
    args = p.parse_args()
    c = load_env(args.env, model_stage="AGENT")
    folder = Path(c["_base"])
    classes = read_json(
        config_path(c, "AGENT_CLASSES_FILE", str(config_path(c, "DATA_DIR") / "classes.json"))
    )
    tools = tool_schemas(classes, c)
    counter = TokenCounter(c, tools)
    report = folder / "reports/model_capability"
    if c.get("_remote"):
        report = folder / "reports/api_probe" / c["_remote_identity"][:16]
        store = Store(report / "capability.sqlite")
    else:
        store = Store(folder / "reports/model_capability.sqlite")
    verify_tool_capability(store, c, classes, tools, counter)
    store.export(report)
    write_json(
        report / "result.json",
        {
            "verified": True,
            "deployment": store.meta("AGENT_deployment"),
            "tokenizer_template_hash": counter.identity,
            "remote_tool_capabilities": store.meta("remote_tool_capabilities"),
        },
    )
    print("MODEL_CAPABILITY_VERIFIED", flush=True)


if __name__ == "__main__":
    main()
