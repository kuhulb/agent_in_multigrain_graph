"""Freeze a proportional sample of IDs from an existing dataset split."""

import argparse
from collections import defaultdict
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_io import config_path, file_hash, load_env, read_json, write_json
from main import TARGET_SPLITS, load_evaluation_labels


def sample_targets(rows, split, size, seed):
    groups = defaultdict(list)
    for row in rows:
        if row["split"] == split:
            groups[row["truth"]].append(row["paper_id"])
    population = sum(map(len, groups.values()))
    if not 1 <= size <= population:
        raise ValueError(f"Sample size must be between 1 and {population}")
    labels = sorted(groups)
    minimum = 1 if size >= len(labels) else 0
    # Allocate whole numbers near proportional shares, including every available class when feasible.
    quotas = {label: max(minimum, size * len(groups[label]) // population) for label in labels}
    while sum(quotas.values()) != size:
        add = sum(quotas.values()) < size
        available = [
            label
            for label in labels
            if (quotas[label] < len(groups[label]) if add else quotas[label] > minimum)
        ]
        direction = 1 if add else -1
        chosen = max(
            available,
            key=lambda label: direction * (size * len(groups[label]) - population * quotas[label]),
        )
        quotas[chosen] += direction

    rng = random.Random(seed)
    selected = []
    for label in labels:
        selected.extend(rng.sample(sorted(groups[label]), quotas[label]))
    rng.shuffle(selected)
    return selected, quotas, population


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="experiment.env")
    parser.add_argument("--split", choices=TARGET_SPLITS, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--seed", type=int, default=234)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_env(args.env)
    graph_dir = config_path(config, "AGENT_GRAPH_BUILD_DIR")
    manifest = read_json(graph_dir / "build_manifest.json")
    if manifest.get("status") != "ready" or manifest.get("dataset") != config["DATASET"]:
        raise ValueError("Choose a ready graph with the same dataset in experiment.env")
    raw_name = "artifacts/raw_graph.json"
    raw_file = graph_dir / raw_name
    entry = next((entry for entry in manifest["artifacts"] if entry["path"] == raw_name), None)
    if entry is None or entry["sha256"] != file_hash(raw_file):
        raise ValueError("Ready graph raw ID artifact is invalid")
    raw = read_json(raw_file)
    if raw["graph_sha256"] != manifest["source_graph_sha256"]:
        raise ValueError("Ready graph source ID mismatch")
    _, rows, source = load_evaluation_labels(config, raw)
    paper_ids, counts, population = sample_targets(rows, args.split, args.size, args.seed)
    plan = {
        "schema_version": 1,
        "dataset": config["DATASET"],
        "source_graph_sha256": raw["graph_sha256"],
        "label_source": source,
        "split": args.split,
        "size": len(paper_ids),
        "seed": args.seed,
        "sampling": "proportional_stratified_v1",
        "paper_ids": paper_ids,
    }
    output = Path(args.output)
    if output.exists():
        if read_json(output) != plan:
            raise ValueError("Refusing to change an existing target plan; choose another filename")
    else:
        write_json(output, plan)
    print(f"TARGET_PLAN_READY {output}: {len(paper_ids)}/{population}, per-class counts={counts}")


if __name__ == "__main__":
    main()
