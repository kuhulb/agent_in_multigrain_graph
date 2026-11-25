"""Python 3.10 boundary for the supplied original bytecode. Never imported by inference."""

import argparse
import hashlib
import importlib.machinery
import json
from pathlib import Path
import random
import os
from types import SimpleNamespace


def main():
    import networkx as nx
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bytecode", required=True)
    args = parser.parse_args()
    raw = json.loads(Path(args.input).read_text())
    random.seed(raw["seed"])
    np.random.seed(raw["seed"])
    graph = nx.Graph()
    # Isolates are handled explicitly; the original optimizer divides by graph weight.
    for a, b in raw["fine_edges"]:
        graph.add_edge(a, b, weight=1.0)
    result = []
    if graph.number_of_edges():
        module = importlib.machinery.SourcelessFileLoader(
            "original_lv_V5", args.bytecode
        ).load_module()
        wrapped = SimpleNamespace(G=graph)
        module.louvainModularityOptimization(
            wrapped,
            priorCommunities=False,
            random=False,
            weight="weight",
            qc_norm="zscore",
            quantiles=(0.33, 0.67),
            save_path=args.output + ".tsv",
            attach_to_graph=True,
        )
        for old_id, item in wrapped.comm_info.items():
            result.append(
                {
                    "old_id": int(old_id),
                    "members": sorted(int(x) for x in item["nodes"]),
                    "original_stats": {
                        k: v.item() if hasattr(v, "item") else v
                        for k, v in item.items()
                        if k != "nodes"
                    },
                }
            )
    accounted = {v for item in result for v in item["members"]}
    for node in range(raw["num_nodes"]):
        if node not in accounted:
            if node in graph:
                raise ValueError("Optimizer omitted a non-isolated node")
            result.append(
                {"old_id": None, "members": [node], "original_stats": {"level": "Peripheral"}}
            )
    result.sort(key=lambda x: min(x["members"]))
    output = {
        "communities": result,
        "source_sha256": hashlib.sha256(Path(args.bytecode).read_bytes()).hexdigest(),
        "weight_convention": "one logical undirected edge has weight 1",
    }
    destination = Path(args.output)
    temporary = destination.with_suffix(".json.tmp")
    with open(temporary, "w") as stream:
        json.dump(output, stream, ensure_ascii=False, default=float)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)


if __name__ == "__main__":
    main()
