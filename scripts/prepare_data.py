"""Export existing user-owned TAG data into explicit, reviewable ID-aligned files."""

import argparse
import csv
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment_io import file_hash, write_json, write_jsonl, atomic_bytes


def main():
    import torch
    import io

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", default="cora")
    args = parser.parse_args()
    source = Path(args.source)
    out = Path(args.output)
    data = torch.load(source / "processed_data.pt", map_location="cpu", weights_only=False)
    n = int(data.num_nodes)
    if len(data.title) != n or len(data.abs) != n or len(data.y) != n:
        raise ValueError("Text/node/label rows not aligned")
    with open(source / "categories.csv") as f:
        categories = [r["name"] for r in csv.DictReader(f)]
    if len(categories) != int(data.num_classes):
        raise ValueError("Class count mismatch")
    labels = list(getattr(data, "label_texts", []))
    if len(labels) == len(categories) and [str(x).replace("_", " ").lower() for x in labels] != [
        x.lower() for x in categories
    ]:
        raise ValueError("Source label_texts disagree with categories.csv")
    ids = [f"{args.dataset}-tag:{i}" for i in range(n)]
    papers = [
        {"paper_id": ids[i], "title": str(data.title[i] or ""), "abstract": str(data.abs[i] or "")}
        for i in range(n)
    ]
    mask = {split: getattr(data, split + "_mask").tolist() for split in ("train", "val", "test")}
    rows = []
    for i in range(n):
        selected = [s for s in mask if mask[s][i]]
        if len(selected) != 1:
            raise ValueError(f"Expected exactly one split for node {i}")
        label = int(data.y[i])
        if not 0 <= label < len(categories):
            raise ValueError("Unknown label")
        rows.append({"paper_id": ids[i], "class_id": str(label), "split": selected[0]})
    edges = data.edge_index.t().tolist()
    if any(not 0 <= a < n or not 0 <= b < n for a, b in edges):
        raise ValueError("Unknown edge endpoint")
    write_jsonl(out / "papers.jsonl", papers)
    for name, fieldnames, records in [
        (
            "citations.tsv",
            ["source_id", "target_id"],
            [{"source_id": ids[a], "target_id": ids[b]} for a, b in edges],
        ),
        ("targets.tsv", ["paper_id", "class_id", "split"], rows),
    ]:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(records)
        atomic_bytes(out / name, buffer.getvalue().encode())
    write_json(
        out / "classes.json",
        [{"class_id": str(i), "name": name} for i, name in enumerate(categories)],
    )
    write_json(
        out / "source_manifest.json",
        {
            "dataset": args.dataset,
            "source": str(source.resolve()),
            "processed_sha256": file_hash(source / "processed_data.pt"),
            "categories_sha256": file_hash(source / "categories.csv"),
            "num_nodes": n,
            "raw_edge_rows": len(edges),
            "id_rule": "dataset-tag:original_processed_row",
            "split_source": "existing masks, no resampling",
            "source_label_texts_sample": labels[:8],
        },
    )
    print("DATA_READY", out, "nodes=", n, "edge_rows=", len(edges))


if __name__ == "__main__":
    main()
