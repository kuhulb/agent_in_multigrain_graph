"""Public model downloads only. All files/cache/partials stay inside this project."""

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request
import urllib.parse

ROOT = Path(__file__).resolve().parents[1]


def get(url):
    with urllib.request.urlopen(url, timeout=45) as r:
        return json.load(r)


def download(repo, name, destination):
    # Public ModelScope mirror is reachable from this host; HF file endpoints return 403.
    base = "https://modelscope.cn/api/v1/models/" + repo
    meta = get(base + "/repo/files?Revision=master&Recursive=true")
    records = [r for r in meta["Data"]["Files"] if "/" not in r["Path"] and r.get("Type") != "tree"]
    revision = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    destination.mkdir(parents=True, exist_ok=True)
    siblings = [s["Path"] for s in records]
    expected = {r["Path"]: r.get("Sha256") for r in records}
    have_safe = any(s.endswith(".safetensors") for s in siblings)
    selected = [
        s
        for s in siblings
        if s.endswith((".json", ".txt", ".model", ".safetensors", ".pt"))
        or s in {"merges.txt", "vocab.json"}
        or (s.endswith(".bin") and not have_safe)
    ]

    def one(filename):
        target = destination / filename
        mark = target.with_name(target.name + ".sha256")
        if target.exists() and mark.exists():
            return filename
        partial = target.with_name(target.name + ".partial")
        for attempt in range(8):
            offset = partial.stat().st_size if partial.exists() else 0
            url = base + "/repo?Revision=master&FilePath=" + urllib.parse.quote(filename)
            request = urllib.request.Request(
                url, headers={"Range": f"bytes={offset}-"} if offset else {}
            )
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    mode = "ab" if offset and response.status == 206 else "wb"
                    with open(partial, mode) as f:
                        while True:
                            chunk = response.read(1024 * 1024)
                            if not chunk:
                                break
                            f.write(chunk)
                        f.flush()
                        os.fsync(f.fileno())
                h = hashlib.sha256()
                with open(partial, "rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        h.update(chunk)
                if expected.get(filename) and h.hexdigest() != expected[filename]:
                    partial.unlink()
                    raise ValueError("Checksum mismatch")
                os.replace(partial, target)
                mark.write_text(h.hexdigest())
                print("MODEL_FILE", name, filename, target.stat().st_size, flush=True)
                return filename
            except Exception as e:
                print("DOWNLOAD_RETRY", name, filename, str(e)[:120], attempt + 1, flush=True)
                time.sleep(min(2**attempt, 30))
        raise RuntimeError("Download failed: " + filename)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(one, selected))
    (destination / "source_manifest.json").write_text(
        json.dumps(
            {
                "repo": repo,
                "source": "ModelScope",
                "revision": revision,
                "files": selected,
                "sha256": expected,
            },
            indent=2,
        )
    )
    print("MODEL_READY", name, revision, flush=True)


if __name__ == "__main__":
    download("Qwen/Qwen3-4B-Instruct-2507", "qwen3", ROOT / "assets/models/Qwen3-4B-Instruct-2507")
    download("BAAI/bge-m3", "bge-m3", ROOT / "assets/models/bge-m3")
