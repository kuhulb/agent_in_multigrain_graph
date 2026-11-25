"""Start/status/stop only this project's dedicated model process."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["start", "stop", "status"])
    args = p.parse_args()
    pidfile = ROOT / "run/model_server.pid"
    script = ROOT / "scripts/serve_model.py"
    if args.action == "stop":
        if pidfile.exists():
            pid = int(pidfile.read_text())
            cmd = Path(f"/proc/{pid}/cmdline")
            if cmd.exists() and str(script).encode() in cmd.read_bytes():
                os.kill(pid, signal.SIGTERM)
            pidfile.unlink(missing_ok=True)
        return
    try:
        with urllib.request.urlopen("http://127.0.0.1:18082/health", timeout=2) as r:
            print(r.read().decode())
            return
    except Exception:
        if args.action == "status":
            raise SystemExit("Model service is not ready")
    if pidfile.exists():
        pid = int(pidfile.read_text())
        cmd = Path(f"/proc/{pid}/cmdline")
        if cmd.exists() and str(script).encode() in cmd.read_bytes():
            print(
                json.dumps(
                    {
                        "pid": pid,
                        "status": "still_starting",
                        "log": str(ROOT / "logs/model_server.log"),
                    }
                )
            )
            return
    for name in ["run", "logs"]:
        (ROOT / name).mkdir(exist_ok=True)
    with open(ROOT / "logs/model_server.log", "ab") as log:
        proc = subprocess.Popen(
            [
                str(ROOT / ".venv/bin/python"),
                str(script),
                "--model",
                str(ROOT / "assets/models/Qwen3-4B-Instruct-2507"),
            ],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            start_new_session=True,
            env=dict(os.environ, HF_HOME=str(ROOT / ".cache/huggingface"), PYTHONUNBUFFERED="1"),
        )
    pidfile.write_text(str(proc.pid))
    print(json.dumps({"pid": proc.pid, "log": str(ROOT / "logs/model_server.log")}))


if __name__ == "__main__":
    main()
