from __future__ import annotations

import json
import os
import subprocess
import sys
import time

BENCH_CONFIG_PATH = "/tmp/jiuwenswarm_bench_config.json"
JIUWENSWARM_INSTALL_DIR = "/opt/jiuwenswarm"
TMP_WORKSPACE = "/tmp_workspace"


def wait_for_gateway(timeout: float = 30.0) -> bool:
    """Poll Gateway WebSocket port until ready."""
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", 19001), timeout=1):
                return True
        except OSError:
            time.sleep(0.5)
    return False


def main() -> int:
    data = json.loads(open(BENCH_CONFIG_PATH, encoding="utf-8").read())
    cfg = data["config"]
    prompt = data["prompt"]
    session_id = data.get("session_id", "bench-session")

    if not wait_for_gateway(timeout=60.0):
        print("ERROR: Gateway did not become ready", file=sys.stderr)
        return 1

    # Run jiuwenswarm chat
    cmd = [
        "jiuwenswarm", "chat",
        "--mode", "code.normal",
        "--session", session_id,
        "--cwd", TMP_WORKSPACE,
        "--project-dir", TMP_WORKSPACE,
        "--trusted-dir", TMP_WORKSPACE,
        "--gateway-url", "ws://127.0.0.1:19001/tui",
        "--jsonl",
        "--timeout", str(cfg.get("timeout", 600)),
        prompt,
    ]

    env = os.environ.copy()
    env["JIUWENSWARM_SKIP_DOTENV"] = "1"

    result = subprocess.run(cmd, env=env)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
