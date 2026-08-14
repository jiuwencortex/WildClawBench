from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

from src.agents.base import AgentExecution, AgentTaskSpec, BaseAgent
from src.utils.docker_utils import (
    inject_lobster_workspace,
    run_background,
    run_warmup,
    setup_skills,
    setup_workspace,
    start_container,
)

load_dotenv()

logger = logging.getLogger(__name__)

# Thin base image: same wildclawbench-ubuntu base the other backends use, WITHOUT
# jiuwenswarm baked in. The user-uploaded jiuwenswarm whl is pip-installed into the
# container at run start (version-check friendly). start_container() reads DOCKER_IMAGE
# from the env, so we set it here before starting the container.
DOCKER_IMAGE = os.environ.get(
    "DOCKER_IMAGE_JIUWENSWARM", "wildclawbench-jiuwenswarm-ubuntu:v0.1"
)
# WebSocket port the in-container jiuwenswarm-agentserver listens on (container loopback).
GATEWAY_PORT = int(os.environ.get("JIUWENSWARM_GATEWAY_PORT", "18092"))
# in-container llm_forward port (per-task token attribution source of truth).
LLM_FORWARD_PORT = int(os.environ.get("LLM_FORWARD_PORT", "8000"))

# Where the agent session transcript lands inside the container (filled per task).
TRANSCRIPT_CONTAINER_DIR = "/root/.jiuwenswarm/agent/sessions"
# Path the embedded graders read (OpenClaw-style transcript we synthesize).
COMPAT_TRANSCRIPT_CONTAINER_PATH = "/root/.openclaw/agents/main/sessions/chat.jsonl"

# Where the in-container drive script and the vendored client land.
DRIVE_SCRIPT_CONTAINER_PATH = "/tmp/_drive_agent.py"
VENDORED_CLIENT_CONTAINER_PATH = "/tmp/_jiuwenswarm_client.py"


class JiuwenSwarmAgent(BaseAgent):
    """JiuwenSwarm backend for WildClawBench.

    Mirrors the openclaw backend: agentserver + agent BOTH run inside the task's own
    Docker container (no host port publishing → safe under --parallel). At run start it
    (a) pip-installs the user-uploaded jiuwenswarm whl (required, per-task version),
    (b) replaces the whl-bundled config (uploaded or generated) and points the model at
        an in-container llm_forward for per-task token attribution,
    (c) starts the agentserver, then drives one task to completion via a small in-container
        script that reuses the installed jiuwenswarm package's websocket client, and
    (d) collects usage from llm_forward's /token-stats (NOT the transcript).
    """

    def __init__(
        self,
        openrouter_api_key: str = "",
        openrouter_base_url: str = "",
        *,
        jiuwenswarm_whl_host_path: str = "",
        jiuwenswarm_config_host_path: str = "",
        llm_forward_host_script: str = "",
        agentserver_client_host_path: str = "",
    ) -> None:
        self.openrouter_api_key = openrouter_api_key
        self.openrouter_base_url = openrouter_base_url
        self.jiuwenswarm_whl_host_path = jiuwenswarm_whl_host_path
        self.jiuwenswarm_config_host_path = jiuwenswarm_config_host_path
        # llm_forward.py lives in BenchRunner (benchrunner-backend/llm_forward/). The bench
        # orchestrator may pass an explicit path; otherwise fall back to env.
        self.llm_forward_host_script = (
            llm_forward_host_script
            or os.environ.get("LLM_FORWARD_SCRIPT_PATH", "")
        )
        # The high-level AgentServerClient lives in BenchRunner (benchrunner-backend/
        # utils/jiuwenswarm_client.py). It only imports from the jiuwenswarm package, so it
        # imports cleanly once jiuwenswarm is installed in-container. Vendoring it into the
        # container gives an exact wire-protocol match with no reimplementation.
        self.agentserver_client_host_path = (
            agentserver_client_host_path
            or os.environ.get("JIUWENSWARM_CLIENT_SCRIPT_PATH", "")
        )
        # Per-task session ids, keyed by task_id (thread-safe under --parallel: each
        # task has its own container + session).
        self._sessions: dict[str, str] = {}

    @property
    def expects_gateway(self) -> bool:
        return True

    @property
    def transcript_container_path(self) -> str:
        return COMPAT_TRANSCRIPT_CONTAINER_PATH

    # ------------------------------------------------------------------ #
    # run_task
    # ------------------------------------------------------------------ #
    def run_task(self, spec: AgentTaskSpec) -> AgentExecution:
        gateway_proc = None
        agent_proc = None
        elapsed_time = float(spec.timeout_seconds)

        try:
            exec_path = os.path.join(spec.workspace_path, "exec")
            tmp_path = os.path.join(spec.workspace_path, "tmp")
            os.makedirs(exec_path, exist_ok=True)

            # The shared start_container() reads DOCKER_IMAGE from env; set our thin image.
            os.environ["DOCKER_IMAGE"] = DOCKER_IMAGE

            start_container(
                spec.task_id,
                exec_path,
                extra_env=spec.task.get("env", ""),
                tmp_path=tmp_path,
                lobster_env=spec.lobster.get("env") if spec.lobster else None,
            )
            if spec.lobster:
                inject_lobster_workspace(spec.task_id, spec.lobster["workspace"])

            setup_workspace(spec.task_id, thinking=spec.thinking)
            setup_skills(
                spec.task_id,
                spec.task.get("skills", ""),
                spec.task.get("skills_path", ""),
                container_skills_root="/root/.jiuwenswarm/skills",
            )
            run_warmup(spec.task_id, spec.task.get("warmup", ""))

            # (a) Install the user-uploaded jiuwenswarm whl (required, per-task version).
            self._install_jiuwenswarm_whl(spec.task_id)

            # (b) Configure the model: replace the whl-bundled config (uploaded or
            #     generated) and point the model at the in-container llm_forward.
            self._configure_model(spec.task_id, spec.model)

            # (c) Start the in-container llm_forward proxy (token attribution source).
            self._start_llm_forward(spec.task_id)

            # (d) Start the jiuwenswarm-agentserver (gateway) and wait for readiness.
            gateway_proc = self._start_agentserver(spec.task_id)

            # (e) Drive the agent one-shot to completion / timeout.
            session_id = f"wcb_{spec.task_id}_{int(time.time() * 1000)}"
            self._sessions[spec.task_id] = session_id
            # Drive mode comes from the bench orchestrator via env (task-level jiuwenswarm
            # mode, e.g. agent.plan); not set → agentserver default.
            mode = os.environ.get("JIUWENSWARM_DRIVE_MODE") or None
            start_time = time.perf_counter()
            agent_proc = self._drive_agent(
                spec.task_id,
                prompt=spec.prompt,
                session_id=self._session_id,
                mode=mode,
                timeout_seconds=spec.timeout_seconds,
                log_path=spec.output_dir / "agent.log",
            )
            try:
                agent_proc.wait(timeout=spec.timeout_seconds + 30)
                elapsed_time = time.perf_counter() - start_time
                logger.info(
                    "[%s] jiuwenswarm drive finished, elapsed: %.2fs",
                    spec.task_id, elapsed_time,
                )
            except subprocess.TimeoutExpired:
                logger.warning("[%s] jiuwenswarm drive timed out", spec.task_id)
                elapsed_time = float(spec.timeout_seconds)
                self._interrupt_session(spec.task_id, session_id)
                agent_proc.kill()
                try:
                    agent_proc.wait(timeout=15)
                except Exception:
                    pass

            return AgentExecution(
                elapsed_time=elapsed_time,
                error=None,
                gateway_proc=gateway_proc,
                agent_proc=agent_proc,
            )
        except Exception as exc:
            logger.error("[%s] jiuwenswarm execution error: %s", spec.task_id, exc)
            return AgentExecution(
                elapsed_time=float(spec.timeout_seconds),
                error=str(exc),
                gateway_proc=gateway_proc,
                agent_proc=agent_proc,
            )

    # ------------------------------------------------------------------ #
    # usage + grading transcript
    # ------------------------------------------------------------------ #
    def collect_usage(self, task_id: str, output_dir: Path, elapsed_time: float) -> dict:
        """Token usage from the in-container llm_forward /token-stats (per-task truth)."""
        usage = self._read_llm_forward_stats(task_id)
        usage["elapsed_time"] = round(elapsed_time, 2)
        return usage

    def prepare_grading_transcript(self, task_id: str) -> str:
        """Convert jiuwenswarm session history.jsonl → OpenClaw-style transcript.

        Embedded graders load the transcript from COMPAT_TRANSCRIPT_CONTAINER_PATH
        (see src/utils/transcript_loader.py). We read the jiuwenswarm session history,
        map user/assistant/tool turns to the {type:message, message:{role,content,usage}}
        shape graders expect, and write it to that path in the container.
        """
        try:
            history_path = self._session_history_container_path(task_id)
            if not history_path:
                logger.warning(
                    "[%s] no session id; cannot locate jiuwenswarm history", task_id
                )
                return self.transcript_container_path
            self._write_compat_transcript(task_id, history_path)
        except Exception as exc:
            logger.warning(
                "[%s] compat transcript build failed, grader will use fallback: %s",
                task_id, exc,
            )
        return self.transcript_container_path

    # ------------------------------------------------------------------ #
    # in-container setup helpers
    # ------------------------------------------------------------------ #
    def _install_jiuwenswarm_whl(self, task_id: str) -> None:
        if not self.jiuwenswarm_whl_host_path or not os.path.isfile(
            self.jiuwenswarm_whl_host_path
        ):
            raise RuntimeError(
                "jiuwenswarm whl upload is required (no default); "
                "supply --jiuwenswarm-whl <path> pointing at the uploaded whl"
            )
        r_cp = subprocess.run(
            ["docker", "cp", self.jiuwenswarm_whl_host_path, f"{task_id}:/tmp/jw.whl"],
            capture_output=True, text=True,
        )
        if r_cp.returncode != 0:
            raise RuntimeError(f"docker cp whl failed: {r_cp.stderr.strip()}")
        r_inst = subprocess.run(
            ["docker", "exec", task_id, "/bin/bash", "-lc",
             "pip install --quiet /tmp/jw.whl && pip show jiuwenswarm | head -3"],
            capture_output=True, text=True,
        )
        if r_inst.returncode != 0:
            raise RuntimeError(
                f"in-container pip install jiuwenswarm failed:\n{r_inst.stderr}"
            )
        logger.info("[%s] Installed jiuwenswarm whl:\n%s", task_id, r_inst.stdout.strip())

    def _configure_model(self, task_id: str, model: str) -> None:
        """Replace /root/.jiuwenswarm/config/config.yaml: model + llm_forward api_base."""
        config_path = "/root/.jiuwenswarm/config/config.yaml"

        if self.jiuwenswarm_config_host_path and os.path.isfile(
            self.jiuwenswarm_config_host_path
        ):
            # User uploaded a config → it replaces the whl-bundled default verbatim.
            r_cp = subprocess.run(
                ["docker", "cp", self.jiuwenswarm_config_host_path, f"{task_id}:{config_path}"],
                capture_output=True, text=True,
            )
            if r_cp.returncode != 0:
                raise RuntimeError(f"docker cp config.yaml failed: {r_cp.stderr.strip()}")
            logger.info("[%s] Replaced config.yaml with uploaded one", task_id)
        else:
            self._ensure_config_exists(task_id, config_path)

        # Point the default model at the in-container llm_forward regardless of source,
        # so token traffic flows through the proxy. Uses the same awk-scope idiom as
        # BenchRunner's _jiuwenswarm_deploy._configure_subject.
        api_base = f"http://127.0.0.1:{LLM_FORWARD_PORT}"
        api_key = self.openrouter_api_key or "forward-proxy"
        replacements = [
            ("api_base:", api_base),
            ("api_key:", api_key),
            ("model_name:", model),
            ("client_provider:", "OpenAI"),
        ]
        for key, val in replacements:
            awk_prog = (
                f"awk '/defaults:/{{f=1}} f && /is_default/{{print; f=0; next}} "
                f"f && /{key}/{{sub(/: .*/, \": {val}\")}} "
                f"1' {config_path} > {config_path}.tmp && mv {config_path}.tmp {config_path}"
            )
            r = subprocess.run(
                ["docker", "exec", task_id, "/bin/bash", "-lc", awk_prog],
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                logger.warning("[%s] config rewrite %s failed: %s",
                               task_id, key, r.stderr.strip())
        logger.info("[%s] config.yaml: model=%s, api_base=%s", task_id, model, api_base)

    def _ensure_config_exists(self, task_id: str, config_path: str) -> None:
        # Only generate if the whl did not ship a config.
        r = subprocess.run(
            ["docker", "exec", task_id, "test", "-f", config_path],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            return
        minimal = (
            "models:\n"
            "  defaults:\n"
            "    - is_default: true\n"
            f"      api_base: http://127.0.0.1:{LLM_FORWARD_PORT}\n"
            "      api_key: forward-proxy\n"
            "      model_name: placeholder\n"
            "      client_provider: OpenAI\n"
        )
        write_cmd = (
            f"mkdir -p $(dirname {config_path}) && "
            f"cat > {config_path} <<'YAML'\n{minimal}YAML"
        )
        subprocess.run(
            ["docker", "exec", task_id, "/bin/bash", "-lc", write_cmd],
            capture_output=True, text=True,
        )

    def _start_llm_forward(self, task_id: str) -> None:
        """Copy llm_forward.py into the container and start it pointed at the real upstream."""
        script = self.llm_forward_host_script
        if not script or not os.path.isfile(script):
            logger.warning(
                "[%s] llm_forward.py path not provided (LLM_FORWARD_SCRIPT_PATH); "
                "skipping in-container proxy — token columns will be 0", task_id
            )
            return
        container_script = "/tmp/llm_forward/llm_forward.py"
        subprocess.run(
            ["docker", "exec", task_id, "mkdir", "-p", "/tmp/llm_forward"],
            capture_output=True, text=True,
        )
        r_cp = subprocess.run(
            ["docker", "cp", script, f"{task_id}:{container_script}"],
            capture_output=True, text=True,
        )
        if r_cp.returncode != 0:
            logger.warning("[%s] docker cp llm_forward.py failed: %s",
                           task_id, r_cp.stderr.strip())
            return
        # Best-effort ensure deps in the container's python.
        subprocess.run(
            ["docker", "exec", task_id, "/bin/bash", "-lc",
             "pip install --quiet fastapi uvicorn httpx python-dotenv 2>/dev/null || true"],
            capture_output=True, text=True,
        )
        # The real upstream creds come from the orchestrator's BASE_URL/API_KEY (the actual
        # model endpoint), not the openrouter* values.
        upstream_base = self.openrouter_base_url
        upstream_key = self.openrouter_api_key
        start_cmd = (
            "cd /tmp/llm_forward && "
            f"LLM_FORWARD_API_BASE={sh_quote(upstream_base)} "
            f"LLM_FORWARD_API_KEY={sh_quote(upstream_key)} "
            "nohup python3 -m uvicorn llm_forward:app --host 127.0.0.1 "
            f"--port {LLM_FORWARD_PORT} > /tmp/llm_forward.log 2>&1 < /dev/null &"
        )
        subprocess.run(
            ["docker", "exec", task_id, "/bin/bash", "-lc", start_cmd],
            capture_output=True, text=True,
        )
        # Wait for health (best-effort).
        for _ in range(40):
            r = subprocess.run(
                ["docker", "exec", task_id, "/bin/bash", "-lc",
                 f"curl -fsS -m 2 http://127.0.0.1:{LLM_FORWARD_PORT}/health >/dev/null 2>&1"],
                capture_output=True, text=True,
            )
            if r.returncode == 0:
                # Reset any token stats left from a previous run in this container.
                subprocess.run(
                    ["docker", "exec", task_id, "/bin/bash", "-lc",
                     f"curl -fsS -m 2 -X DELETE http://127.0.0.1:{LLM_FORWARD_PORT}/token-stats >/dev/null 2>&1 || true"],
                    capture_output=True, text=True,
                )
                logger.info("[%s] in-container llm_forward ready on :%d",
                            task_id, LLM_FORWARD_PORT)
                return
            time.sleep(0.5)
        logger.warning("[%s] llm_forward health-check failed; token stats may be 0", task_id)

    def _start_agentserver(self, task_id: str):
        """Start jiuwenswarm-agentserver in the container; return the background proc."""
        # run_background cds into TMP_WORKSPACE; the agentserver writes its workspace there.
        proc = run_background(
            task_id,
            bash_cmd=(
                f"python3 -m jiuwenswarm.server.app_agentserver "
                f"--host 127.0.0.1 --port {GATEWAY_PORT}"
            ),
            log_path=Path("/tmp") / "agentserver.log",
        )
        # Readiness: poll until the in-container websocket is accepting connections.
        ready = False
        for _ in range(60):
            if proc.poll() is not None:
                raise RuntimeError("jiuwenswarm-agentserver exited during startup")
            r = subprocess.run(
                ["docker", "exec", task_id, "/bin/bash", "-lc",
                 f"python3 -c 'import socket; "
                 f"s=socket.socket(); s.settimeout(1); "
                 f"s.connect((\"127.0.0.1\",{GATEWAY_PORT})); s.close()' 2>/dev/null"],
                capture_output=True, text=True,
            )
            if r.returncode == 0:
                ready = True
                break
            time.sleep(1.0)
        if not ready:
            raise RuntimeError(
                f"jiuwenswarm-agentserver not ready on :{GATEWAY_PORT} after 60s"
            )
        logger.info("[%s] jiuwenswarm-agentserver ready on :%d", task_id, GATEWAY_PORT)
        return proc

    def _drive_agent(
        self,
        task_id: str,
        *,
        prompt: str,
        session_id: str,
        mode: str | None,
        timeout_seconds: int,
        log_path: Path,
    ):
        """Run an in-container python script that drives the agentserver one-shot.

        Vendors the high-level AgentServerClient (from the bench orchestrator) into the
        container — it imports only from the installed jiuwenswarm package, so it works
        in-container and gives an exact wire-protocol match. The drive creates a session,
        sends the prompt (stream + timeout), and writes the final answer length to stdout.
        """
        if not self._vendor_client(task_id):
            raise RuntimeError(
                "AgentServerClient source not configured "
                "(JIUWENSWARM_CLIENT_SCRIPT_PATH) — cannot drive agent"
            )

        mode_arg = json.dumps(mode)
        drive_script = _DRIVE_SCRIPT.format(
            uri=f"ws://127.0.0.1:{GATEWAY_PORT}",
            session_id=session_id,
            mode=mode_arg,
            timeout=timeout_seconds,
        )
        with tempfile.NamedTemporaryFile(
            "w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(drive_script)
            host_script = f.name
        try:
            subprocess.run(
                ["docker", "cp", host_script, f"{task_id}:{DRIVE_SCRIPT_CONTAINER_PATH}"],
                capture_output=True, text=True,
            )
            # Pass the prompt via stdin so it survives any shell-quoting, then close
            # stdin so the drive script's sys.stdin.read() returns.
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(
                ["docker", "exec", "-i", task_id, "python3", DRIVE_SCRIPT_CONTAINER_PATH],
                stdin=subprocess.PIPE,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
            )
            proc._log_file = log_file
            try:
                proc.stdin.write(prompt)
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            logger.info("[%s] Started jiuwenswarm drive PID=%s → %s",
                        task_id, proc.pid, log_path)
            return proc
        finally:
            Path(host_script).unlink(missing_ok=True)

    def _vendor_client(self, task_id: str) -> bool:
        """Copy the high-level AgentServerClient into the container (in-container import)."""
        src = self.agentserver_client_host_path
        if not src or not os.path.isfile(src):
            return False
        r = subprocess.run(
            ["docker", "cp", src, f"{task_id}:{VENDORED_CLIENT_CONTAINER_PATH}"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            logger.warning("[%s] docker cp agentserver client failed: %s",
                           task_id, r.stderr.strip())
            return False
        return True

    def _interrupt_session(self, task_id: str, session_id: str) -> None:
        """Best-effort: ask the agentserver to cancel the in-flight request on timeout."""
        interrupt_script = _INTERRUPT_SCRIPT.format(
            uri=f"ws://127.0.0.1:{GATEWAY_PORT}",
            session_id=session_id,
        )
        with tempfile.NamedTemporaryFile(
            "w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(interrupt_script)
            host_script = f.name
        try:
            subprocess.run(
                ["docker", "cp", host_script, f"{task_id}:/tmp/_interrupt.py"],
                capture_output=True, text=True,
            )
            subprocess.run(
                ["docker", "exec", task_id, "python3", "/tmp/_interrupt.py"],
                capture_output=True, text=True, timeout=30,
            )
        except Exception:
            pass  # container may already be tearing down
        finally:
            Path(host_script).unlink(missing_ok=True)

    # ------------------------------------------------------------------ #
    # token stats + transcript readback
    # ------------------------------------------------------------------ #
    def _read_llm_forward_stats(self, task_id: str) -> dict:
        # Read /token-stats from the in-container proxy, map to the standard usage dict.
        r = subprocess.run(
            ["docker", "exec", task_id, "/bin/bash", "-lc",
             f"curl -fsS -m 5 http://127.0.0.1:{LLM_FORWARD_PORT}/token-stats 2>/dev/null"],
            capture_output=True, text=True,
        )
        stats: dict = {}
        if r.returncode == 0 and r.stdout.strip():
            try:
                stats = json.loads(r.stdout.strip())
            except json.JSONDecodeError:
                stats = {}
        if not stats:
            logger.warning("[%s] could not read in-container llm_forward /token-stats", task_id)
        return {
            "input_tokens": int(stats.get("prompt_tokens", 0) or 0),
            "output_tokens": int(stats.get("completion_tokens", 0) or 0),
            "cache_read_tokens": int(stats.get("prompt_cache_hit_tokens", 0) or 0),
            "cache_write_tokens": 0,
            "total_tokens": int(stats.get("total_tokens", 0) or 0),
            "cost_usd": 0.0,
            "request_count": int(stats.get("total_requests", 0) or 0),
        }

    def _session_history_container_path(self, task_id: str) -> str:
        session_id = self._sessions.get(task_id, "")
        if not session_id:
            return ""
        return f"{TRANSCRIPT_CONTAINER_DIR}/{session_id}/history.jsonl"

    def _write_compat_transcript(self, task_id: str, history_path: str) -> None:
        # Copy the jiuwenswarm session history out, convert to OpenClaw-style messages,
        # write to COMPAT_TRANSCRIPT_CONTAINER_PATH in-container for the grader.
        converter = _COMPAT_SCRIPT.format(
            history_path=history_path,
            out_path=COMPAT_TRANSCRIPT_CONTAINER_PATH,
        )
        with tempfile.NamedTemporaryFile(
            "w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(converter)
            host_script = f.name
        try:
            subprocess.run(
                ["docker", "cp", host_script, f"{task_id}:/tmp/_compat_transcript.py"],
                capture_output=True, text=True,
            )
            r = subprocess.run(
                ["docker", "exec", task_id, "python3", "/tmp/_compat_transcript.py"],
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                logger.warning("[%s] compat transcript conversion failed: %s",
                               task_id, r.stderr.strip()[:400])
            else:
                logger.info("[%s] compat transcript written: %s", task_id, r.stdout.strip())
        finally:
            Path(host_script).unlink(missing_ok=True)


def sh_quote(value: str) -> str:
    """Single-quote a value for safe shell interpolation."""
    return "'" + str(value).replace("'", "'\"'\"'") + "'"


# In-container python: best-effort cancel the in-flight chat request (on timeout).
_INTERRUPT_SCRIPT = '''\
from __future__ import annotations
import asyncio
import sys

sys.path.insert(0, "/tmp")
from _jiuwenswarm_client import AgentServerClient


async def main() -> int:
    try:
        async with AgentServerClient({uri!r}) as client:
            await asyncio.wait_for(client.config_get(), timeout=10.0)
            await asyncio.wait_for(client.chat_interrupt({session_id!r}), timeout=10.0)
            print("[interrupt] ok", flush=True)
    except Exception as exc:
        print(f"[interrupt] failed: {{exc}}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
'''


# In-container python: drive the agent one task to completion/timeout.
# Imports the vendored high-level client (_jiuwenswarm_client.py), which in turn imports
# only from the installed jiuwenswarm package — so the websocket protocol matches exactly.
_DRIVE_SCRIPT = '''\
from __future__ import annotations
import asyncio
import sys

sys.path.insert(0, "/tmp")
from _jiuwenswarm_client import AgentServerClient


async def main() -> int:
    uri = {uri!r}
    session_id = {session_id!r}
    mode = {mode}
    timeout = float({timeout})
    prompt = sys.stdin.read()
    try:
        async with AgentServerClient(uri) as client:
            await asyncio.wait_for(client.config_get(), timeout=20.0)
            try:
                await client.session_delete(session_id)
            except Exception:
                pass
            await client.session_create(session_id)

            async def _run():
                chunks = await client.chat_send(
                    session_id, prompt, mode=mode, stream=True
                )
                return await AgentServerClient.collect_stream(chunks)

            try:
                resp = await asyncio.wait_for(_run(), timeout=timeout)
                content = (resp.payload or {{}}).get("content", "")
                print(f"[drive] answer_len={{len(content)}}", flush=True)
                return 0
            except asyncio.TimeoutError:
                try:
                    await client.chat_interrupt(session_id)
                except Exception:
                    pass
                print("[drive] timed_out", flush=True)
                return 0
    except Exception as exc:
        print(f"[drive] error: {{exc}}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
'''


# In-container python: convert jiuwenswarm session history → OpenClaw-style transcript.
# The jiuwenswarm history.jsonl is a list of events; we map role-tagged turns into the
# {type:message, message:{role,content,usage}} items the graders expect (see
# transcript_loader.load_transcript / grading.extract_usage_from_jsonl).
_COMPAT_SCRIPT = '''\
from __future__ import annotations
import json
import pathlib

HISTORY = pathlib.Path({history_path!r})
OUT = pathlib.Path({out_path!r})
OUT.parent.mkdir(parents=True, exist_ok=True)


def _load_events() -> list:
    if not HISTORY.exists():
        return []
    raw = HISTORY.read_text(encoding="utf-8", errors="ignore").strip()
    if not raw:
        return []
    try:
        whole = json.loads(raw)
        if isinstance(whole, list):
            return whole
    except json.JSONDecodeError:
        pass
    events = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            events.append(obj)
        elif isinstance(obj, list):
            events.extend(obj)
    return events


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", "") if block.get("type") == "text"
                             else json.dumps(block, ensure_ascii=False))
            else:
                parts.append(str(block))
        return "".join(parts)
    return json.dumps(content, ensure_ascii=False)


def main() -> None:
    events = _load_events()
    converted = []
    for ev in events:
        role = ev.get("role") or ev.get("speaker") or ""
        content = ev.get("content") or ev.get("text") or ev.get("message") or ""
        usage = ev.get("usage") or {{}}
        if not role:
            et = ev.get("event_type") or ev.get("type") or ""
            if et in ("user", "human", "chat.user"):
                role = "user"
            elif et in ("assistant", "chat.delta", "chat.assistant", "ai"):
                role = "assistant"
            elif et in ("tool", "tool_result"):
                role = "tool"
        if role not in ("user", "assistant", "tool"):
            continue
        if role == "tool":
            converted.append({{
                "type": "toolResult",
                "toolResult": {{
                    "content": _content_text(content),
                    "tool_call_id": str(ev.get("tool_call_id", "")),
                }},
            }})
            continue
        converted.append({{
            "type": "message",
            "message": {{
                "role": role,
                "content": _content_text(content),
                "usage": usage if isinstance(usage, dict) else {{}},
            }},
        }})
    payload = ""
    if converted:
        payload = "\\n".join(json.dumps(item, ensure_ascii=False) for item in converted) + "\\n"
    OUT.write_text(payload, encoding="utf-8")
    print(f"wrote {{len(converted)}} messages to {{OUT}}")


if __name__ == "__main__":
    main()
'''
