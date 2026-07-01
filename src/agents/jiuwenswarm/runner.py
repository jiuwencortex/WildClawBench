from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from src.agents.base import AgentExecution, AgentTaskSpec, BaseAgent
from src.utils.docker_utils import (
    build_ca_cert_args,
    build_custom_hosts_args,
    inject_lobster_workspace,
    run_warmup,
    setup_skills,
    TMP_WORKSPACE,
)
from src.utils.grading import extract_usage_from_jsonl

load_dotenv()

logger = logging.getLogger(__name__)

JIUWENSWARM_IMAGE = os.environ.get(
    "JIUWENSWARM_DOCKER_IMAGE",
    "wildclawbench-jiuwenswarm-base:v0.1",
)
JIUWENSWARM_SOURCE_PATH = os.environ.get(
    "JIUWENSWARM_SOURCE_PATH",
    os.path.expanduser("~/workspace/jiuwenswarm"),
)

JIUWENSWARM_HOME = "/root/.jiuwenswarm"
JIUWENSWARM_INSTALL_DIR = "/opt/jiuwenswarm"
JIUWENSWARM_SESSIONS_DIR = f"{JIUWENSWARM_HOME}/agent/sessions"

AGENTSERVER_PORT = 18092
GATEWAY_PORT = 19001

OPENCLAW_COMPAT_TRANSCRIPT_PATH = "/root/.openclaw/agents/main/sessions/chat.jsonl"
BENCH_RUNNER_HOST_PATH = Path(__file__).with_name("bench_runner.py")
BENCH_CONFIG_CONTAINER_PATH = "/tmp/jiuwenswarm_bench_config.json"
COMPAT_TRANSCRIPT_HOST_PATH = Path(__file__).with_name("compat_transcript.py")


class JiuwenSwarmAgent(BaseAgent):
    def __init__(
        self,
        image: str | None = None,
        openrouter_api_key: str = "",
        openrouter_base_url: str = "https://openrouter.ai/api/v1",
        brave_api_key: str = "",
    ) -> None:
        self.image = image or JIUWENSWARM_IMAGE
        self.source_path = JIUWENSWARM_SOURCE_PATH
        self.openrouter_api_key = openrouter_api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.openrouter_base_url = openrouter_base_url
        self.brave_api_key = brave_api_key or os.environ.get("BRAVE_API_KEY", "")

    @property
    def expects_gateway(self) -> bool:
        return True

    @property
    def transcript_container_path(self) -> str:
        return OPENCLAW_COMPAT_TRANSCRIPT_PATH

    def prepare_grading_transcript(self, task_id: str) -> str:
        self._write_compat_transcript(task_id)
        return self.transcript_container_path

    def run_task(self, spec: AgentTaskSpec) -> AgentExecution:
        gateway_proc = None
        agent_proc = None
        elapsed_time = float(spec.timeout_seconds)

        try:
            api_key, base_url = self._resolve_runtime_provider(spec.model, spec.models_config)

            exec_path = os.path.join(spec.workspace_path, "exec")
            tmp_path = os.path.join(spec.workspace_path, "tmp")
            os.makedirs(exec_path, exist_ok=True)

            # 1. Start container
            self._start_container(
                spec.task_id,
                exec_path,
                api_key=api_key,
                base_url=base_url,
                extra_env=spec.task.get("env", ""),
                tmp_path=tmp_path,
                lobster_env=spec.lobster.get("env") if spec.lobster else None,
            )
            if spec.lobster:
                inject_lobster_workspace(spec.task_id, spec.lobster["workspace"])

            # 2. Copy source + install + init
            self._install_jiuwenswarm(spec.task_id)

            # 3. Prepare workspace
            self._prepare_workspace(spec.task_id)

            # 4. Setup skills
            setup_skills(
                spec.task_id,
                spec.task.get("skills", ""),
                spec.task.get("skills_path", ""),
                container_skills_root=f"{JIUWENSWARM_HOME}/agent/workspace/skills",
            )

            # 5. Run warmup
            run_warmup(spec.task_id, spec.task.get("warmup", ""))

            # 6. Configure jiuwenswarm (API keys, model)
            self._configure_jiuwenswarm(spec.task_id, api_key, base_url, spec.model)

            # 7. Write bench runner config
            self._write_bench_runner(
                spec.task_id, spec.prompt, spec.model, api_key, base_url,
            )

            # 8. Start AgentServer (background)
            self._start_agentserver(spec.task_id, spec.output_dir / "agentserver.log")
            logger.info("[%s] Waiting for AgentServer to be ready...", spec.task_id)
            if not self._wait_for_port(spec.task_id, AGENTSERVER_PORT, timeout=30):
                raise RuntimeError("AgentServer did not become ready within 30s")
            logger.info("[%s] AgentServer ready on port %s", spec.task_id, AGENTSERVER_PORT)

            # 9. Start Gateway (background)
            gateway_proc = self._start_gateway(spec.task_id, spec.output_dir / "gateway.log")
            logger.info("[%s] Waiting for Gateway to be ready...", spec.task_id)
            if not self._wait_for_port(spec.task_id, GATEWAY_PORT, timeout=30):
                raise RuntimeError("Gateway did not become ready within 30s")
            logger.info("[%s] Gateway ready on port %s", spec.task_id, GATEWAY_PORT)

            # 10. Run the task via bench runner
            start_time = time.perf_counter()
            agent_proc = self._run_bench_runner_background(
                task_id=spec.task_id,
                log_path=spec.output_dir / "agent.log",
            )

            logger.info("[%s] Waiting for jiuwenswarm to finish...", spec.task_id)
            try:
                agent_proc.wait(timeout=spec.timeout_seconds)
                elapsed_time = time.perf_counter() - start_time
                logger.info(
                    "[%s] jiuwenswarm finished, elapsed: %.2f seconds",
                    spec.task_id,
                    elapsed_time,
                )
            except subprocess.TimeoutExpired:
                logger.info("[%s] jiuwenswarm timed out...", spec.task_id)
                elapsed_time = float(spec.timeout_seconds)
                agent_proc.kill()
                agent_proc.wait()
            self._close_runner_streams(agent_proc)

            logger.info("[%s] jiuwenswarm exit code: %s", spec.task_id, agent_proc.returncode)
            self._cleanup_bench_config(spec.task_id)

            return AgentExecution(
                elapsed_time=elapsed_time,
                error=None,
                gateway_proc=gateway_proc,
                agent_proc=agent_proc,
            )
        except Exception as exc:
            if agent_proc is not None:
                self._close_runner_streams(agent_proc)
            self._cleanup_bench_config(spec.task_id)
            self._kill_background_services(spec.task_id)
            logger.error("[%s] jiuwenswarm execution error: %s", spec.task_id, exc)
            return AgentExecution(
                elapsed_time=float(spec.timeout_seconds),
                error=str(exc),
                gateway_proc=gateway_proc,
                agent_proc=agent_proc,
            )

    def collect_usage(self, task_id: str, output_dir: Path, elapsed_time: float) -> dict[str, Any]:
        transcript_host = output_dir / "chat.jsonl"
        output_dir.mkdir(parents=True, exist_ok=True)
        r_cp = subprocess.run(
            ["docker", "cp", f"{task_id}:{self.transcript_container_path}", str(transcript_host)],
            capture_output=True,
            text=True,
        )
        if r_cp.returncode == 0 and transcript_host.exists():
            usage = extract_usage_from_jsonl(transcript_host)
        else:
            logger.warning("[%s] Transcript copy failed: %s", task_id, r_cp.stderr.strip())
            usage = self._extract_usage_from_session_logs(task_id)

        if self._usage_has_no_tokens(usage):
            log_usage = self._extract_usage_from_agent_log(output_dir / "agent.log")
            if not self._usage_has_no_tokens(log_usage):
                usage = log_usage

        self._copy_session_log(task_id, output_dir)

        usage["elapsed_time"] = round(elapsed_time, 2)
        return usage

    # ------------------------------------------------------------------
    # Provider / thinking helpers
    # ------------------------------------------------------------------

    def _resolve_runtime_provider(self, model: str, models_config: dict | None) -> tuple[str, str]:
        api_key = self.openrouter_api_key
        base_url = self.openrouter_base_url
        config_key, config_base_url = self._resolve_provider_config(model, models_config)
        if config_key:
            api_key = config_key
        if config_base_url:
            base_url = config_base_url
        return api_key, base_url

    @staticmethod
    def _resolve_provider_config(model: str, models_config: dict | None) -> tuple[str, str]:
        if not models_config:
            return "", ""
        providers = models_config.get("providers", {})
        for _prov_name, prov in providers.items():
            if not isinstance(prov, dict):
                continue
            for m in prov.get("models", []):
                if isinstance(m, dict) and m.get("id") == model:
                    return prov.get("apiKey", ""), prov.get("baseUrl", "")
        if providers:
            first = next(iter(providers.values()))
            if isinstance(first, dict):
                return first.get("apiKey", ""), first.get("baseUrl", "")
        return "", ""

    # ------------------------------------------------------------------
    # Container setup
    # ------------------------------------------------------------------

    def _start_container(
        self,
        task_id: str,
        workspace_path: str,
        api_key: str = "",
        base_url: str = "",
        extra_env: str = "",
        tmp_path: str = "",
        lobster_env: list[str] | None = None,
    ) -> None:
        proxy_http = os.environ.get("HTTP_PROXY_INNER", "")
        proxy_https = os.environ.get("HTTPS_PROXY_INNER", "")
        env_args = [
            "-e", f"BRAVE_API_KEY={self.brave_api_key}",
            "-e", f"OPENROUTER_API_KEY={api_key}",
            "-e", f"OPENROUTER_BASE_URL={base_url}",
        ]
        if proxy_http:
            env_args += [
                "-e", f"http_proxy={proxy_http}",
                "-e", f"HTTP_PROXY={proxy_http}",
            ]
        if proxy_https:
            env_args += [
                "-e", f"https_proxy={proxy_https}",
                "-e", f"HTTPS_PROXY={proxy_https}",
            ]
        if proxy_http or proxy_https:
            no_proxy = os.environ.get("NO_PROXY_INNER", "")
            env_args += ["-e", f"no_proxy={no_proxy}"]
        for line in extra_env.splitlines():
            key = line.strip()
            if not key or key.startswith("#"):
                continue
            value = os.environ.get(key, "")
            env_args += ["-e", f"{key}={value}"]
        for key in (lobster_env or []):
            value = os.environ.get(key, "")
            if not value:
                continue
            env_args += ["-e", f"{key}={value}"]

        if os.environ.get("CA_CERTIFICATES_HOST_PATH", "").strip():
            system_ca_bundle = "/etc/ssl/certs/ca-certificates.crt"
            for var_name in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
                env_args += ["-e", f"{var_name}={system_ca_bundle}"]

        networking_args = build_custom_hosts_args() + build_ca_cert_args()

        cmd = [
            "docker", "run", "-d",
            "--name", task_id,
            *env_args,
            *networking_args,
            "-v", f"{workspace_path}:/app:ro",
            self.image,
            "/bin/bash", "-c", "tail -f /dev/null",
        ]
        logger.info("[%s] Starting jiuwenswarm container (image=%s)", task_id, self.image)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"jiuwenswarm container startup failed:\n{r.stderr}")
        logger.info("[%s] Container ID: %s", task_id, r.stdout.strip()[:12])
        self._update_ca_certificates(task_id)

        if tmp_path and os.path.exists(tmp_path):
            subprocess.run(
                ["docker", "exec", task_id, "mkdir", "-p", "/tmp_workspace/tmp"],
                capture_output=True,
            )
            cp_r = subprocess.run(
                ["docker", "cp", f"{tmp_path}/.", f"{task_id}:/tmp_workspace/tmp/"],
                capture_output=True, text=True,
            )
            if cp_r.returncode != 0:
                logger.error("[%s] Temp file copy failed: %s", task_id, cp_r.stderr)

    def _update_ca_certificates(self, task_id: str) -> None:
        if not os.environ.get("CA_CERTIFICATES_HOST_PATH", "").strip():
            return
        logger.info("[%s] Updating CA certificates in jiuwenswarm container...", task_id)
        r = subprocess.run(
            ["docker", "exec", task_id, "update-ca-certificates"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            logger.warning("[%s] update-ca-certificates failed: %s", task_id, r.stderr.strip())
        else:
            logger.info("[%s] CA certificates updated successfully", task_id)

    def _prepare_workspace(self, task_id: str) -> None:
        r = subprocess.run(
            [
                "docker", "exec", task_id, "/bin/bash", "-c",
                f"mkdir -p {TMP_WORKSPACE} && cp -r /app/. {TMP_WORKSPACE} && chmod -R u+w {TMP_WORKSPACE}",
            ],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"jiuwenswarm workspace copy failed:\n{r.stderr}")

    # ------------------------------------------------------------------
    # JiuwenSwarm install & configure
    # ------------------------------------------------------------------

    def _install_jiuwenswarm(self, task_id: str) -> None:
        """Copy source from host and install jiuwenswarm inside the container."""
        if not os.path.isdir(self.source_path):
            raise RuntimeError(f"jiuwenswarm source path not found: {self.source_path}")

        # Create install directory
        r_mkdir = subprocess.run(
            ["docker", "exec", task_id, "mkdir", "-p", JIUWENSWARM_INSTALL_DIR],
            capture_output=True, text=True,
        )
        if r_mkdir.returncode != 0:
            raise RuntimeError(f"mkdir install dir failed:\n{r_mkdir.stderr}")

        # Copy source
        logger.info("[%s] Copying jiuwenswarm source from %s", task_id, self.source_path)
        r_cp = subprocess.run(
            ["docker", "cp", f"{self.source_path}/.", f"{task_id}:{JIUWENSWARM_INSTALL_DIR}/"],
            capture_output=True, text=True,
        )
        if r_cp.returncode != 0:
            raise RuntimeError(f"jiuwenswarm source copy failed:\n{r_cp.stderr}")

        # Since the base image already has jiuwenswarm installed in editable mode
        # (via __editable__.jiuwenswarm-0.2.2.pth pointing to /opt/jiuwenswarm),
        # we just need to overwrite the source files. No pip reinstall needed.
        # Only force reinstall if JIUWENSWARM_FORCE_REINSTALL=1.
        force_reinstall = os.environ.get("JIUWENSWARM_FORCE_REINSTALL", "") == "1"
        if force_reinstall:
            logger.info("[%s] Force reinstall requested...", task_id)
            r_install = subprocess.run(
                ["docker", "exec", task_id, "/bin/bash", "-c",
                 f"cd {JIUWENSWARM_INSTALL_DIR} && pip install -e . --no-build-isolation --no-cache-dir 2>&1"],
                capture_output=True, text=True,
            )
            if r_install.returncode != 0:
                raise RuntimeError(f"jiuwenswarm reinstall failed:\n{r_install.stderr}")
            logger.info("[%s] jiuwenswarm reinstalled successfully", task_id)
        else:
            logger.info("[%s] Skipping pip reinstall (editable install picks up source changes)", task_id)

        # Initialize workspace
        logger.info("[%s] Initializing jiuwenswarm workspace...", task_id)
        r_init = subprocess.run(
            ["docker", "exec", task_id, "/bin/bash", "-c",
             "export JIUWENSWARM_SKIP_DOTENV=1 && jiuwenswarm-init"],
            capture_output=True, text=True,
        )
        if r_init.returncode != 0:
            logger.warning("[%s] jiuwenswarm-init reported issues:\n%s", task_id, r_init.stderr)
        else:
            logger.info("[%s] jiuwenswarm workspace initialized", task_id)

    def _configure_jiuwenswarm(
        self, task_id: str, api_key: str, base_url: str, model: str,
    ) -> None:
        """Write model/provider config to jiuwenswarm's .env file."""
        env_content = (
            f'API_BASE={base_url}\n'
            f'API_KEY={api_key}\n'
            f'MODEL_NAME={model}\n'
            f'MODEL_PROVIDER=OpenAI\n'
            f'CUSTOM_HEADERS=\n'
            f'\n'
            f'EMBED_API_BASE=\n'
            f'EMBED_API_KEY=\n'
            f'EMBED_MODEL=\n'
            f'\n'
            f'VIDEO_API_BASE=\n'
            f'VIDEO_API_KEY=\n'
            f'VIDEO_MODEL_NAME=\n'
            f'VIDEO_PROVIDER=\n'
            f'\n'
            f'AUDIO_API_BASE=\n'
            f'AUDIO_API_KEY=\n'
            f'AUDIO_MODEL_NAME=\n'
            f'AUDIO_PROVIDER=\n'
            f'\n'
            f'VISION_API_BASE=\n'
            f'VISION_API_KEY=\n'
            f'VISION_MODEL_NAME=\n'
            f'VISION_PROVIDER=\n'
            f'\n'
            f'BRAVE_API_KEY={self.brave_api_key}\n'
            f'NO_PROXY=localhost,127.0.0.1\n'
        )

        with tempfile.NamedTemporaryFile("w", suffix=".env", delete=False, encoding="utf-8") as f:
            f.write(env_content)
            env_tmp = f.name

        try:
            r = subprocess.run(
                ["docker", "cp", env_tmp, f"{task_id}:{JIUWENSWARM_HOME}/config/.env"],
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                raise RuntimeError(f"Failed to copy .env into container:\n{r.stderr}")
        finally:
            Path(env_tmp).unlink(missing_ok=True)

        logger.info("[%s] jiuwenswarm configured with model=%s", task_id, model)

    # ------------------------------------------------------------------
    # Bench runner
    # ------------------------------------------------------------------

    def _write_bench_runner(
        self,
        task_id: str,
        prompt: str,
        model: str,
        api_key: str,
        base_url: str,
    ) -> None:
        config_payload = {
            "config": {
                "model": model,
                "api_key": api_key,
                "base_url": base_url,
                "timeout": 600,
            },
            "prompt": prompt,
            "session_id": task_id,
        }

        config_tmp = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", suffix=".json", delete=False, encoding="utf-8",
            ) as f:
                json.dump(config_payload, f, ensure_ascii=False)
                config_tmp = f.name

            r = subprocess.run(
                ["docker", "cp", config_tmp, f"{task_id}:{BENCH_CONFIG_CONTAINER_PATH}"],
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                raise RuntimeError(f"Failed to copy bench config into container:\n{r.stderr}")
        finally:
            if config_tmp:
                Path(config_tmp).unlink(missing_ok=True)

    def _run_bench_runner_background(self, task_id: str, log_path: Path) -> subprocess.Popen[str]:
        if not BENCH_RUNNER_HOST_PATH.exists():
            raise RuntimeError(f"bench_runner.py not found: {BENCH_RUNNER_HOST_PATH}")

        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = log_path.open("w", encoding="utf-8")
        script_file = BENCH_RUNNER_HOST_PATH.open("r", encoding="utf-8")
        proc = subprocess.Popen(
            [
                "docker",
                "exec",
                "-i",
                task_id,
                "/bin/bash",
                "-c",
                f"cd {JIUWENSWARM_INSTALL_DIR} && python3 -",
            ],
            stdin=script_file,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        )
        proc._log_file = log_file  # type: ignore[attr-defined]
        proc._script_file = script_file  # type: ignore[attr-defined]
        logger.info("[%s] Started jiuwenswarm bench runner PID=%s -> %s", task_id, proc.pid, log_path)
        return proc

    @staticmethod
    def _close_runner_streams(proc: subprocess.Popen[str] | None) -> None:
        if proc is None:
            return
        stream = getattr(proc, "_script_file", None)
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass

    @staticmethod
    def _cleanup_bench_config(task_id: str) -> None:
        subprocess.run(
            ["docker", "exec", task_id, "rm", "-f", BENCH_CONFIG_CONTAINER_PATH],
            capture_output=True, text=True,
        )

    # ------------------------------------------------------------------
    # Service lifecycle (background processes inside container)
    # ------------------------------------------------------------------

    def _start_agentserver(self, task_id: str, log_path: Path) -> None:
        """Start AgentServer as a background process inside the container."""
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("[%s] Starting AgentServer...", task_id)
        r = subprocess.run(
            ["docker", "exec", "-d", task_id, "/bin/bash", "-c",
             f"export JIUWENSWARM_SKIP_DOTENV=1 && "
             f"cd {JIUWENSWARM_HOME}/config && "
             f"export $(grep -v '^#' .env | xargs) && "
             f"jiuwenswarm-agentserver > /tmp/agentserver.log 2>&1 & "
             f"echo $! > /tmp/agentserver.pid"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"Failed to start AgentServer:\n{r.stderr}")

    def _start_gateway(self, task_id: str, log_path: Path) -> subprocess.Popen[str]:
        """Start Gateway as a background process inside the container.

        Returns a Popen handle for cleanup tracking (gateway runs detached).
        """
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("[%s] Starting Gateway...", task_id)
        r = subprocess.run(
            ["docker", "exec", "-d", task_id, "/bin/bash", "-c",
             f"export JIUWENSWARM_SKIP_DOTENV=1 && "
             f"cd {JIUWENSWARM_HOME}/config && "
             f"export $(grep -v '^#' .env | xargs) && "
             f"jiuwenswarm-gateway > /tmp/gateway.log 2>&1 & "
             f"echo $! > /tmp/gateway.pid"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"Failed to start Gateway:\n{r.stderr}")
        # Return a dummy Popen for interface compatibility
        return subprocess.Popen(["true"])

    @staticmethod
    def _wait_for_port(task_id: str, port: int, timeout: float = 30.0) -> bool:
        """Poll a TCP port inside the container until it is listening."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            r = subprocess.run(
                ["docker", "exec", task_id, "/bin/bash", "-c",
                 f"python3 -c \"import socket; s=socket.socket(); s.connect(('127.0.0.1',{port})); s.close()\" 2>/dev/null && echo 'ready'"],
                capture_output=True, text=True,
            )
            if "ready" in r.stdout:
                return True
            time.sleep(0.5)
        return False

    def _kill_background_services(self, task_id: str) -> None:
        """Kill AgentServer and Gateway processes inside the container."""
        logger.info("[%s] Killing background jiuwenswarm services...", task_id)
        subprocess.run(
            ["docker", "exec", task_id, "/bin/bash", "-c",
             "for pidfile in /tmp/agentserver.pid /tmp/gateway.pid; do "
             "  if [ -f \\$pidfile ]; then kill -9 \\$(cat \\$pidfile) 2>/dev/null; rm -f \\$pidfile; fi; "
             "done; pkill -9 -f 'jiuwenswarm-(agentserver|gateway)' 2>/dev/null"],
            capture_output=True, text=True,
        )

    # ------------------------------------------------------------------
    # Transcript conversion
    # ------------------------------------------------------------------

    def _write_compat_transcript(self, task_id: str) -> None:
        """Convert jiuwenswarm history.jsonl to OpenClaw-compatible chat.jsonl."""
        if not COMPAT_TRANSCRIPT_HOST_PATH.exists():
            logger.warning(
                "[%s] compat_transcript.py not found: %s",
                task_id, COMPAT_TRANSCRIPT_HOST_PATH,
            )
            return

        # Copy script into container
        r = subprocess.run(
            ["docker", "cp", str(COMPAT_TRANSCRIPT_HOST_PATH),
             f"{task_id}:/tmp/compat_transcript.py"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            logger.warning("[%s] Compat script copy failed: %s", task_id, r.stderr)
            return

        # Run conversion
        r_run = subprocess.run(
            ["docker", "exec", task_id, "/bin/bash", "-c",
             f"export JIUWENSWARM_BENCH_SESSION={task_id} && "
             f"python3 /tmp/compat_transcript.py"],
            capture_output=True, text=True,
        )
        if r_run.returncode != 0:
            logger.warning("[%s] Compat transcript conversion failed: %s", task_id, r_run.stderr)
        else:
            logger.info("[%s] Compat transcript written to %s", task_id, OPENCLAW_COMPAT_TRANSCRIPT_PATH)

    # ------------------------------------------------------------------
    # Usage extraction (fallbacks)
    # ------------------------------------------------------------------

    def _extract_usage_from_session_logs(self, task_id: str) -> dict[str, Any]:
        """Fallback: extract usage from jiuwenswarm session history.jsonl."""
        usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "request_count": 0,
        }

        with tempfile.TemporaryDirectory(prefix="jiuwenswarm_usage_") as tmp_dir:
            sessions_host = Path(tmp_dir) / "sessions"
            sessions_host.mkdir(parents=True, exist_ok=True)
            copied = subprocess.run(
                ["docker", "cp", f"{task_id}:{JIUWENSWARM_SESSIONS_DIR}/.", str(sessions_host)],
                capture_output=True, text=True,
            )
            if copied.returncode != 0:
                return usage

            for session_dir in sessions_host.iterdir():
                if not session_dir.is_dir():
                    continue
                history_path = session_dir / "history.jsonl"
                if not history_path.exists():
                    history_path = session_dir / "history.json"
                if not history_path.exists():
                    continue

                try:
                    if history_path.suffix == ".jsonl":
                        for line in history_path.read_text(encoding="utf-8").splitlines():
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                record = json.loads(line)
                            except Exception:
                                continue
                            if record.get("role") == "assistant":
                                usage["request_count"] += 1
                                u = record.get("usage", {})
                                if isinstance(u, dict):
                                    usage["input_tokens"] += u.get("prompt_tokens", 0) or u.get("input_tokens", 0)
                                    usage["output_tokens"] += u.get("completion_tokens", 0) or u.get("output_tokens", 0)
                                    usage["total_tokens"] += u.get("total_tokens", 0)
                                    usage["cost_usd"] += u.get("cost", 0.0)
                    else:
                        data = json.loads(history_path.read_text(encoding="utf-8"))
                        if isinstance(data, list):
                            for record in data:
                                if record.get("role") == "assistant":
                                    usage["request_count"] += 1
                                    u = record.get("usage", {})
                                    if isinstance(u, dict):
                                        usage["input_tokens"] += u.get("prompt_tokens", 0) or u.get("input_tokens", 0)
                                        usage["output_tokens"] += u.get("completion_tokens", 0) or u.get("output_tokens", 0)
                                        usage["total_tokens"] += u.get("total_tokens", 0)
                                        usage["cost_usd"] += u.get("cost", 0.0)
                except Exception:
                    continue

        return usage

    @staticmethod
    def _usage_has_no_tokens(usage: dict[str, Any]) -> bool:
        return (
            usage.get("input_tokens", 0) == 0
            and usage.get("output_tokens", 0) == 0
            and usage.get("total_tokens", 0) == 0
        )

    def _extract_usage_from_agent_log(self, log_path: Path) -> dict[str, Any]:
        usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "request_count": 0,
        }
        if not log_path.exists():
            return usage

        for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "API Response received" not in line or "CompletionUsage(" not in line:
                continue
            usage["request_count"] += 1
            usage["input_tokens"] += (
                self._extract_int_from_log(line, "prompt_tokens")
                or self._extract_int_from_log(line, "input_tokens")
            )
            usage["output_tokens"] += (
                self._extract_int_from_log(line, "completion_tokens")
                or self._extract_int_from_log(line, "output_tokens")
            )
            usage["total_tokens"] += self._extract_int_from_log(line, "total_tokens")
            usage["cache_read_tokens"] += self._extract_int_from_log(line, "cached_tokens")
            usage["cache_write_tokens"] += self._extract_int_from_log(line, "cache_write_tokens")
            usage["cost_usd"] += self._extract_float_from_log(line, "cost")

        usage["cost_usd"] = round(usage["cost_usd"], 6)
        return usage

    @staticmethod
    def _extract_int_from_log(line: str, field: str) -> int:
        match = re.search(rf"\b{re.escape(field)}=(\d+)", line)
        return int(match.group(1)) if match else 0

    @staticmethod
    def _extract_float_from_log(line: str, field: str) -> float:
        match = re.search(rf"\b{re.escape(field)}=([0-9.eE+-]+)", line)
        return float(match.group(1)) if match else 0.0

    def _copy_session_log(self, task_id: str, output_dir: Path) -> None:
        """Copy all jiuwenswarm session logs from the container to output dir."""
        dest = output_dir / "jiuwenswarm_sessions"
        dest.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            ["docker", "cp", f"{task_id}:{JIUWENSWARM_SESSIONS_DIR}/.", str(dest)],
            capture_output=True, text=True,
        )
        if r.returncode == 0:
            logger.info("[%s] jiuwenswarm sessions copied to %s", task_id, dest)
        else:
            logger.warning("[%s] jiuwenswarm session copy failed: %s", task_id, r.stderr.strip())
