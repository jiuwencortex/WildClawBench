from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DOCKER_IMAGE  = os.environ.get("DOCKER_IMAGE",   "wildclawbench-ubuntu:v1.3")
TMP_WORKSPACE = os.environ.get("TMP_WORKSPACE",  "/tmp_workspace")
WORKSPACE_BASELINE_PATH = "/tmp/wildclaw_workspace_baseline.json"

BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")


def build_custom_hosts_args() -> list[str]:
    """Parse DOCKER_CUSTOM_HOSTS and return Docker --add-host arguments.

    Format: ``host1:ip1,host2:ip2`` or ``host1:ip1;host2:ip2``
    (comma or semicolon separated).

    Empty or unset variable returns an empty list.
    """
    raw = os.environ.get("DOCKER_CUSTOM_HOSTS", "").strip()
    if not raw:
        return []

    args: list[str] = []
    for segment in raw.replace(";", ",").split(","):
        segment = segment.strip()
        if not segment:
            continue
        if ":" not in segment:
            logger.warning("Skipping invalid custom host entry (missing colon): %s", segment)
            continue
        host, ip = segment.rsplit(":", 1)
        host = host.strip()
        ip = ip.strip()
        if not host or not ip:
            logger.warning("Skipping invalid custom host entry (empty host or ip): %s", segment)
            continue
        args += ["--add-host", f"{host}:{ip}"]
        logger.info("Docker custom host mapping: %s -> %s", host, ip)
    return args


def build_ca_cert_args() -> list[str]:
    """Parse CA_CERTIFICATES_HOST_PATH and return Docker volume mount arguments.

    Mounts the host CA certificates directory/file into the container's
    system trust store at ``/usr/local/share/ca-certificates/`` and runs
    ``update-ca-certificates`` via an entrypoint wrapper.

    Empty or unset variable returns an empty list.
    """
    raw = os.environ.get("CA_CERTIFICATES_HOST_PATH", "").strip()
    if not raw:
        return []

    host_path = Path(raw).expanduser()
    if not host_path.exists():
        logger.warning("CA_CERTIFICATES_HOST_PATH does not exist: %s", host_path)
        return []

    # Determine the target path inside the container
    if host_path.is_dir():
        target = "/usr/local/share/ca-certificates/custom"
        logger.info("Mounting CA certificates directory: %s -> %s", host_path, target)
        return ["-v", f"{host_path}:{target}:ro"]
    else:
        target = "/usr/local/share/ca-certificates/custom_ca.crt"
        logger.info("Mounting CA certificate file: %s -> %s", host_path, target)
        return ["-v", f"{host_path}:{target}:ro"]


def remove_container(name: str) -> None:
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)

def start_container(task_id: str, workspace_path: str, extra_env: str = "",
                    tmp_path: str = "", lobster_env: list[str] | None = None) -> None:
    workspace = Path(workspace_path).expanduser()
    if not workspace.is_dir():
        raise RuntimeError(f"Workspace path does not exist or is not a directory: {workspace}")

    proxy_http = os.environ.get('HTTP_PROXY_INNER', '')
    proxy_https = os.environ.get('HTTPS_PROXY_INNER', '')
    env_args = [
        "-e", f"http_proxy={proxy_http}",
        "-e", f"https_proxy={proxy_https}",
        "-e", f"HTTP_PROXY={proxy_http}",
        "-e", f"HTTPS_PROXY={proxy_https}",
        "-e", f"BRAVE_API_KEY={BRAVE_API_KEY}",
        "-e", f"no_proxy={'' if not proxy_http else os.environ.get('NO_PROXY_INNER', '')}",
    ]
    for line in extra_env.splitlines():
        key = line.strip()
        if not key or key.startswith("#"):
            continue
        value = os.environ.get(key, "")
        env_args += ["-e", f"{key}={value}"]
        masked = (value[:4] + "***") if value else "(empty)"
        logger.info("[%s] Injecting env var: %s=%s", task_id, key, masked)

    for key in (lobster_env or []):
        value = os.environ.get(key, "")
        if not value:
            logger.warning("[%s] Lobster env key %s not found in environment, skipping", task_id, key)
            continue
        env_args += ["-e", f"{key}={value}"]
        masked = value[:4] + "***"
        logger.info("[%s] Injecting lobster env: %s=%s", task_id, key, masked)

    # When custom CA certs are mounted, tell Python HTTP libraries to use the system bundle.
    if os.environ.get("CA_CERTIFICATES_HOST_PATH", "").strip():
        # update-ca-certificates produces the combined ca-certificates.crt
        system_ca_bundle = "/etc/ssl/certs/ca-certificates.crt"
        env_args += [
            "-e", f"SSL_CERT_FILE={system_ca_bundle}",
            "-e", f"REQUESTS_CA_BUNDLE={system_ca_bundle}",
            "-e", f"CURL_CA_BUNDLE={system_ca_bundle}",
        ]
        logger.info("[%s] Injecting CA bundle env vars: %s", task_id, system_ca_bundle)

    networking_args = build_custom_hosts_args() + build_ca_cert_args()

    cmd = [
        "docker", "run", "-d",
        "--name", task_id,
        *env_args,
        *networking_args,
        "-v", f"{workspace}:/app:ro",
        DOCKER_IMAGE,
        "/bin/bash", "-c", "tail -f /dev/null",
    ]
    logger.info("[%s] Starting container, mounting %s -> /app (ro)", task_id, workspace)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Container startup failed:\n{r.stderr}")
    logger.info("[%s] Container ID: %s", task_id, r.stdout.strip()[:12])
    _update_ca_certificates(task_id)

    if tmp_path and os.path.exists(tmp_path):
        mkdir_cmd = ["docker", "exec", task_id, "mkdir", "-p", "/tmp_workspace/tmp"]
        subprocess.run(mkdir_cmd, capture_output=True)

        cp_cmd = ["docker", "cp", f"{tmp_path}/.", f"{task_id}:/tmp_workspace/tmp/"]
        
        logger.info("[%s] Copying temp files: %s → /tmp_workspace/tmp", task_id, tmp_path)
        cp_r = subprocess.run(cp_cmd, capture_output=True, text=True)
        
        if cp_r.returncode != 0:
            logger.error("[%s] File copy failed: %s", task_id, cp_r.stderr)
        else:
            logger.info("[%s] Temp file copy complete", task_id)

def setup_workspace(task_id: str, thinking: str | None = None) -> None:
    logger.info("[%s] Copying /app → %s", task_id, TMP_WORKSPACE)
    r = subprocess.run(
        ["docker", "exec", task_id, "/bin/bash", "-c",
         f"cp -r /app/. {TMP_WORKSPACE} && chmod -R u+w {TMP_WORKSPACE}"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"Workspace copy failed:\n{r.stderr}")

    if thinking is not None:
        logger.info("[%s] Setting thinkingDefault to %s", task_id, thinking)
        thinking_result = subprocess.run(
            ["docker", "exec", task_id,
             "openclaw", "config", "set", "agents.defaults.thinkingDefault", thinking],
            capture_output=True, text=True,
        )
        if thinking_result.returncode != 0:
            raise RuntimeError(
                f"Failed to set thinkingDefault to {thinking}:\n{thinking_result.stderr}"
            )

    # Symlink OpenClaw workspace → TMP_WORKSPACE so the image tool's
    # media-local-roots check allows reading files under /tmp_workspace.
    subprocess.run(
        ["docker", "exec", task_id, "/bin/bash", "-c",
         f"rm -rf /root/.openclaw/workspace && ln -s {TMP_WORKSPACE} /root/.openclaw/workspace"],
        capture_output=True, text=True,
    )

def setup_skills(
    task_id: str,
    skills: str,
    skills_path: str,
    container_skills_root: str = "/root/skills",
) -> None:
    container_skills_root = container_skills_root.rstrip("/")
    subprocess.run(
        ["docker", "exec", task_id, "mkdir", "-p", container_skills_root],
        capture_output=True,
        text=True,
    )
    seen_dest_names: set[str] = set()
    for line in skills.splitlines():
        line = line.strip()
        if not line:
            continue
        src_rel = line.replace("\\", "/").strip("/")
        dest_name = PurePosixPath(src_rel).name
        if not dest_name:
            logger.warning("[%s] Invalid skill path %r, skipping", task_id, line)
            continue
        if dest_name in seen_dest_names:
            logger.warning(
                "[%s] Duplicate flattened skill target %s from %s, skipping",
                task_id,
                dest_name,
                line,
            )
            continue
        seen_dest_names.add(dest_name)
        subprocess.run(
            ["docker", "exec", task_id,
             "mkdir", "-p", f"{container_skills_root}/{dest_name}"],
            capture_output=True, text=True,
        )
        r = subprocess.run(
            ["docker", "cp",
             f"{skills_path}/{src_rel}/.", f"{task_id}:{container_skills_root}/{dest_name}/"],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            logger.warning(
                "[%s] Failed to copy skill %s to %s/%s: %s",
                task_id,
                line,
                container_skills_root,
                dest_name,
                r.stderr.strip(),
            )


def inject_openclaw_models(task_id: str, models_config: dict) -> None:
    """Inject custom models into ~/.openclaw/openclaw.json."""
    container_tmp_path = "/tmp/openclaw_models.json"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as tmp_file:
        json.dump(models_config, tmp_file, indent=2)
        tmp_file_path = tmp_file.name

    try:
        cp_r = subprocess.run(
            ["docker", "cp", tmp_file_path, f"{task_id}:{container_tmp_path}"],
            capture_output=True, text=True,
        )
        if cp_r.returncode != 0:
            raise RuntimeError(f"Failed to copy models config into container:\n{cp_r.stderr}")

        inject_cmd = f"""python3 - <<'PY'
import json
import pathlib

config_path = pathlib.Path('/root/.openclaw/openclaw.json')
models_path = pathlib.Path('{container_tmp_path}')

config = json.loads(config_path.read_text()) if config_path.exists() else {{}}
models = json.loads(models_path.read_text())
config['models'] = models

config_path.write_text(json.dumps(config, indent=2))
PY"""
        r = subprocess.run(
            ["docker", "exec", task_id, "/bin/bash", "-c", inject_cmd],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"Failed to inject models config:\n{r.stderr}")
    finally:
        Path(tmp_file_path).unlink(missing_ok=True)

    logger.info("[%s] Injected custom models config", task_id)


def run_warmup(
    task_id: str,
    warmup: str,
    *,
    detach_background: bool = False,
) -> None:
    """Execute warmup bash commands line by line inside the container (skip blank lines and comments)."""
    if not warmup.strip():
        return
    commands = [
        line.strip()
        for line in warmup.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not commands:
        return

    logger.info("[%s] Running warmup (%d commands)", task_id, len(commands))
    for idx, cmd in enumerate(commands, start=1):
        logger.info("[%s] warmup: %s", task_id, cmd)
        stripped_cmd = cmd.rstrip()
        if detach_background and stripped_cmd.endswith("&"):
            background_cmd = stripped_cmd[:-1].strip()
            log_path = f"/tmp/wildclaw_warmup_{idx}.log"
            wrapped = (
                f"cd {TMP_WORKSPACE} && "
                f"nohup /bin/bash -lc {shlex.quote(background_cmd)} "
                f"> {shlex.quote(log_path)} 2>&1 < /dev/null &"
            )
            r = subprocess.run(
                ["docker", "exec", task_id, "/bin/bash", "-lc", wrapped],
                capture_output=True,
                text=True,
            )
            if r.returncode != 0:
                raise RuntimeError(
                    f"Warmup background command failed: {cmd!r}\n{r.stderr}"
                )
            continue

        r = subprocess.run(
            ["docker", "exec", task_id, "/bin/bash", "-c", cmd],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"Warmup command failed: {cmd!r}\n{r.stderr}")


def run_background(task_id: str, bash_cmd: str, log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        ["docker", "exec", task_id, "/bin/bash", "-c",
         f"cd {TMP_WORKSPACE} && {bash_cmd}"],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
    )
    proc._log_file = log_file
    logger.info("[%s] Started process PID=%s → %s", task_id, proc.pid, log_path)
    return proc


def close_proc_log(proc: subprocess.Popen) -> None:
    """Close the log file handle created by run_background."""
    log_file = getattr(proc, "_log_file", None)
    if log_file and not log_file.closed:
        log_file.close()


def collect_output_from_container(
    task_id: str,
    output_dir: Path,
    *,
    include_workspace_changes: bool = False,
) -> None:
    """Collect task output files from the container to output_dir/task_output/.

    Collection strategy:
      1. All files under /tmp/openclaw/ (agent session logs, etc.)
      2. Task output files under /tmp_workspace/results/
      3. Optionally, the full /tmp_workspace/ tree for runners that may write
         deliverables outside results/
    """
    task_output_dir = output_dir / "task_output"
    task_output_dir.mkdir(parents=True, exist_ok=True)

    _copy_dir_from_container(task_id, "/tmp/openclaw/.", str(task_output_dir))

    workspace_out = task_output_dir / "workspace"
    workspace_out.mkdir(parents=True, exist_ok=True)

    if include_workspace_changes:
        ok = _copy_dir_from_container(task_id, f"{TMP_WORKSPACE}/.", str(workspace_out))
        if not ok:
            logger.warning("[%s] workspace directory does not exist or is empty", task_id)
        return

    results_out = workspace_out / "results"
    results_out.mkdir(parents=True, exist_ok=True)
    ok = _copy_dir_from_container(
        task_id, f"{TMP_WORKSPACE}/results/.", str(results_out),
    )
    if not ok:
        logger.warning("[%s] results/ directory does not exist or is empty", task_id)


def snapshot_workspace_state(task_id: str) -> None:
    snapshot_cmd = "python3 - <<'PY'\n" + "\n".join([
        "import json",
        "from pathlib import Path",
        f"root = Path({TMP_WORKSPACE!r})",
        f"snapshot_path = Path({WORKSPACE_BASELINE_PATH!r})",
        "files = {}",
        "if root.exists():",
        "    for path in root.rglob('*'):",
        "        if not (path.is_file() or path.is_symlink()):",
        "            continue",
        "        try:",
        "            stat = path.lstat()",
        "        except OSError:",
        "            continue",
        "        rel = path.relative_to(root).as_posix()",
        "        files[rel] = {",
        "            'size': stat.st_size,",
        "            'mtime_ns': stat.st_mtime_ns,",
        "            'is_symlink': path.is_symlink(),",
        "        }",
        "snapshot_path.write_text(json.dumps(files, sort_keys=True), encoding='utf-8')",
        "PY",
    ])
    r = subprocess.run(
        ["docker", "exec", task_id, "/bin/bash", "-c", snapshot_cmd],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"Workspace baseline snapshot failed:\n{r.stderr}")
    logger.info("[%s] Workspace baseline saved at %s", task_id, WORKSPACE_BASELINE_PATH)


def _copy_changed_workspace_outputs_from_container(task_id: str, dest: Path) -> None:
    list_cmd = "python3 - <<'PY'\n" + "\n".join([
        "import json",
        "from pathlib import Path",
        f"root = Path({TMP_WORKSPACE!r})",
        f"snapshot_path = Path({WORKSPACE_BASELINE_PATH!r})",
        "if not snapshot_path.exists():",
        "    print(json.dumps([]))",
        "    raise SystemExit(0)",
        "before = json.loads(snapshot_path.read_text(encoding='utf-8'))",
        "excluded_prefixes = (",
        "    'results/', 'gt/', 'tmp/', '.git/',",
        "    'node_modules/', '.venv/', 'venv/', '__pycache__/', '.cache/',",
        ")",
        "excluded_names = {'results', 'gt', 'tmp', '.git', 'node_modules', '.venv', 'venv', '__pycache__', '.cache'}",
        "changed = []",
        "if root.exists():",
        "    for path in root.rglob('*'):",
        "        if not (path.is_file() or path.is_symlink()):",
        "            continue",
        "        rel = path.relative_to(root).as_posix()",
        "        if rel in excluded_names or rel.startswith(excluded_prefixes):",
        "            continue",
        "        try:",
        "            stat = path.lstat()",
        "        except OSError:",
        "            continue",
        "        current = {",
        "            'size': stat.st_size,",
        "            'mtime_ns': stat.st_mtime_ns,",
        "            'is_symlink': path.is_symlink(),",
        "        }",
        "        if before.get(rel) != current:",
        "            changed.append(rel)",
        "print(json.dumps(sorted(changed)))",
        "PY",
    ])
    r = subprocess.run(
        ["docker", "exec", task_id, "/bin/bash", "-c", list_cmd],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        logger.warning("[%s] Failed to list changed workspace outputs: %s", task_id, r.stderr.strip())
        return

    try:
        changed_paths = json.loads(r.stdout.strip() or "[]")
    except json.JSONDecodeError:
        logger.warning("[%s] Failed to parse changed workspace output list: %s", task_id, r.stdout[:200])
        return

    for rel_path in changed_paths:
        if not isinstance(rel_path, str) or rel_path.startswith("/") or ".." in Path(rel_path).parts:
            continue
        dest_path = dest / rel_path
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        _copy_file_from_container(task_id, f"{TMP_WORKSPACE}/{rel_path}", dest_path)


def inject_lobster_workspace(task_id: str, workspace_path: str) -> None:
    """Copy the entire lobster workspace into /root/ (the OpenClaw workspace in the image).

    This brings in everything: SOUL.md, USER.md, MEMORY.md, memory/, skills/, etc.
    """
    r = subprocess.run(
        ["docker", "cp", f"{workspace_path}/.", f"{task_id}:/root/"],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        logger.error("[%s] Lobster workspace copy failed: %s", task_id, r.stderr)
    else:
        logger.info("[%s] Lobster workspace copied: %s → /root/", task_id, workspace_path)


def _update_ca_certificates(task_id: str) -> None:
    """Run update-ca-certificates inside the container if CA certs were mounted.
    
    Also injects the custom CA certificate into Python certifi bundles so that
    Python HTTP clients (requests, httpx, urllib3, openai, etc.) trust it.
    """
    raw = os.environ.get("CA_CERTIFICATES_HOST_PATH", "").strip()
    if not raw:
        return

    host_path = Path(raw).expanduser()
    if not host_path.exists():
        logger.warning("[%s] CA_CERTIFICATES_HOST_PATH does not exist: %s", task_id, host_path)
        return

    logger.info("[%s] Updating CA certificates in container...", task_id)
    r = subprocess.run(
        ["docker", "exec", task_id, "update-ca-certificates"],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        logger.warning("[%s] update-ca-certificates failed: %s", task_id, r.stderr.strip())
    else:
        logger.info("[%s] System CA certificates updated", task_id)

    # Inject custom CA certs into Python certifi bundles
    # so requests/httpx/openai inside the container also trust them.
    _inject_ca_into_certifi(task_id, host_path)


def _inject_ca_into_certifi(task_id: str, host_path: Path) -> None:
    """Append custom CA certificate(s) to every certifi cacert.pem in the container."""
    # Determine source paths inside the container
    if host_path.is_dir():
        # When a directory is mounted, the individual .crt files are in /usr/local/share/ca-certificates/custom/
        custom_cert_paths = "/usr/local/share/ca-certificates/custom/*.crt"
    else:
        # When a single file is mounted
        custom_cert_paths = "/usr/local/share/ca-certificates/custom_ca.crt"

    inject_script = f"""python3 - <<'PY'
import glob
import os
import subprocess
import sys

custom_certs = []
for p in glob.glob("{custom_cert_paths}"):
    try:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()
            if content:
                custom_certs.append(content)
    except Exception as e:
        print(f"certifi-inject: failed to read {{p}}: {{e}}", file=sys.stderr)

if not custom_certs:
    print("certifi-inject: no custom CA certs found to inject")
    sys.exit(0)

custom_certs_text = "\\n\\n".join(custom_certs)

# Find all certifi cacert.pem files
result = subprocess.run(
    ["find", "/", "-name", "cacert.pem", "-path", "*/certifi/*"],
    capture_output=True, text=True,
)
certifi_paths = [p for p in result.stdout.strip().split("\\n") if p]

updated = 0
for certifi_path in certifi_paths:
    try:
        with open(certifi_path, "r", encoding="utf-8", errors="ignore") as f:
            existing = f.read()
        if custom_certs_text in existing:
            print(f"certifi-inject: already present in {{certifi_path}}")
            continue
        with open(certifi_path, "a", encoding="utf-8") as f:
            f.write("\\n\\n" + custom_certs_text + "\\n")
        updated += 1
        print(f"certifi-inject: updated {{certifi_path}}")
    except Exception as e:
        print(f"certifi-inject: failed to patch {{certifi_path}}: {{e}}", file=sys.stderr)

print(f"certifi-inject: patched {{updated}} certifi bundle(s)")
PY"""

    r = subprocess.run(
        ["docker", "exec", task_id, "/bin/bash", "-c", inject_script],
        capture_output=True,
        text=True,
    )
    for line in (r.stdout + r.stderr).strip().splitlines():
        if line.strip():
            logger.info("[%s] %s", task_id, line.strip())


def _copy_dir_from_container(task_id: str, src: str, dest: str) -> bool:
    r = subprocess.run(
        ["docker", "cp", f"{task_id}:{src}", dest],
        capture_output=True, text=True,
    )
    if r.returncode == 0:
        logger.info("[%s] Collected container directory %s → %s", task_id, src, dest)
        return True
    return False


def _copy_file_from_container(task_id: str, src: str, dest: Path) -> bool:
    r = subprocess.run(
        ["docker", "cp", f"{task_id}:{src}", str(dest)],
        capture_output=True,
        text=True,
    )
    if r.returncode == 0:
        logger.info("[%s] Collected container file %s → %s", task_id, src, dest)
        return True
    logger.warning("[%s] Container file copy failed (%s): %s", task_id, src, r.stderr.strip())
    return False
