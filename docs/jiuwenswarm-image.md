# JiuwenSwarm Harness Image (wildclawbench-jiuwenswarm-ubuntu)

WildClawBench's other harnesses (openclaw / claudecode / codex / hermesagent) each ship
a prebuilt image with the agent baked in. JiuwenSwarm is **not** baked in: it is under
active version checking, so hard-coding a version into an image would be wrong. Instead:

- `wildclawbench-jiuwenswarm-ubuntu:v0.1` is a **thin** image = the shared
  `wildclawbench-ubuntu:v1.3` base + jiuwenswarm's runtime deps only.
- The **jiuwenswarm wheel is a per-task input**: uploaded on the bench test page,
  pip-installed into each task container at run start (version logged via
  `pip show jiuwenswarm`).
- An optional **uploaded `config.yaml` fully replaces** the wheel-bundled
  `/root/.jiuwenswarm/config/config.yaml` (then only `model`/`api_base`/`api_key` are
  rewritten to point at the in-container llm_forward).

## Build

```bash
# 1. Load the shared base image (from HuggingFace, once):
hf download internlm/WildClawBench Images/wildclawbench-ubuntu_v1.3.tar --repo-type dataset --local-dir .
docker load -i Images/wildclawbench-ubuntu_v1.3.tar

# 2. Build the thin image:
cd ~/git/WildClawBench
docker build -f docker/Dockerfile.jiuwenswarm -t wildclawbench-jiuwenswarm-ubuntu:v0.1 .
```

The image never needs rebuilding for a jiuwenswarm release — rotate versions purely by
uploading a different wheel per task.

## What happens at run start (per task container)

Implemented in `src/agents/jiuwenswarm/runner.py`:

1. Container starts from the thin image.
2. The uploaded wheel is `docker cp`'d to `/tmp/jw.whl` and `pip install`ed; the
   installed version is logged (feeds version-checking).
3. Config: if the task uploaded a `config.yaml`, it **replaces** the wheel-bundled one
   verbatim; otherwise the wheel's own config is used (or a minimal one is generated).
   Then `models.defaults[0]` is rewritten to point at the in-container llm_forward:
   `api_base: http://127.0.0.1:8000`, `model_name: <task model>`, `client_provider: OpenAI`.
4. An in-container **llm_forward** proxy is started (upstream = the real model endpoint
   from the task's BASE_URL/API_KEY). All model traffic flows through it, and
   **token usage is read from its `/token-stats`** — the per-task source of truth, not
   the agent transcript.
5. `jiuwenswarm-agentserver` starts on the container loopback (`:18092`), and a small
   in-container drive script (using the vendored high-level client) runs one task to
   completion/timeout.
6. For grading, the session's `history.jsonl` is converted to the OpenClaw-style
   transcript the embedded graders expect.

## Manual smoke test

```bash
cd ~/git/WildClawBench
python eval/run_batch.py --agent-backend jiuwenswarm \
  --jiuwenswarm-whl /path/to/jiuwenswarm-<ver>-py3-none-any.whl \
  --jiuwenswarm-config /path/to/config.yaml \   # optional
  --task tasks/01_Productivity_Flow/01_task_001.md \
  --model <model-name>
```

Expect, under `output/jiuwenswarm/<category>/<task_id_ori>/<suffix>/`:
`score.json` (with `overall_score`) and `usage.json` (populated from llm_forward).

## Env overrides

| Var | Default | Purpose |
|---|---|---|
| `DOCKER_IMAGE_JIUWENSWARM` | `wildclawbench-jiuwenswarm-ubuntu:v0.1` | Thin image tag |
| `JIUWENSWARM_GATEWAY_PORT` | `18092` | In-container agentserver port |
| `LLM_FORWARD_PORT` | `8000` | In-container llm_forward port |
| `LLM_FORWARD_SCRIPT_PATH` | *(empty)* | Host path of `llm_forward.py` (BenchRunner's) |
| `JIUWENSWARM_CLIENT_SCRIPT_PATH` | *(empty)* | Host path of the vendored `AgentServerClient` (BenchRunner's `utils/jiuwenswarm_client.py`) |
