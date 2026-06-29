# Lemonade lm-eval Benchmark WebUI

A small stdlib Python WebUI for selecting OpenAI-compatible chat models,
launching `lm-evaluation-harness` benchmarks, and viewing rolled-up leaderboard
results.

## Run

```bash
cd lm-eval-webui
git submodule update --init --recursive
python -m lm_eval_webui
```

Then open <http://127.0.0.1:8080>.

The default OpenAI-compatible endpoint is `http://localhost:11434/v1`.
Set a different startup default with either:

```bash
OPENAI_BASE_URL="https://your-openai-compatible-host" python -m lm_eval_webui
# or
python -m lm_eval_webui --openai-base-url "https://your-openai-compatible-host"
```

The WebUI also lets you edit the OpenAI-compatible base URL before refreshing
models or starting benchmark jobs.

## SWE Mini / pi-bench

SWE Mini support uses upstream `pi-bench` as a clean git submodule at
`third_party/pi-bench`. WebUI-specific customizations live in this repo under
`scripts/` and `lm_eval_webui/`, so the submodule can be updated independently:

```bash
git submodule update --remote third_party/pi-bench
```

For gpt-5.5 judging, login with Pi on the machine running the WebUI:

```bash
pi
# then login for openai-codex
```

SWE Mini Docker jobs receive only a temporary copy of
`~/.pi/agent/auth.json`; the token file is not stored in WebUI job metadata.
Override the submodule location with:

```bash
python -m lm_eval_webui --pi-bench-dir /path/to/pi-bench
```

## Docker Compose

The Compose setup runs the WebUI plus a Docker-in-Docker sidecar. This lets SWE
Mini containers mount the shared workspace path inside the sidecar daemon.

```bash
git submodule update --init --recursive
OPENAI_BASE_URL="http://host.docker.internal:11434/v1" \
PI_AUTH_JSON="$HOME/.pi/agent/auth.json" \
  docker compose -f deploy/docker-compose.yml up --build
```

Then open <http://127.0.0.1:8080>.

## Kubernetes

Build and push an image that includes initialized submodules:

```bash
git submodule update --init --recursive
docker build -f deploy/Dockerfile -t savagemindz/lm-eval-webui:latest .
docker push savagemindz/lm-eval-webui:latest
```

Edit `deploy/k8s/deployment.yaml` to use that image, then create the Pi auth
secret and deploy:

```bash
kubectl apply -f deploy/k8s/namespace.yaml
kubectl -n lm-eval-webui create secret generic pi-auth \
  --from-file=auth.json="$HOME/.pi/agent/auth.json"
kubectl apply -f deploy/k8s/pvc.yaml
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/service.yaml
```

Set `OPENAI_BASE_URL` in `deploy/k8s/deployment.yaml` to the OpenAI-compatible
endpoint reachable from the pod. The manifest also points Hugging Face caches at
`/data/huggingface` so downloaded lm-eval datasets persist on the `lm-eval-data`
PVC across pod restarts. Transient Hugging Face dataset API failures are retried
by default; tune this with `LMEVAL_WEBUI_HF_RETRIES`,
`LMEVAL_WEBUI_HF_RETRY_DELAY`, and `LMEVAL_WEBUI_HF_RETRY_MAX_DELAY`.

The Kubernetes manifest uses a privileged Docker-in-Docker sidecar. If your
cluster disallows privileged pods, replace the sidecar with a cluster-native job
runner before enabling SWE Mini jobs.

## Notes

- This software was created with the help of AI coding assistants.
- Uses the `openai-compatible-chat-completions` lm-eval plugin, with the legacy
  `lemonade-chat-completions` alias retained for existing jobs.
- Generation-style (`generate_until`) tasks are the safest fit for chat
  completion backends.
- Leaderboard scores use curated primary metrics and category rollups.
- Job cleanup removes selected job metadata, logs, telemetry, and run outputs.
  Legacy jobs with missing artifact paths are safely ignored instead of treating
  an empty path as `.`.
