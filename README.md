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

The default OpenAI-compatible endpoint is `https://llm.savagelands.net`.
Ollama can be targeted with an OpenAI-compatible base URL such as
`http://localhost:11434/v1`.

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

The Kubernetes manifest uses a privileged Docker-in-Docker sidecar. If your
cluster disallows privileged pods, replace the sidecar with a cluster-native job
runner before enabling SWE Mini jobs.

## Notes

- Uses the `openai-compatible-chat-completions` lm-eval plugin, with the legacy
  `lemonade-chat-completions` alias retained for existing jobs.
- Generation-style (`generate_until`) tasks are the safest fit for chat
  completion backends.
- Leaderboard scores use curated primary metrics and category rollups.
- Job cleanup removes selected job metadata, logs, telemetry, and run outputs.
  Legacy jobs with missing artifact paths are safely ignored instead of treating
  an empty path as `.`.
