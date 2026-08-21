# Lemonade Benchmark WebUI

A small stdlib Python WebUI for running Lemonade Bench, lm-evaluation-harness,
and SWE Mini against installed models. Each suite has its own leaderboard and
detailed result view.

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

## Lemonade Bench

Lemonade Bench is the first suite in the setup and result tabs. It wraps the
upstream `lemonade bench` command and records TTFT, output tokens per second,
request duration, peak VRAM/RAM, failed runs, backend, and context size. Its
scenario picker is populated from Lemonade's bundled scenario catalog and
includes chat, coding, long-context, embedding, and image-generation workloads.
Long-context scenarios remain opt-in.

The benchmark options support backend and context-size matrices, measurement
and warmup counts, request timeout, memory tracking, model reloads between runs,
and optional response logging. With the backend field blank, the WebUI uses each
llama.cpp model's registered backend instead of asking the CLI to try every
installed backend; enter explicit backends only for a cross-backend comparison.
These per-run selections use Lemonade's non-persistent `save_options=false`
behavior and do not rewrite model registrations. The CLI runs through a
loopback-only HTTP bridge so Python handles the remote TLS connection; this
avoids Lemonade CLI 11.6 `Failed to read connection` errors on valid responses
without changing the Lemonade server or benchmark result format. A CLI exit code
of zero is still classified as a failed job when its result contains no
successful requests. Lemonade Bench jobs are always serialized because the CLI controls
model loading and unloading on a shared Lemonade server. The
container image includes the checksum-pinned Lemonade 11.6 CLI and supports both
amd64 and arm64 builds. Source-based local runs require a compatible `lemonade`
CLI on `PATH`; override its location with `LEMONADE_CLI=/path/to/lemonade`.

## SWE Mini / pi-bench

SWE Mini support uses upstream `pi-bench` as a clean git submodule at
`third_party/pi-bench`. WebUI-specific customizations live in this repo under
`scripts/` and `lm_eval_webui/`, so the submodule can be updated independently:

```bash
git submodule update --remote third_party/pi-bench
```

SWE Mini judging uses a Lemonade model from the configured
OpenAI-compatible endpoint. The WebUI lets you choose the judge from the
available model list and defaults to `gpt-oss-120b-mxfp-GGUF` when it is
available.

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

Edit `deploy/k8s/statefulset.yaml` to use that image, then deploy:

```bash
kubectl apply -f deploy/k8s/namespace.yaml
# Optional, improves Hugging Face dataset download bandwidth/rate limits:
kubectl -n lm-eval-webui create secret generic huggingface-token \
  --from-literal=token="$HF_TOKEN"
kubectl apply -f deploy/k8s/pvc.yaml
kubectl apply -f deploy/k8s/service.yaml
kubectl apply -f deploy/k8s/statefulset.yaml
```

Set `OPENAI_BASE_URL` in `deploy/k8s/statefulset.yaml` to the OpenAI-compatible
endpoint reachable from the pod. If you previously deployed the old Deployment
manifest, delete or scale down that Deployment before applying the StatefulSet so
only one pod uses the `ReadWriteOnce` data PVC. The example runs as a
single-replica StatefulSet with `/data` mounted from the `lm-eval-data` PVC. It
also points Hugging Face
caches at `/data/huggingface` so downloaded lm-eval datasets persist across pod
restarts. Authenticated Hugging Face downloads are optional: if the
`huggingface-token` Secret exists, Kubernetes exposes it as `HF_TOKEN` and
`HUGGING_FACE_HUB_TOKEN` inside the WebUI container. Transient Hugging Face
dataset API failures are retried by default, and corrupt cached dataset metadata
is removed before retrying. Tune retries with `LMEVAL_WEBUI_HF_RETRIES`,
`LMEVAL_WEBUI_HF_RETRY_DELAY`, and `LMEVAL_WEBUI_HF_RETRY_MAX_DELAY`.

The Kubernetes manifest uses a privileged Docker-in-Docker sidecar. If your
cluster disallows privileged pods, replace the sidecar with a cluster-native job
runner before enabling SWE Mini jobs.

## Job control and API behavior

Queued and running jobs can be cancelled from the Jobs panel. Running
subprocess groups receive `SIGTERM` and then `SIGKILL` after a grace period;
SWE Mini containers are labelled and removed as part of cancellation. Active
jobs must be cancelled before they can be cleared or rerun. After an application
restart, queued jobs are resumed and jobs interrupted while running are marked
failed instead of remaining stuck.

The browser polls only lightweight job summaries and never overlaps polling
requests. Leaderboard data is refreshed when jobs reach a terminal state, while
detailed rows are loaded lazily in paginated requests. The backend builds each
result summary once, persists compact per-job summaries under the data directory,
and caches serialized ETag/gzip responses. HTTP request concurrency is bounded
to 16 workers by default; override it with:

```bash
python -m lm_eval_webui --max-request-workers 8
```

The relevant read APIs are:

- `GET /api/jobs` — lightweight job summaries
- `GET /api/jobs/<id>` — full job details
- `GET /api/jobs/<id>/log` — an efficient log tail
- `GET /api/leaderboard` — compact leaderboard entries
- `GET /api/results?offset=0&limit=1000&suite=lemonade_bench` — paginated Lemonade Bench rows
- `GET /api/results?offset=0&limit=1000&suite=lm_eval` — paginated lm-eval rows
- `GET /api/results?offset=0&limit=1000&suite=swe_mini` — paginated SWE Mini rows

## Balanced lm-eval profiles and scoring

The lm-eval task picker includes three versioned **Strix Balanced** profiles.
They use the same eight generation-compatible tasks and the same 32,768-token
reasoning budget so results remain comparable; only the number of examples per
task changes:

- **Quick Screen** — up to 50 examples per task (up to 400 requests per model)
- **Standard Compare** — up to 200 examples per task (up to 1,600 requests)
- **Full Validation** — every example from every profile task

The shared task set covers IFEval, GSM8K, MATH-500, MMLU-Pro computer science
and engineering, BBH logical deduction, ARC Challenge Chat, and JSON Schema
Bench. Applying a profile also selects task-default few-shot settings, batch size
1, request concurrency 2, a 7,200-second timeout, task batches of 4, chat
templating, sample logging, and one concurrent job. Changing any profile task or
setting marks the job as **Custom**. The backend verifies and persists the
profile rather than trusting a label sent by the browser, and older jobs are
shown as **Custom (legacy)**.

Leaderboard ranking is kept separate by profile. **Balanced Overall** is the
equal-weight mean of four category scores—Reasoning, Math, Instruction
Following, and Coding / Structured Output—so categories with more tasks do not
silently dominate. Recognized profile runs are ranked only after every required
task produced its canonical score. Custom and incomplete runs retain their
metrics but are not assigned a profile rank. Generation throughput and TTFT stay
separate from the quality percentage. Result tables also show each job's
wall-clock runtime from start through final cleanup, and completed job cards keep
that duration visible in a runtime badge. Legacy results fall back to lm-eval's
recorded evaluation time when full job timestamps are unavailable.

## Thinking models and quality preflight

Normal benchmark jobs are unlimited by default and preserve each task's own
few-shot setting. The OpenAI-compatible adapter allows at least 32,768 output
tokens, removes server-side task stop strings so they cannot terminate an
internal thinking block, and applies those stop strings locally to the final
answer. It accepts the `reasoning`, `reasoning_content`, and `analysis` stream
fields used by current vLLM, llama.cpp, and other compatible providers. Only
final `content` is scored; an unfinished reasoning trace is never treated as an
answer.

For performance telemetry, streaming requests ask OpenAI-compatible providers
for the final usage chunk. Native server timings are preferred; when vLLM does
not emit them, output tokens per second is calculated from completion-token
usage and the client-observed generation interval. Model context comes from the
advertised maximum or, for vLLM registry entries, the effective `ctx_size`
recipe setting. Runs completed before this telemetry was captured cannot be
reliably backfilled with tokens-per-second data.

A full run without **Limit** can be extremely expensive. Before starting one,
dry-run the all-model preflight:

```bash
python scripts/smoke-all-models.py \
  --webui-url https://lm-eval.example.net \
  --openai-base-url https://llm.example.net/v1
```

Review the discovered models and payload, then add `--run` to enqueue serial
GSM8K, generative MMLU, and IFEval jobs for every downloaded chat model
(non-chat TTS/transcription models are skipped). The preflight uses three
samples per task, enables sample logging, and fails any model that does not
return final answer content for every request or still reaches its generation
cap. Use repeated
`--model MODEL_ID` options to test a subset.

The `gpt-oss-120b-mxfp-GGUF` value in this WebUI is only the default SWE Mini
judge; selecting or listing it does not send an inference request or preload it.
Lemonade loads a model when a client asks for that model. With
`max_loaded_models=1`, unrelated client traffic could otherwise evict a
benchmark model.

lm-eval jobs load and pin their selected model before the first request, keep it
pinned across every task batch, and unpin it when the job succeeds, fails, or is
cancelled. A competing model request receives HTTP 409 instead of evicting the
benchmark model. SWE Mini temporarily unpins the candidate and pins the
configured Lemonade judge while judging each task, then restores the candidate
pin before continuing. SWE Mini advertises a 65,536-token context and an
independent 16,384-token generation cap by default. This preserves typical
40K-token agent conversations while bounding generation and keeping a full cold
prefill inside the default 15-minute provider timeout. Automatic provider
retries are disabled, and the 60-minute agent task timeout remains independent.
Context, maximum output tokens, and provider timeout are configurable in the SWE
Mini benchmark options; provider retries are visibly locked to zero.

SWE Mini treats each Lemonade registration as read-only configuration. It loads
and pins the selected candidate and judge as already configured, but never
passes recipe, backend, slot, speculative-decoding/MTP, or save-options
overrides. The effective policy is persisted with each job as
`lemonade_unchanged`. A provider timeout is recorded as a zero-score
infrastructure failure; the runner waits for the candidate backend to become
idle before continuing so stale requests cannot overlap. Aggregate summaries
are rebuilt from every completed per-task artifact, including after partial
runs.

Lemonade Bench is intentionally not pinned because the
upstream CLI reloads models between scenarios and backend/context combinations.
Hosts without Lemonade's lifecycle endpoints continue without model protection
and record that state in the job log. Alternatively, increase
`max_loaded_models` only when the host has enough memory for every concurrently
used model.

## Notes

- This software was created with the help of AI coding assistants.
- Uses the `openai-compatible-chat-completions` lm-eval plugin, with the legacy
  `lemonade-chat-completions` alias retained for existing jobs.
- Generation-style (`generate_until`) tasks are the safest fit for chat
  completion backends.
- For broad/full lm-eval sweeps, **Task batch size** splits selected tasks into
  sequential subprocesses. The default `1` keeps memory lowest because each
  subprocess exits and releases loaded dataset/task state. Job cards report
  completed task batches separately from the live API-request count within the
  current task.
- Leaderboard scores use canonical primary metrics and equal-weight category
  rollups; use the profile filter when comparing models.
- Job cleanup removes selected job metadata, logs, telemetry, and run outputs.
  Legacy jobs with missing artifact paths are safely ignored instead of treating
  an empty path as `.`.
