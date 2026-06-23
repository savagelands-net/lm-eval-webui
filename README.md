# Lemonade lm-eval Benchmark WebUI

A small stdlib Python WebUI for selecting OpenAI-compatible chat models,
launching `lm-evaluation-harness` benchmarks, and viewing rolled-up leaderboard
results.

## Run

```bash
cd /mpool/iain/repos/lm-eval
/home/iain/.venv/lm-eval/bin/python -m lm_eval_webui
```

Then open <http://127.0.0.1:8080>.

The default OpenAI-compatible endpoint is `https://llm.savagelands.net`.
Ollama can be targeted with an OpenAI-compatible base URL such as
`http://localhost:11434/v1`.

## Notes

- Uses the `openai-compatible-chat-completions` lm-eval plugin, with the legacy
  `lemonade-chat-completions` alias retained for existing jobs.
- Generation-style (`generate_until`) tasks are the safest fit for chat
  completion backends.
- Leaderboard scores use curated primary metrics and category rollups.
- Job cleanup removes selected job metadata, logs, telemetry, and run outputs.
  Legacy jobs with missing artifact paths are safely ignored instead of treating
  an empty path as `.`.
