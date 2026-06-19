# Local lm-eval Benchmark WebUI

A small stdlib Python WebUI for selecting Lemonade-hosted models, launching
`lm-evaluation-harness` benchmarks, and viewing rolled-up leaderboard results.

## Run

```bash
cd /mpool/iain/repos/lm-eval
/home/iain/.venv/lm-eval/bin/python -m lm_eval_webui
```

Then open <http://127.0.0.1:8080>.

The default Lemonade host is `https://llm.savagelands.net`.

## Notes

- Uses the `lemonade-chat-completions` lm-eval plugin to avoid empty auth
  headers and to support Lemonade `reasoning_content`.
- Generation-style (`generate_until`) tasks are the safest fit for this backend.
- Leaderboard scores use curated primary metrics and category rollups.
- Job cleanup removes selected job metadata, logs, telemetry, and run outputs.
  Legacy jobs with missing artifact paths are safely ignored instead of treating
  an empty path as `.`.
