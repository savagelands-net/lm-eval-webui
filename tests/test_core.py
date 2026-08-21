import asyncio
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
import types
import unittest
from importlib import import_module
from pathlib import Path
from typing import Any
from unittest import mock


def symbol(module_name, attribute):
    return import_module(module_name).__dict__[attribute]


class OpenAICompatibleEndpointTests(unittest.TestCase):
    def test_default_openai_base_url_points_to_localhost(self):
        default_openai_base_url = symbol(
            "lm_eval_webui.lemonade", "DEFAULT_OPENAI_BASE_URL"
        )

        self.assertEqual(default_openai_base_url, "http://localhost:11434/v1")

    def test_openai_api_url_accepts_root_or_v1_base(self):
        openai_api_url = symbol("lm_eval_webui.lemonade", "openai_api_url")

        self.assertEqual(
            openai_api_url("http://localhost:11434", "/models"),
            "http://localhost:11434/v1/models",
        )
        self.assertEqual(
            openai_api_url("http://localhost:11434/v1", "/models"),
            "http://localhost:11434/v1/models",
        )

    def test_openai_api_url_rejects_non_http_schemes(self):
        openai_api_url = symbol("lm_eval_webui.lemonade", "openai_api_url")

        for base_url in ("file:///etc/passwd", "ftp://example.test", "localhost:11434"):
            with (
                self.subTest(base_url=base_url),
                self.assertRaisesRegex(ValueError, "http:// or https://"),
            ):
                openai_api_url(base_url, "/models")

    def test_lemonade_management_url_is_derived_from_openai_base(self):
        management_url = symbol("lm_eval_webui.lemonade", "lemonade_management_url")

        self.assertEqual(
            management_url("https://llm.example.test/v1", "/api/v1/load"),
            "https://llm.example.test/api/v1/load",
        )
        self.assertEqual(
            management_url("https://proxy.example.test/llm/v1", "/internal/pin"),
            "https://proxy.example.test/llm/internal/pin",
        )

    def test_lemonade_model_lifecycle_uses_load_and_pin_endpoints(self):
        load_and_pin_model = symbol("lm_eval_webui.lemonade", "load_and_pin_model")
        unpin_model = symbol("lm_eval_webui.lemonade", "unpin_model")
        response = mock.Mock(status_code=200, content=b"{}")
        response.json.return_value = {}
        requests_module = types.SimpleNamespace(post=mock.Mock(return_value=response))

        with mock.patch(
            "lm_eval_webui.lemonade.importlib.import_module",
            return_value=requests_module,
        ):
            load_and_pin_model("https://llm.example.test/v1", "Model-A", 600)
            unpin_model("https://llm.example.test/v1", "Model-A", 30)

        self.assertEqual(requests_module.post.call_count, 2)
        load_call, unpin_call = requests_module.post.call_args_list
        self.assertEqual(load_call.args[0], "https://llm.example.test/api/v1/load")
        self.assertEqual(
            load_call.kwargs["json"], {"model_name": "Model-A", "pinned": True}
        )
        self.assertEqual(unpin_call.args[0], "https://llm.example.test/internal/pin")
        self.assertEqual(
            unpin_call.kwargs["json"], {"model": "Model-A", "pinned": False}
        )

    def test_eval_command_accepts_openai_v1_base_without_duplicate_path(self):
        EvalRequest = symbol("lm_eval_webui.runner", "EvalRequest")
        build_eval_command = symbol("lm_eval_webui.runner", "build_eval_command")

        command, _env = build_eval_command(
            EvalRequest(
                model_id="llama3.2",
                tasks=["gsm8k"],
                output_path="out",
                openai_base_url="http://localhost:11434/v1",
            ),
            project_root="/repo",
        )

        self.assertIn("base_url=http://localhost:11434/v1/chat/completions", command)
        self.assertNotIn(
            "base_url=http://localhost:11434/v1/v1/chat/completions", command
        )

    def test_eval_command_enables_streaming_for_in_run_ttft(self):
        EvalRequest = symbol("lm_eval_webui.runner", "EvalRequest")
        build_eval_command = symbol("lm_eval_webui.runner", "build_eval_command")

        command, _env = build_eval_command(
            EvalRequest(model_id="Model-A", tasks=["gsm8k"], output_path="out"),
            project_root="/repo",
        )

        self.assertIn("stream_responses=True", command)

    def test_eval_command_defaults_to_full_quality_run(self):
        EvalRequest = symbol("lm_eval_webui.runner", "EvalRequest")
        build_eval_command = symbol("lm_eval_webui.runner", "build_eval_command")

        command, _env = build_eval_command(
            EvalRequest(model_id="Model-A", tasks=["gsm8k"], output_path="out"),
            project_root="/repo",
        )

        self.assertIn("max_gen_toks=32768", command)
        self.assertIn("timeout=7200", command)
        self.assertNotIn("--limit", command)
        self.assertNotIn("--num_fewshot", command)

    def test_eval_command_applies_chat_template_by_default(self):
        EvalRequest = symbol("lm_eval_webui.runner", "EvalRequest")
        build_eval_command = symbol("lm_eval_webui.runner", "build_eval_command")

        command, _env = build_eval_command(
            EvalRequest(model_id="Model-A", tasks=["gsm8k"], output_path="out"),
            project_root="/repo",
        )

        self.assertIn("--apply_chat_template", command)

    def test_eval_command_passes_selected_llamacpp_backend(self):
        EvalRequest = symbol("lm_eval_webui.runner", "EvalRequest")
        build_eval_command = symbol("lm_eval_webui.runner", "build_eval_command")

        command, _env = build_eval_command(
            EvalRequest(
                model_id="Model-A",
                tasks=["gsm8k"],
                output_path="out",
                llamacpp_backend="vulkan",
            ),
            project_root="/repo",
        )

        self.assertIn("llamacpp_backend=vulkan", command)


class LemonadeBenchTests(unittest.TestCase):
    @staticmethod
    def result_payload(measurement_runs=3):
        return {
            "timestamp": "2026-08-18T20:00:00Z",
            "hardware": {"os": "Linux"},
            "models": [
                {
                    "model": "user.Qwen/Model-A",
                    "timestamp": "2026-08-18T20:00:01Z",
                    "config": {
                        "measurement_runs": measurement_runs,
                        "warmup_runs": 0,
                        "memory_tracking": True,
                    },
                    "results": [
                        {
                            "recipe": "vllm",
                            "backend": "rocm",
                            "ctx_size": 262144,
                            "backend_args": "--kv-cache-memory-bytes 10G",
                            "scenarios": [
                                {
                                    "name": "chat-short",
                                    "category": "chat",
                                    "input_tokens": 100,
                                    "output_tokens": 20,
                                    "ttft_ms": {
                                        "mean": 100,
                                        "min": 90,
                                        "max": 110,
                                        "p50": 100,
                                        "p95": 109,
                                    },
                                    "duration_ms": {
                                        "mean": 1000,
                                        "min": 900,
                                        "max": 1100,
                                        "p50": 1000,
                                        "p95": 1090,
                                    },
                                    "tps": {
                                        "mean": 20,
                                        "min": 18,
                                        "max": 22,
                                        "p50": 20,
                                        "p95": 21.8,
                                    },
                                    "vram_peak_gb": 60,
                                    "memory_peak_gb": 70,
                                    "failed_runs": 0,
                                },
                                {
                                    "name": "code-short",
                                    "category": "coding",
                                    "input_tokens": 80,
                                    "output_tokens": 30,
                                    "ttft_ms": {
                                        "mean": 200,
                                        "min": 180,
                                        "max": 220,
                                        "p50": 200,
                                        "p95": 218,
                                    },
                                    "duration_ms": {
                                        "mean": 2000,
                                        "min": 1800,
                                        "max": 2200,
                                        "p50": 2000,
                                        "p95": 2180,
                                    },
                                    "tps": {
                                        "mean": 10,
                                        "min": 9,
                                        "max": 11,
                                        "p50": 10,
                                        "p95": 10.9,
                                    },
                                    "vram_peak_gb": 61,
                                    "memory_peak_gb": 72,
                                    "failed_runs": 1,
                                },
                            ],
                        }
                    ],
                }
            ],
        }

    def test_command_targets_lemonade_server_and_persists_json(self):
        LemonadeBenchRequest = symbol(
            "lm_eval_webui.lemonade_bench", "LemonadeBenchRequest"
        )
        build_command = symbol(
            "lm_eval_webui.lemonade_bench", "build_lemonade_bench_command"
        )

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "run" / "results.json"
            command, env = build_command(
                LemonadeBenchRequest(
                    model_id="Qwen/Model-A",
                    lemonade_model_id="user.Qwen/Model-A",
                    scenarios=["chat-short", "code-short"],
                    output_path=str(output_path),
                    openai_base_url="https://llm.example.test/v1",
                    backends=["rocm"],
                    context_sizes=[4096, 32768],
                    measurement_runs=5,
                    warmup_runs=1,
                    timeout=600,
                    memory_tracking=False,
                    reload_between_runs=False,
                    log_responses=True,
                    lemonade_cli="/usr/local/bin/lemonade",
                )
            )

        self.assertEqual(
            command[:6],
            [
                sys.executable,
                "-m",
                "lm_eval_webui.lemonade_bench_proxy",
                "--upstream",
                "https://llm.example.test",
                "--",
            ],
        )
        self.assertEqual(command[6:8], ["/usr/local/bin/lemonade", "bench"])
        self.assertIn("bench", command)
        self.assertIn("--json", command)
        self.assertIn("--output", command)
        self.assertIn(str(output_path), command)
        self.assertIn("--backend", command)
        self.assertIn("rocm", command)
        self.assertIn("--ctx-size", command)
        self.assertIn("4096", command)
        self.assertIn("32768", command)
        self.assertEqual(command.count("--scenarios"), 2)
        self.assertIn("--no-memory", command)
        self.assertIn("--no-reload", command)
        self.assertIn("--response-log", command)
        self.assertEqual(command[-1], "user.Qwen/Model-A")
        self.assertEqual(env["NO_COLOR"], "1")

    def test_loopback_proxy_forwards_cli_requests_to_upstream(self):
        import http.server

        run_with_proxy = symbol("lm_eval_webui.lemonade_bench_proxy", "run_with_proxy")
        requests = []

        class UpstreamHandler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                size = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(size)
                requests.append((self.path, body, self.headers.get("X-Test")))
                payload = json.dumps({"ok": True}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, _format, *_args):
                return

        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), UpstreamHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        with tempfile.TemporaryDirectory() as tmp:
            fake_cli = Path(tmp) / "fake-lemonade"
            fake_cli.write_text(
                """#!/usr/bin/env python3
import argparse
import json
import urllib.request
parser = argparse.ArgumentParser()
parser.add_argument("--host", required=True)
args, _ = parser.parse_known_args()
request = urllib.request.Request(
    args.host + "/api/v1/test",
    data=b'{"ping":true}',
    headers={"Content-Type": "application/json", "X-Test": "yes"},
)
with urllib.request.urlopen(request, timeout=10) as response:
    assert json.load(response) == {"ok": True}
""",
                encoding="utf-8",
            )
            fake_cli.chmod(0o755)
            try:
                returncode = run_with_proxy(
                    f"http://127.0.0.1:{server.server_port}/prefix",
                    [str(fake_cli), "bench"],
                    timeout=10,
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

        self.assertEqual(returncode, 0)
        self.assertEqual(
            requests,
            [("/prefix/api/v1/test", b'{"ping":true}', "yes")],
        )

    def test_scenario_catalog_reads_lemonade_resource_schema(self):
        find_scenarios = symbol(
            "lm_eval_webui.lemonade_bench", "find_lemonade_bench_scenarios"
        )

        with tempfile.TemporaryDirectory() as tmp:
            scenario_file = Path(tmp) / "bench_scenarios.json"
            scenario_file.write_text(
                json.dumps(
                    {
                        "scenarios": [
                            {
                                "name": "chat-short",
                                "category": "chat",
                                "max_tokens": 20,
                            },
                            {
                                "name": "context-32k",
                                "category": "long-context",
                                "max_tokens": 20,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            scenarios = find_scenarios(scenario_file)

        self.assertEqual(
            [item["name"] for item in scenarios], ["chat-short", "context-32k"]
        )
        self.assertEqual(scenarios[0]["suite"], "lemonade_bench")
        self.assertEqual(scenarios[0]["kind"], "scenario")
        self.assertTrue(scenarios[0]["default_selected"])
        self.assertFalse(scenarios[1]["default_selected"])
        self.assertIn("20 output tokens", scenarios[0]["description"])

    def test_frontend_keeps_scenarios_out_of_lm_eval_view_mode_filters(self):
        script = Path("static/app.js").read_text(encoding="utf-8")
        render_start = script.index("function renderTasks()")
        render_end = script.index("function setTaskLoading", render_start)
        render_tasks = " ".join(script[render_start:render_end].split())

        self.assertIn(
            "if (isLmEval) pruneSelectedTasksForViewMode(taskViewMode);",
            render_tasks,
        )
        self.assertIn(
            'isLmEval && taskViewMode === "leaves" && (task.kind || "task") !== "task"',
            render_tasks,
        )
        self.assertIn(
            'isLmEval && taskViewMode === "groups" && (task.kind || "task") === "task"',
            render_tasks,
        )

    def test_server_task_loader_can_return_lemonade_bench_scenarios(self):
        load_available_tasks = symbol("lm_eval_webui.server", "load_available_tasks")
        expected = [{"name": "chat-short", "suite": "lemonade_bench"}]

        with mock.patch(
            "lm_eval_webui.server.find_lemonade_bench_scenarios",
            return_value=expected,
        ):
            scenarios = load_available_tasks(suite="lemonade_bench")

        self.assertEqual(scenarios, expected)

    def test_results_parse_details_leaderboard_and_telemetry(self):
        extract_rows = symbol(
            "lm_eval_webui.lemonade_bench", "extract_lemonade_bench_result_rows"
        )
        extract_entries = symbol(
            "lm_eval_webui.lemonade_bench",
            "extract_lemonade_bench_leaderboard_entries",
        )
        summarize = symbol(
            "lm_eval_webui.lemonade_bench", "summarize_lemonade_bench_telemetry"
        )
        job = {
            "id": "job-1",
            "suite": "lemonade_bench",
            "model_id": "Qwen/Model-A",
            "status": "succeeded",
            "runtime_seconds": 12.5,
        }
        payload = self.result_payload()

        rows = extract_rows(job, payload)
        entries = extract_entries(job, payload)
        telemetry = summarize(payload)

        ttft_row = next(
            row
            for row in rows
            if row["task"] == "chat-short" and row["metric"] == "ttft_mean_ms"
        )
        self.assertEqual(ttft_row["model"], "Qwen/Model-A")
        self.assertEqual(ttft_row["value"], 100.0)
        self.assertEqual(ttft_row["samples"], 3)
        self.assertEqual(ttft_row["context_window"], 262144)
        self.assertEqual(
            ttft_row["configuration"],
            "vllm/rocm · 262,144 ctx · --kv-cache-memory-bytes 10G",
        )
        self.assertTrue(all(row["suite"] == "lemonade_bench" for row in rows))
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertEqual(entry["model"], "Qwen/Model-A")
        self.assertEqual(entry["average_ttft_ms"], 150.0)
        self.assertEqual(entry["average_tps"], 15.0)
        self.assertEqual(entry["overall_score"], 15.0)
        self.assertEqual(entry["failed_runs"], 1)
        self.assertEqual(entry["total_runs"], 5)
        self.assertEqual(entry["measured_duration_seconds"], 7.0)
        self.assertEqual(entry["vram_peak_gb"], 61.0)
        self.assertTrue(entry["partial"])
        self.assertEqual(telemetry["request_count"], 5)
        self.assertEqual(telemetry["failed_request_count"], 1)
        self.assertEqual(telemetry["ttft_s"], 0.15)
        self.assertEqual(telemetry["generation_tok_s"], 15.0)

    def test_job_runs_cli_without_pinning_and_rerun_preserves_options(self):
        JobManager = symbol("lm_eval_webui.jobs", "JobManager")
        commands = []
        pin_events = []

        def launcher(command, _env, _log_path):
            commands.append(command)
            output_path = Path(command[command.index("--output") + 1])
            output_path.write_text(
                json.dumps(self.result_payload(measurement_runs=2)),
                encoding="utf-8",
            )
            return 0

        def pin_model(*args):
            pin_events.append(("pin", args))
            return {}

        def unpin_model(*args):
            pin_events.append(("unpin", args))
            return {}

        with tempfile.TemporaryDirectory() as tmp:
            manager = JobManager(
                data_dir=Path(tmp) / "data",
                project_root=tmp,
                launcher=launcher,
                run_async=False,
                protect_models=True,
                model_pin_loader=pin_model,
                model_unpinner=unpin_model,
            )
            original = manager.create_jobs(
                {
                    "suite": "lemonade_bench",
                    "model_ids": ["Qwen/Model-A"],
                    "lemonade_model_ids": {"Qwen/Model-A": "user.Qwen/Model-A"},
                    "tasks": ["chat-short", "code-short"],
                    "openai_base_url": "https://llm.example.test/v1",
                    "bench_backends": ["rocm"],
                    "bench_context_sizes": [262144],
                    "bench_runs": 2,
                    "bench_warmup": 1,
                    "bench_timeout": 600,
                    "bench_memory_tracking": False,
                    "bench_reload_between_runs": False,
                    "bench_log_responses": True,
                    "max_concurrent_jobs": 4,
                }
            )[0]
            original_job = manager.get_job(original["id"])
            rows = manager.result_rows()
            entries = manager.leaderboard_entries()
            rerun = manager.rerun_jobs([original["id"]])[0]
            rerun_job = manager.get_job(rerun["id"])

        self.assertEqual(manager.max_concurrent_jobs, 1)
        self.assertEqual(original_job["suite"], "lemonade_bench")
        self.assertNotIn("benchmark_profile", original_job)
        self.assertEqual(
            original_job["lemonade_bench_options"]["server_model_id"],
            "user.Qwen/Model-A",
        )
        self.assertEqual(original_job["lemonade_bench_options"]["measurement_runs"], 2)
        self.assertEqual(
            original_job["lemonade_bench_options"]["context_sizes"], [262144]
        )
        self.assertEqual(
            original_job["result_files"],
            [str(Path(original_job["output_path"]) / "results.json")],
        )
        self.assertEqual(pin_events, [])
        self.assertEqual(commands[0][-1], "user.Qwen/Model-A")
        self.assertIn("--no-memory", commands[0])
        self.assertIn("--no-reload", commands[0])
        self.assertEqual(rows[0]["suite"], "lemonade_bench")
        self.assertEqual(entries[0]["suite"], "lemonade_bench")
        self.assertEqual(original_job["telemetry"]["request_count"], 3)
        self.assertEqual(rerun_job["suite"], "lemonade_bench")
        self.assertEqual(rerun_job["rerun_of"], original_job["id"])
        self.assertEqual(
            rerun_job["lemonade_bench_options"], original_job["lemonade_bench_options"]
        )
        self.assertEqual(commands[1][-1], "user.Qwen/Model-A")

    def test_job_uses_registered_backend_and_fails_when_all_requests_fail(self):
        JobManager = symbol("lm_eval_webui.jobs", "JobManager")
        commands = []

        def launcher(command, _env, _log_path):
            commands.append(command)
            output_path = Path(command[command.index("--output") + 1])
            output_path.write_text(
                json.dumps(
                    {
                        "models": [
                            {
                                "model": "user.Qwen/Model-A",
                                "config": {"measurement_runs": 2},
                                "results": [
                                    {
                                        "recipe": "llamacpp",
                                        "backend": "system",
                                        "ctx_size": 0,
                                        "scenarios": [
                                            {
                                                "name": "chat-short",
                                                "category": "chat",
                                                "all_runs_failed": True,
                                                "failed_runs": 2,
                                            }
                                        ],
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            return 0

        with tempfile.TemporaryDirectory() as tmp:
            manager = JobManager(
                data_dir=Path(tmp) / "data",
                project_root=tmp,
                launcher=launcher,
                run_async=False,
            )
            created = manager.create_jobs(
                {
                    "suite": "lemonade_bench",
                    "model_ids": ["Qwen/Model-A"],
                    "lemonade_model_ids": {"Qwen/Model-A": "user.Qwen/Model-A"},
                    "lemonade_model_backends": {"Qwen/Model-A": "system"},
                    "tasks": ["chat-short"],
                    "openai_base_url": "https://llm.example.test/v1",
                    "bench_backends": [],
                    "bench_runs": 2,
                }
            )[0]
            job = manager.get_job(created["id"])

        backend_index = commands[0].index("--backend")
        self.assertEqual(commands[0][backend_index + 1], "system")
        self.assertEqual(job["lemonade_bench_options"]["backends"], ["system"])
        self.assertEqual(job["lemonade_bench_options"]["timeout"], 1800)
        self.assertEqual(
            job["lemonade_bench_options"]["backend_source"],
            "model_configuration",
        )
        self.assertEqual(job["returncode"], 0)
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["telemetry"]["request_count"], 0)
        self.assertEqual(job["telemetry"]["failed_request_count"], 2)
        self.assertIn("without a successful request", job["error"])

    def test_progress_counts_discovered_backend_configurations(self):
        JobManager = symbol("lm_eval_webui.jobs", "JobManager")

        with tempfile.TemporaryDirectory() as tmp:
            manager = JobManager(
                data_dir=Path(tmp) / "data",
                project_root=tmp,
                run_async=False,
            )
            log_path = Path(tmp) / "bench.log"
            log_path.write_text(
                """=== [Model-A] llamacpp/rocm ===
  Scenario: chat-short (chat)
  Scenario: code-short (coding)
=== [Model-A] llamacpp/system ===
  Scenario: chat-short (chat)
""",
                encoding="utf-8",
            )
            progress = manager._lemonade_bench_progress(
                {
                    "status": "running",
                    "tasks": ["chat-short", "code-short"],
                    "log_path": str(log_path),
                    "lemonade_bench_options": {
                        "backends": [],
                        "context_sizes": [],
                    },
                }
            )

        self.assertIsNotNone(progress)
        self.assertEqual(progress["current"], 3)
        self.assertEqual(progress["completed"], 2)
        self.assertEqual(progress["total"], 4)
        self.assertEqual(progress["percent"], 75.0)
        self.assertEqual(progress["current_scenario"], "chat-short")


class SweMiniRunnerTests(unittest.TestCase):
    def test_swe_mini_defaults_bound_agent_context_and_provider_timeout(self):
        SweMiniRequest = symbol("lm_eval_webui.swe_mini", "SweMiniRequest")
        default_timeout = symbol(
            "lm_eval_webui.swe_mini", "DEFAULT_SWE_MINI_TIMEOUT_MINUTES"
        )
        default_provider_timeout = symbol(
            "lm_eval_webui.swe_mini",
            "DEFAULT_SWE_MINI_PROVIDER_TIMEOUT_MINUTES",
        )
        default_context = symbol(
            "lm_eval_webui.swe_mini", "DEFAULT_SWE_MINI_CONTEXT_WINDOW"
        )
        default_max_output = symbol(
            "lm_eval_webui.swe_mini", "DEFAULT_SWE_MINI_MAX_OUTPUT_TOKENS"
        )
        recipe_policy = symbol("lm_eval_webui.swe_mini", "SWE_MINI_RECIPE_POLICY")

        request = SweMiniRequest(
            model_id="Model-A",
            task_target="task.json",
            output_path="results",
        )

        self.assertEqual(default_timeout, 60)
        self.assertEqual(default_provider_timeout, 15)
        self.assertEqual(default_context, 65536)
        self.assertEqual(default_max_output, 16384)
        self.assertEqual(recipe_policy, "lemonade_unchanged")
        self.assertEqual(request.timeout_minutes, 60)
        self.assertEqual(request.provider_timeout_minutes, 15)
        self.assertEqual(request.context_window, 65536)
        self.assertEqual(request.max_output_tokens, 16384)

    def test_swe_mini_command_uses_repo_owned_wrapper_for_lemonade_judge(self):
        SweMiniRequest = symbol("lm_eval_webui.swe_mini", "SweMiniRequest")
        build_swe_mini_command = symbol(
            "lm_eval_webui.swe_mini", "build_swe_mini_command"
        )
        swe_mini_output_path = symbol("lm_eval_webui.swe_mini", "swe_mini_output_path")

        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "repo"
            pi_bench_dir = project_root / "third_party" / "pi-bench"
            scripts_dir = project_root / "scripts"
            pi_bench_dir.mkdir(parents=True)
            scripts_dir.mkdir()
            wrapper = scripts_dir / "run-swe-mini.sh"
            wrapper.write_text("#!/bin/sh\n", encoding="utf-8")
            output_path = swe_mini_output_path(
                "Gemma-4-26B-A4B-it-GGUF",
                "job123",
                "lemonade-swe",
                pi_bench_dir=pi_bench_dir,
            )

            command, env = build_swe_mini_command(
                SweMiniRequest(
                    model_id="Gemma-4-26B-A4B-it-GGUF",
                    task_target="tasks/verified-mini/django__django-12209.json",
                    output_path=str(output_path),
                    pi_bench_dir=str(pi_bench_dir),
                    project_root=str(project_root),
                    openai_base_url="https://llm.savagelands.net",
                    judge_model="lemonade/gpt-oss-120b-mxfp-GGUF",
                    model_tag="job123",
                    platform="lemonade-swe",
                    timeout_minutes=45,
                    pass_count=2,
                    context_window=131072,
                    max_output_tokens=16384,
                )
            )
            models_path = Path(env["PI_BENCH_MODELS_JSON"])
            models_path_exists = models_path.is_file()
            try:
                models_payload = json.loads(models_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                self.fail(f"invalid generated models.json: {exc}")
            model_ids = [
                model["id"]
                for model in models_payload["providers"]["lemonade"]["models"]
            ]

        self.assertEqual(
            command[:2],
            [
                str(wrapper),
                "tasks/verified-mini/django__django-12209.json",
            ],
        )
        self.assertIn("--provider", command)
        self.assertIn("lemonade", command)
        self.assertIn("--model", command)
        self.assertIn("Gemma-4-26B-A4B-it-GGUF", command)
        self.assertIn("--judge-model", command)
        self.assertIn("lemonade/gpt-oss-120b-mxfp-GGUF", command)
        self.assertIn("--model-tag", command)
        self.assertIn("job123", command)
        self.assertIn("--platform", command)
        self.assertIn("lemonade-swe", command)
        self.assertIn("--timeout", command)
        self.assertIn("45", command)
        self.assertIn("--pass", command)
        self.assertIn("2", command)
        self.assertIn("--context", command)
        self.assertIn("131072", command)
        self.assertEqual(env["SWE_MINI_OUTPUT_PATH"], str(output_path))
        self.assertEqual(env["LMEVAL_WEBUI_LAUNCH_CWD"], str(project_root))
        self.assertEqual(env["PI_BENCH_DIR"], str(pi_bench_dir))
        self.assertEqual(env["LMEVAL_WEBUI_SWE_PROVIDER_TIMEOUT_MS"], "900000")
        self.assertTrue(models_path_exists)
        self.assertEqual(
            model_ids,
            ["Gemma-4-26B-A4B-it-GGUF", "gpt-oss-120b-mxfp-GGUF"],
        )
        self.assertTrue(
            all(
                model["maxTokens"] == 16384
                for model in models_payload["providers"]["lemonade"]["models"]
            )
        )

    def test_swe_mini_lifecycle_env_switches_lemonade_candidate_and_judge(self):
        SweMiniRequest = symbol("lm_eval_webui.swe_mini", "SweMiniRequest")
        lifecycle_env = symbol("lm_eval_webui.swe_mini", "swe_mini_model_lifecycle_env")

        env = lifecycle_env(
            SweMiniRequest(
                model_id="Candidate-A",
                task_target="task.json",
                output_path="results",
                openai_base_url="https://llm.example.test/v1",
                provider="lemonade",
                judge_model="lemonade/Judge-B",
            )
        )

        self.assertEqual(
            env,
            {
                "LMEVAL_WEBUI_LEMONADE_BASE_URL": "https://llm.example.test",
                "LMEVAL_WEBUI_CANDIDATE_MODEL_ID": "Candidate-A",
                "LMEVAL_WEBUI_JUDGE_MODEL_ID": "Judge-B",
            },
        )

    def test_default_pi_bench_dir_is_repo_submodule(self):
        default_pi_bench_dir = symbol("lm_eval_webui.swe_mini", "default_pi_bench_dir")

        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "repo"
            expected = project_root / "third_party" / "pi-bench"

            self.assertEqual(default_pi_bench_dir(project_root), expected)

    def test_write_swe_mini_models_json_uses_selected_endpoint_and_model(self):
        write_swe_mini_models_json = symbol(
            "lm_eval_webui.swe_mini", "write_swe_mini_models_json"
        )

        with tempfile.TemporaryDirectory() as tmp:
            models_path = write_swe_mini_models_json(
                Path(tmp),
                base_url="https://llm.savagelands.net",
                model_id="Gemma-4-26B-A4B-it-GGUF",
                context_window=131072,
                max_output_tokens=16384,
                judge_model_id="gpt-oss-120b-mxfp-GGUF",
            )
            try:
                payload = json.loads(Path(models_path).read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                self.fail(f"invalid generated models.json: {exc}")

        lemonade = payload["providers"]["lemonade"]
        self.assertEqual(lemonade["baseUrl"], "https://llm.savagelands.net/v1")
        self.assertEqual(lemonade["api"], "openai-completions")
        self.assertEqual(lemonade["apiKey"], "lemonade")
        self.assertEqual(
            [model["id"] for model in lemonade["models"]],
            ["Gemma-4-26B-A4B-it-GGUF", "gpt-oss-120b-mxfp-GGUF"],
        )
        self.assertEqual(lemonade["models"][0]["contextWindow"], 131072)
        self.assertEqual(lemonade["models"][0]["maxTokens"], 16384)

    def test_default_swe_model_entry_uses_64k_context_and_16k_generation_cap(self):
        write_swe_mini_models_json = symbol(
            "lm_eval_webui.swe_mini", "write_swe_mini_models_json"
        )

        with tempfile.TemporaryDirectory() as tmp:
            models_path = write_swe_mini_models_json(
                tmp,
                base_url="https://llm.example.test",
                model_id="Model-A",
            )
            payload = json.loads(models_path.read_text(encoding="utf-8"))

        model = payload["providers"]["lemonade"]["models"][0]
        self.assertEqual(model["contextWindow"], 65536)
        self.assertEqual(model["maxTokens"], 16384)

    def test_swe_summary_is_rebuilt_from_all_partial_task_artifacts(self):
        write_swe_mini_summary = symbol(
            "lm_eval_webui.swe_mini", "write_swe_mini_summary"
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = {
                "task": "django__django-1",
                "judgeScore": 1,
                "durationMs": 1000,
            }
            second = {
                "task": "django__django-2",
                "judgeScore": 0,
                "durationMs": 3000,
                "infrastructureError": "inference_timeout",
            }
            (root / "results-django__django-1.json").write_text(
                json.dumps(first), encoding="utf-8"
            )
            (root / "results-django__django-2.json").write_text(
                json.dumps(second), encoding="utf-8"
            )
            (root / "results-django__django-2-attempt1.json").write_text(
                json.dumps(second), encoding="utf-8"
            )
            (root / "summary.json").write_text(
                json.dumps({"results": [second]}), encoding="utf-8"
            )

            summary_path = write_swe_mini_summary(root, scheduled_tasks=50)
            self.assertIsNotNone(summary_path)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(summary["completedTasks"], 2)
        self.assertEqual(summary["scheduledTasks"], 50)
        self.assertEqual(summary["passedTasks"], 1)
        self.assertEqual(summary["passRate"], 0.5)
        self.assertEqual(summary["coverage"], 0.04)
        self.assertTrue(summary["partial"])
        self.assertEqual(summary["infrastructureFailedTasks"], 1)
        self.assertEqual(summary["totalDurationMs"], 4000)
        self.assertEqual(len(summary["results"]), 2)

    def test_find_swe_mini_tasks_reads_verified_mini_task_files(self):
        find_swe_mini_tasks = symbol("lm_eval_webui.swe_mini", "find_swe_mini_tasks")

        with tempfile.TemporaryDirectory() as tmp:
            task_dir = Path(tmp) / "pi-bench" / "tasks" / "verified-mini"
            task_dir.mkdir(parents=True)
            (task_dir / "django__django-12209.json").write_text(
                json.dumps(
                    {
                        "id": "django__django-12209",
                        "repo": "django/django",
                        "prompt": "Fix the queryset bug.",
                    }
                ),
                encoding="utf-8",
            )

            tasks = find_swe_mini_tasks(task_dir.parents[1])

        self.assertEqual(tasks[0]["name"], "django__django-12209")
        self.assertEqual(tasks[0]["repo"], "django/django")
        self.assertEqual(tasks[0]["suite"], "swe_mini")
        self.assertEqual(tasks[0]["compatibility"], "compatible")
        self.assertEqual(tasks[0]["kind"], "task")

    def test_server_task_loader_can_return_swe_mini_tasks(self):
        load_available_tasks = symbol("lm_eval_webui.server", "load_available_tasks")

        with tempfile.TemporaryDirectory() as tmp:
            task_dir = Path(tmp) / "pi-bench" / "tasks" / "verified-mini"
            task_dir.mkdir(parents=True)
            (task_dir / "django__django-12209.json").write_text(
                json.dumps(
                    {
                        "id": "django__django-12209",
                        "repo": "django/django",
                        "prompt": "Fix the queryset bug.",
                    }
                ),
                encoding="utf-8",
            )

            tasks = load_available_tasks(
                suite="swe_mini", pi_bench_dir=task_dir.parents[1]
            )

        self.assertEqual([task["name"] for task in tasks], ["django__django-12209"])
        self.assertEqual(tasks[0]["suite"], "swe_mini")

    def test_swe_mini_results_parse_rows_and_leaderboard(self):
        extract_swe_mini_result_rows = symbol(
            "lm_eval_webui.swe_mini", "extract_swe_mini_result_rows"
        )
        extract_swe_mini_leaderboard_entry = symbol(
            "lm_eval_webui.swe_mini", "extract_swe_mini_leaderboard_entry"
        )
        summary = {
            "totalTasks": 2,
            "passedTasks": 1,
            "passRate": 0.5,
            "averageDurationMs": 1500,
            "results": [
                {
                    "task": "django__django-12209",
                    "durationMs": 1000,
                    "judgeScore": 1,
                    "judgeRationale": "fixed",
                    "succeededAtAttempt": 1,
                    "attempts": [{"judgeScore": 0}, {"judgeScore": 1}],
                },
                {
                    "task": "sphinx-doc__sphinx-10435",
                    "durationMs": 2000,
                    "judgeScore": 0,
                    "judgeRationale": "missed",
                },
            ],
        }
        job = {
            "id": "job-1",
            "suite": "swe_mini",
            "model_id": "Model-A",
            "status": "succeeded",
            "runtime_seconds": 12.5,
            "provider_backend": "rocm",
            "swe_options": {
                "judge_model": "lemonade/gpt-oss-120b-mxfp-GGUF",
                "platform": "lemonade-swe",
                "pass_count": 2,
            },
        }

        rows = extract_swe_mini_result_rows(job, summary)
        entry = extract_swe_mini_leaderboard_entry(job, summary)

        self.assertEqual(
            [(row["task"], row["metric"], row["value"]) for row in rows],
            [
                ("django__django-12209", "judge_score", 1.0),
                ("django__django-12209", "duration_seconds", 1.0),
                ("sphinx-doc__sphinx-10435", "judge_score", 0.0),
                ("sphinx-doc__sphinx-10435", "duration_seconds", 2.0),
            ],
        )
        self.assertTrue(all(row["suite"] == "swe_mini" for row in rows))
        self.assertTrue(all(row["runtime_seconds"] == 12.5 for row in rows))
        self.assertEqual(entry["suite"], "swe_mini")
        self.assertEqual(entry["runtime_seconds"], 12.5)
        self.assertEqual(entry["overall_score"], 50.0)
        self.assertEqual(entry["total_tasks"], 2)
        self.assertEqual(entry["passed_tasks"], 1)
        self.assertEqual(entry["judge_model"], "lemonade/gpt-oss-120b-mxfp-GGUF")
        self.assertEqual(entry["task_scores"][0]["attempts"], 2)


class LmEvalRunnerTests(unittest.TestCase):
    def test_acp_duplicate_filter_registration_is_ignored(self):
        allow_duplicate_acp_filter_registration = symbol(
            "lm_eval_webui.lm_eval_runner",
            "allow_duplicate_acp_filter_registration",
        )
        calls = []

        class FakeRegistryModule:
            def register_filter(self, name):
                def decorate(cls):
                    calls.append((name, cls.__name__))
                    if len(calls) > 1:
                        raise ValueError(
                            "'filter' alias 'ACP_grammar_filter' already registered"
                        )
                    return cls

                return decorate

        registry_module = FakeRegistryModule()
        allow_duplicate_acp_filter_registration(registry_module)

        @registry_module.register_filter("ACP_grammar_filter")
        class FirstFilter:
            pass

        @registry_module.register_filter("ACP_grammar_filter")
        class SecondFilter:
            pass

        self.assertEqual(SecondFilter.__name__, "SecondFilter")
        self.assertEqual(
            calls,
            [
                ("ACP_grammar_filter", "FirstFilter"),
                ("ACP_grammar_filter", "SecondFilter"),
            ],
        )

    def test_non_acp_duplicate_filter_registration_still_raises(self):
        allow_duplicate_acp_filter_registration = symbol(
            "lm_eval_webui.lm_eval_runner",
            "allow_duplicate_acp_filter_registration",
        )

        class FakeRegistryModule:
            def register_filter(self, _name):
                def decorate(_cls):
                    raise ValueError("some other filter already registered")

                return decorate

        registry_module = FakeRegistryModule()
        allow_duplicate_acp_filter_registration(registry_module)

        with self.assertRaisesRegex(ValueError, "some other filter"):

            @registry_module.register_filter("other_filter")
            class OtherFilter:
                pass

    def test_transient_huggingface_gateway_timeout_is_retried(self):
        run_cli_with_hf_retries = symbol(
            "lm_eval_webui.lm_eval_runner", "run_cli_with_hf_retries"
        )
        attempts = []
        sleeps = []

        class Response:
            status_code = 504
            url = "https://huggingface.co/api/datasets/SaylorTwift/bbh/tree/main"

        class HfError(OSError):
            response = Response()

        def cli_evaluate():
            attempts.append(1)
            if len(attempts) == 1:
                raise HfError("504 Server Error: Gateway Time-out")
            return 0

        result = run_cli_with_hf_retries(
            cli_evaluate,
            retries=2,
            initial_delay=0,
            sleep=sleeps.append,
            stderr=types.SimpleNamespace(
                write=lambda _message: None, flush=lambda: None
            ),
        )

        self.assertEqual(result, 0)
        self.assertEqual(len(attempts), 2)
        self.assertEqual(sleeps, [0])

    def test_huggingface_dataset_cache_config_miss_is_retried(self):
        run_cli_with_hf_retries = symbol(
            "lm_eval_webui.lm_eval_runner", "run_cli_with_hf_retries"
        )
        attempts = []
        sleeps = []

        def cli_evaluate():
            attempts.append(1)
            if len(attempts) == 1:
                raise ValueError(
                    "Couldn't find cache for fxmarty/mmlu-redux-2.0-ok "
                    "for config 'high_school_microeconomics'\n"
                    "Available configs in the cache: ['high_school_mathematics']"
                )
            return 0

        result = run_cli_with_hf_retries(
            cli_evaluate,
            retries=2,
            initial_delay=0,
            sleep=sleeps.append,
            stderr=types.SimpleNamespace(
                write=lambda _message: None, flush=lambda: None
            ),
        )

        self.assertEqual(result, 0)
        self.assertEqual(len(attempts), 2)
        self.assertEqual(sleeps, [0])

    def test_non_huggingface_errors_are_not_retried(self):
        run_cli_with_hf_retries = symbol(
            "lm_eval_webui.lm_eval_runner", "run_cli_with_hf_retries"
        )
        attempts = []

        def cli_evaluate():
            attempts.append(1)
            raise RuntimeError("model endpoint failed")

        with self.assertRaisesRegex(RuntimeError, "model endpoint"):
            run_cli_with_hf_retries(
                cli_evaluate,
                retries=2,
                initial_delay=0,
                sleep=lambda _delay: None,
            )

        self.assertEqual(len(attempts), 1)

    def test_corrupt_huggingface_dataset_info_cache_is_removed_and_retried(self):
        run_cli_with_hf_retries = symbol(
            "lm_eval_webui.lm_eval_runner", "run_cli_with_hf_retries"
        )
        attempts = []
        sleeps = []

        with tempfile.TemporaryDirectory() as tmp:
            cache_root = Path(tmp) / "datasets"
            corrupt_dir = cache_root / "SaylorTwift___bbh" / "tracking" / "1.0.0"
            corrupt_dir.mkdir(parents=True)
            (corrupt_dir / "dataset_info.json").write_text("", encoding="utf-8")

            def cli_evaluate():
                attempts.append(1)
                if len(attempts) == 1:
                    raise json.JSONDecodeError("Expecting value", "", 0)
                return 0

            result = run_cli_with_hf_retries(
                cli_evaluate,
                retries=2,
                initial_delay=0,
                sleep=sleeps.append,
                cache_roots=[cache_root],
                stderr=types.SimpleNamespace(
                    write=lambda _message: None, flush=lambda: None
                ),
            )

            self.assertEqual(result, 0)
            self.assertEqual(len(attempts), 2)
            self.assertEqual(sleeps, [0])
            self.assertFalse(corrupt_dir.exists())

    def test_json_decode_errors_without_corrupt_hf_cache_are_not_retried(self):
        run_cli_with_hf_retries = symbol(
            "lm_eval_webui.lm_eval_runner", "run_cli_with_hf_retries"
        )
        attempts = []

        with tempfile.TemporaryDirectory() as tmp:
            cache_root = Path(tmp) / "datasets"
            cache_root.mkdir()

            def cli_evaluate():
                attempts.append(1)
                raise json.JSONDecodeError("Expecting value", "", 0)

            with self.assertRaises(json.JSONDecodeError):
                run_cli_with_hf_retries(
                    cli_evaluate,
                    retries=2,
                    initial_delay=0,
                    sleep=lambda _delay: None,
                    cache_roots=[cache_root],
                )

        self.assertEqual(len(attempts), 1)


class LemonadeModelTests(unittest.TestCase):
    @staticmethod
    def completion():
        completion_type = symbol(
            "lm_eval_webui.lemonade_model", "OpenAICompatibleChatCompletion"
        )
        return object.__new__(completion_type)

    def test_add_runtime_options_adds_selected_llamacpp_backend(self):
        add_runtime_options = symbol(
            "lm_eval_webui.lemonade_model", "add_runtime_options"
        )

        payload: dict[str, Any] = {"model": "Model-A"}
        add_runtime_options(payload, llamacpp_backend="rocm")

        recipe_options = payload["recipe_options"]
        self.assertEqual(payload["llamacpp_backend"], "rocm")
        self.assertIsInstance(recipe_options, dict)
        self.assertEqual(recipe_options["llamacpp_backend"], "rocm")

    def test_add_runtime_options_omits_auto_llamacpp_backend(self):
        add_runtime_options = symbol(
            "lm_eval_webui.lemonade_model", "add_runtime_options"
        )

        payload: dict[str, Any] = {"model": "Model-A"}
        add_runtime_options(payload, llamacpp_backend="")

        self.assertNotIn("llamacpp_backend", payload)

    def test_streaming_usage_preserves_options_and_requests_final_usage(self):
        enable_streaming_usage = symbol(
            "lm_eval_webui.lemonade_model", "enable_streaming_usage"
        )
        payload = {"stream_options": {"continuous_usage_stats": True}}

        enabled = enable_streaming_usage(payload)

        self.assertIs(enabled, payload)
        self.assertTrue(payload["stream"])
        self.assertEqual(
            payload["stream_options"],
            {"continuous_usage_stats": True, "include_usage": True},
        )

    def test_stream_model_call_records_telemetry_before_parsing(self):
        completion = self.completion()
        completion._stream_responses = True
        completion._telemetry_path = None
        completion._seed = 0
        completion._llamacpp_backend = None
        completion.base_url = "http://example.test/v1/chat/completions"
        completion.verify_certificate = True
        completion.timeout = 30
        completion.__dict__["header"] = {}
        completion.__dict__["eos_string"] = None
        completion.create_message = lambda messages: messages
        completion._create_payload = lambda *_args, **_kwargs: {"model": "Model-A"}
        posted_payloads = []

        class Response:
            def raise_for_status(self):
                return None

            def iter_lines(self, decode_unicode=False):
                lines = [
                    'data: {"model":"Model-A","choices":[{"index":0,"delta":{"reasoning":"work"}}]}',
                    'data: {"choices":[{"index":0,"delta":{"content":"42"},"finish_reason":"stop"}]}',
                    'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}',
                    "data: [DONE]",
                ]
                return lines if decode_unicode else [line.encode() for line in lines]

        def post(_url, *, json, **_kwargs):
            posted_payloads.append(json)
            return Response()

        with tempfile.TemporaryDirectory() as tmp:
            telemetry_path = Path(tmp) / "vllm.jsonl"
            completion._telemetry_path = str(telemetry_path)
            with mock.patch(
                "lm_eval_webui.lemonade_model.importlib.import_module",
                return_value=types.SimpleNamespace(post=post),
            ):
                output = completion.model_call(
                    [{"role": "user", "content": "question"}], gen_kwargs={}
                )

            self.assertIsInstance(output, dict)
            self.assertEqual(
                len(telemetry_path.read_text(encoding="utf-8").splitlines()), 1
            )
            self.assertEqual(completion.parse_generations(output), ["42"])
            self.assertEqual(
                len(telemetry_path.read_text(encoding="utf-8").splitlines()), 1
            )

        self.assertTrue(posted_payloads[0]["stream"])
        self.assertEqual(posted_payloads[0]["stream_options"], {"include_usage": True})

    def test_async_stream_model_call_uses_idle_timeout_and_records_telemetry(self):
        aggregate_telemetry_file = symbol(
            "lm_eval_webui.telemetry", "aggregate_telemetry_file"
        )
        completion = self.completion()
        completion._stream_responses = True
        completion._seed = 0
        completion.timeout = 7200
        completion._llamacpp_backend = None
        completion.base_url = "http://example.test/v1/chat/completions"
        completion.__dict__["header"] = {}
        completion.create_message = lambda messages: messages
        completion._create_payload = lambda *_args, **_kwargs: {"model": "Model-A"}

        class Content:
            def __init__(self):
                self.lines = iter(
                    [
                        b'data: {"model":"Model-A","choices":[{"index":0,"delta":{"reasoning":"work"}}]}\n',
                        b'data: {"choices":[{"index":0,"delta":{"content":"42"},"finish_reason":"stop"}]}\n',
                        b'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n',
                        b"data: [DONE]\n",
                    ]
                )

            async def readline(self):
                await asyncio.sleep(0.001)
                return next(self.lines, b"")

        class Response:
            def __init__(self):
                self.content = Content()

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            def raise_for_status(self):
                return None

        class Session:
            def __init__(self):
                self.payloads = []
                self.request_options = []

            def post(self, _url, *, json, **kwargs):
                self.payloads.append(json)
                self.request_options.append(kwargs)
                return Response()

        class Semaphore:
            released = False

            async def acquire(self):
                return True

            def release(self):
                self.released = True

        with tempfile.TemporaryDirectory() as tmp:
            telemetry_path = Path(tmp) / "vllm-async.jsonl"
            completion._telemetry_path = str(telemetry_path)
            session = Session()
            semaphore = Semaphore()

            answers = asyncio.run(
                completion.amodel_call(
                    session,
                    semaphore,
                    [[{"role": "user", "content": "question"}]],
                    gen_kwargs={},
                )
            )
            telemetry = aggregate_telemetry_file(telemetry_path)
            event = json.loads(telemetry_path.read_text(encoding="utf-8"))

        self.assertEqual(answers, ["42"])
        self.assertTrue(semaphore.released)
        self.assertEqual(session.payloads[0]["stream_options"], {"include_usage": True})
        request_timeout = session.request_options[0]["timeout"]
        self.assertIsNone(request_timeout.total)
        self.assertEqual(request_timeout.connect, 60.0)
        self.assertEqual(request_timeout.sock_connect, 60.0)
        self.assertEqual(request_timeout.sock_read, 7200.0)
        self.assertEqual(event["timings"]["generation_timing_source"], "client_stream")
        self.assertEqual(event["timings"]["predicted_n"], 2)
        self.assertGreater(event["timings"]["predicted_ms"], 0)
        self.assertEqual(telemetry["request_count"], 1)
        self.assertEqual(telemetry["generated_tokens"], 2)
        self.assertGreater(telemetry["generation_tok_s"], 0)

    def test_create_payload_defers_stops_and_floors_generation_budget(self):
        completion_type = symbol(
            "lm_eval_webui.lemonade_model", "OpenAICompatibleChatCompletion"
        )
        parent_type = completion_type.__mro__[1]
        stop_context = symbol("lm_eval_webui.lemonade_model", "_CURRENT_STOP_SEQUENCES")
        context_token = stop_context.set(stop_context.get())

        def parent_payload(
            _self, _messages, generate=False, gen_kwargs=None, **_kwargs
        ):
            generation_kwargs = gen_kwargs or {}
            return {
                "max_tokens": generation_kwargs.get("max_gen_toks", 7),
                "stop": generation_kwargs.get("until", ["parent-default"]),
            }

        try:
            with mock.patch.object(
                parent_type, "_create_payload", parent_payload, create=True
            ):
                completion = self.completion()
                completion._max_gen_toks = 32768
                completion._llamacpp_backend = "rocm"
                payload = completion._create_payload(
                    [{"role": "user", "content": "question"}],
                    generate=True,
                    gen_kwargs={"max_gen_toks": 256, "until": ["\n"]},
                )

            self.assertNotIn("stop", payload)
            self.assertEqual(payload["max_tokens"], 32768)
            self.assertEqual(payload["llamacpp_backend"], "rocm")
            self.assertEqual(
                self.completion().parse_generations(
                    {"choices": [{"index": 0, "message": {"content": "answer\nextra"}}]}
                ),
                ["answer"],
            )
        finally:
            stop_context.reset(context_token)

    def test_parse_generations_preserves_empty_choice_responses(self):
        generations = self.completion().parse_generations(
            [
                {"model": "Model-A", "timings": {"predicted_n": 0}, "choices": []},
                {
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "ok"}}
                    ]
                },
            ]
        )

        self.assertEqual(generations, ["", "ok"])

    def test_parse_generations_never_grades_unfinished_reasoning(self):
        for reasoning_field in ("reasoning", "reasoning_content", "analysis"):
            with self.subTest(reasoning_field=reasoning_field):
                output: dict[str, Any] = {
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "length",
                            "message": {
                                "role": "assistant",
                                "content": "",
                                reasoning_field: "The answer may be 42.",
                            },
                        }
                    ]
                }
                generations = self.completion().parse_generations(output)

                self.assertEqual(generations, [""])
                self.assertTrue(output["response"]["has_reasoning"])
                self.assertFalse(output["response"]["has_final_content"])
                self.assertTrue(output["response"]["hit_generation_limit"])

    def test_parse_generations_extracts_final_answer_and_applies_task_stop(self):
        completion = self.completion()

        generations = completion.parse_generations(
            {
                "_lm_eval_stop_sequences": ["\n"],
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "<think>work it out</think>\n\n42\nextra",
                        },
                    }
                ],
            }
        )

        self.assertEqual(generations, ["42"])
        self.assertEqual(
            completion.parse_generations(
                {
                    "_lm_eval_stop_sequences": ["\n"],
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "reasoning": "work it out",
                                "content": "\n42\nextra",
                            },
                        }
                    ],
                }
            ),
            ["42"],
        )

    def test_parse_generations_keeps_concurrent_request_stops_separate(self):
        completion = self.completion()

        generations = completion.parse_generations(
            [
                {
                    "_lm_eval_stop_sequences": ["\n"],
                    "choices": [{"index": 0, "message": {"content": "first\nextra"}}],
                },
                {
                    "_lm_eval_stop_sequences": ["<END>"],
                    "choices": [
                        {"index": 0, "message": {"content": "second<END>extra"}}
                    ],
                },
            ]
        )

        self.assertEqual(generations, ["first", "second"])

    def test_generation_payload_defers_stops_and_raises_task_token_caps(self):
        prepare_generation_payload = symbol(
            "lm_eval_webui.lemonade_model", "prepare_generation_payload"
        )
        for token_key in ("max_tokens", "max_completion_tokens"):
            with self.subTest(token_key=token_key):
                payload = {"model": "Model-A", token_key: 7, "stop": ["\n"]}

                prepared = prepare_generation_payload(payload, 32768)

                self.assertNotIn("stop", prepared)
                self.assertEqual(prepared[token_key], 32768)

    def test_stream_response_json_records_client_ttft(self):
        stream_response_json = symbol(
            "lm_eval_webui.lemonade_model", "stream_response_json"
        )

        class Response:
            ok = True
            text = ""

            def raise_for_status(self):
                return None

            def iter_lines(self, decode_unicode=False):
                lines = [
                    'data: {"model":"Model-A","choices":[{"index":0,"delta":{"role":"assistant"}}]}',
                    'data: {"choices":[{"index":0,"delta":{"content":"red"}}],"timings":{"predicted_n":1,"predicted_ms":10}}',
                    'data: {"choices":[{"index":0,"delta":{"content":" blue"}}],"usage":{"completion_tokens":2}}',
                    "data: [DONE]",
                ]
                return (
                    lines
                    if decode_unicode
                    else [line.encode("utf-8") for line in lines]
                )

        times = iter([101.0, 102.0, 103.0, 104.0, 105.0])

        output = stream_response_json(
            Response(), started=100.0, clock=lambda: next(times)
        )

        self.assertEqual(output["model"], "Model-A")
        self.assertEqual(output["choices"][0]["message"]["content"], "red blue")
        self.assertEqual(output["timings"]["request_elapsed_s"], 5.0)
        self.assertEqual(output["timings"]["generation_elapsed_s"], 2.0)
        self.assertEqual(output["timings"]["time_to_headers_s"], 1.0)
        self.assertEqual(output["timings"]["time_to_first_event_s"], 2.0)
        self.assertEqual(output["timings"]["ttft_s"], 3.0)
        self.assertEqual(output["timings"]["predicted_n"], 1)
        self.assertEqual(output["usage"], {"completion_tokens": 2})
        self.assertTrue(output["response"]["has_final_content"])
        self.assertFalse(output["response"]["has_reasoning"])

    def test_stream_response_json_supports_reasoning_from_all_provider_schemas(self):
        stream_response_json = symbol(
            "lm_eval_webui.lemonade_model", "stream_response_json"
        )
        provider_fields = {
            "vllm-0.20": "reasoning",
            "llamacpp-and-deepseek": "reasoning_content",
            "analysis-alias": "analysis",
        }

        for provider, reasoning_field in provider_fields.items():
            with self.subTest(provider=provider):

                class Response:
                    def __init__(self, model_name, field_name):
                        self.model_name = model_name
                        self.field_name = field_name

                    def iter_lines(self, decode_unicode=False):
                        chunks = [
                            {
                                "model": self.model_name,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "role": "assistant",
                                            self.field_name: "work",
                                        },
                                    }
                                ],
                            },
                            {
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {self.field_name: " more"},
                                    }
                                ]
                            },
                            {"choices": [{"index": 0, "delta": {"content": "42"}}]},
                        ]
                        lines = [f"data: {json.dumps(chunk)}" for chunk in chunks]
                        lines.append("data: [DONE]")
                        return (
                            lines
                            if decode_unicode
                            else [line.encode("utf-8") for line in lines]
                        )

                times = iter([1.0, 2.0, 3.0, 4.0, 5.0])
                output = stream_response_json(
                    Response(provider, reasoning_field),
                    started=0.0,
                    clock=lambda iterator=times: next(iterator),
                )

                message = output["choices"][0]["message"]
                self.assertEqual(message["reasoning"], "work more")
                self.assertEqual(message["content"], "42")
                self.assertEqual(output["timings"]["ttft_source"], "first_reasoning")
                self.assertTrue(output["response"]["has_reasoning"])
                self.assertTrue(output["response"]["has_final_content"])
                self.assertEqual(self.completion().parse_generations(output), ["42"])

    def test_telemetry_aggregates_final_content_and_reasoning_coverage(self):
        aggregate_telemetry_events = symbol(
            "lm_eval_webui.telemetry", "aggregate_telemetry_events"
        )

        aggregate = aggregate_telemetry_events(
            [
                {
                    "timings": {"ttft_s": 1.0, "generation_elapsed_s": 1.0},
                    "usage": {"completion_tokens": 10, "prompt_tokens": 5},
                    "response": {
                        "has_final_content": True,
                        "has_reasoning": True,
                    },
                },
                {
                    "timings": {"ttft_s": 2.0, "generation_elapsed_s": 2.0},
                    "usage": {"completion_tokens": 20, "prompt_tokens": 7},
                    "response": {
                        "has_final_content": False,
                        "has_reasoning": True,
                        "hit_generation_limit": True,
                    },
                },
            ]
        )

        self.assertEqual(aggregate["response_metadata_count"], 2)
        self.assertEqual(aggregate["final_content_response_count"], 1)
        self.assertEqual(aggregate["reasoning_response_count"], 2)
        self.assertEqual(aggregate["empty_response_count"], 1)
        self.assertEqual(aggregate["generation_limited_response_count"], 1)
        self.assertEqual(aggregate["generated_tokens"], 30)
        self.assertEqual(aggregate["generation_tok_s"], 10.0)
        self.assertEqual(aggregate["prompt_tokens"], 12)

    def test_normalize_models_extracts_vllm_context_from_recipe_options(self):
        normalize_models = symbol("lm_eval_webui.lemonade", "normalize_models")

        models = normalize_models(
            {
                "data": [
                    {
                        "id": "Model-vLLM",
                        "downloaded": True,
                        "recipe": "vllm",
                        "recipe_options": {"ctx_size": 131072},
                    }
                ]
            }
        )

        self.assertEqual(models[0]["context_window"], 131072)
        self.assertEqual(models[0]["runtime_backend"], "vllm")

    def test_normalize_models_extracts_llamacpp_runtime_backend(self):
        normalize_models = symbol("lm_eval_webui.lemonade", "normalize_models")

        models = normalize_models(
            {
                "data": [
                    {
                        "id": "Model-A",
                        "downloaded": True,
                        "recipe": "llamacpp",
                        "recipe_options": {"llamacpp_backend": "vulkan"},
                    }
                ]
            }
        )

        self.assertEqual(models[0]["llamacpp_backend"], "vulkan")
        self.assertEqual(models[0]["runtime_backend"], "vulkan")

    def test_normalize_models_reports_system_for_llamacpp_without_explicit_backend(
        self,
    ):
        normalize_models = symbol("lm_eval_webui.lemonade", "normalize_models")

        models = normalize_models(
            {
                "data": [
                    {
                        "id": "Model-A",
                        "downloaded": True,
                        "recipe": "llamacpp",
                    }
                ]
            }
        )

        self.assertEqual(models[0]["llamacpp_backend"], "system")
        self.assertEqual(models[0]["runtime_backend"], "system")

    def test_health_metadata_extracts_llamacpp_runtime_backend(self):
        loaded_model_metadata_from_health = symbol(
            "lm_eval_webui.lemonade", "loaded_model_metadata_from_health"
        )

        metadata = loaded_model_metadata_from_health(
            {
                "all_models_loaded": [
                    {
                        "model_name": "Gemma-4-31B-it-GGUF",
                        "checkpoint": "unsloth/gemma-4-31B-it-GGUF:Q4_K_M",
                        "device": "gpu",
                        "max_context_window": 131072,
                        "recipe": "llamacpp",
                        "recipe_options": {"llamacpp_backend": "rocm"},
                    }
                ]
            },
            "Gemma-4-31B-it-GGUF",
        )

        self.assertEqual(metadata["recipe"], "llamacpp")
        self.assertEqual(metadata["llamacpp_backend"], "rocm")
        self.assertEqual(metadata["runtime_backend"], "rocm")
        self.assertEqual(metadata["context_window"], 131072)
        self.assertEqual(metadata["device"], "gpu")

    def test_health_metadata_reports_system_for_llamacpp_without_explicit_backend(self):
        loaded_model_metadata_from_health = symbol(
            "lm_eval_webui.lemonade", "loaded_model_metadata_from_health"
        )

        metadata = loaded_model_metadata_from_health(
            {
                "all_models_loaded": [
                    {
                        "model_name": "Gemma-4-31B-it-GGUF",
                        "checkpoint": "unsloth/gemma-4-31B-it-GGUF:Q4_K_M",
                        "recipe": "llamacpp",
                    }
                ]
            },
            "Gemma-4-31B-it-GGUF",
        )

        self.assertEqual(metadata["recipe"], "llamacpp")
        self.assertEqual(metadata["llamacpp_backend"], "system")
        self.assertEqual(metadata["runtime_backend"], "system")


class SweMiniWrapperScriptTests(unittest.TestCase):
    def test_wrapper_consumes_pass_flag_instead_of_forwarding_to_pi_bench(self):
        script = Path("scripts/run-swe-mini.sh").read_text(encoding="utf-8")
        pass_case = script[
            script.index("--pass)") : script.index("shift 2", script.index("--pass)"))
        ]

        self.assertIn('PASS_COUNT="$2"', pass_case)
        self.assertNotIn("EXTRA_ARGS", pass_case)

    def test_wrapper_defaults_agent_timeout_to_one_hour(self):
        script = Path("scripts/run-swe-mini.sh").read_text(encoding="utf-8")

        self.assertIn("DEFAULT_SWE_TIMEOUT_MINUTES=60", script)
        self.assertIn('EXTRA_ARGS+=(--timeout "$DEFAULT_SWE_TIMEOUT_MINUTES")', script)

    def test_wrapper_fails_fast_when_docker_run_produces_no_result(self):
        script = Path("scripts/run-swe-mini.sh").read_text(encoding="utf-8")

        self.assertIn("No result file produced", script)
        self.assertIn('exit "$EXIT_CODE"', script)

    def test_wrapper_scores_lifecycle_race_zero_and_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime"
            (runtime / "src").mkdir(parents=True)
            (runtime / "tasks").mkdir()
            (runtime / "src" / "index.ts").write_text(
                Path("third_party/pi-bench/src/index.ts").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            task_id = "sphinx-doc__sphinx-race"
            task_file = runtime / "tasks" / "sphinx-doc__sphinx-race.json"
            task_file.write_text(
                json.dumps({"id": task_id, "repo": "sphinx-doc/sphinx"}),
                encoding="utf-8",
            )
            bin_dir = root / "bin"
            bin_dir.mkdir()
            fake_docker = bin_dir / "docker"
            fake_docker.write_text(
                """#!/usr/bin/env bash
if [ "${1:-}" = "volume" ]; then
    exit 0
fi
printf '%s\\n' "$@"
echo 'error: Lemonade model lifecycle request failed (409): slots_pinned_error'
exit 1
""",
                encoding="utf-8",
            )
            fake_docker.chmod(0o755)
            output_dir = runtime / "results"
            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{bin_dir}:{env['PATH']}",
                    "PI_BENCH_DIR": str(runtime),
                    "PI_BENCH_RUN_DIR": str(runtime),
                    "PI_BENCH_SKIP_CHOWN": "1",
                    "SWE_MINI_OUTPUT_PATH": str(output_dir),
                    "LMEVAL_WEBUI_JOB_ID": "race-test",
                }
            )

            completed = subprocess.run(
                ["bash", "scripts/run-swe-mini.sh", str(task_file)],
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("scoring this attempt zero and continuing", completed.stdout)
        self.assertIn("'--timeout' '60'", completed.stdout)
        self.assertEqual(summary["totalTasks"], 1)
        self.assertEqual(summary["passedTasks"], 0)
        self.assertEqual(summary["results"][0]["judgeScore"], 0)
        self.assertEqual(
            summary["results"][0]["infrastructureError"],
            "lemonade_model_lifecycle",
        )

    def test_wrapper_scores_provider_timeout_zero_and_continues(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runtime = root / "runtime"
            (runtime / "src").mkdir(parents=True)
            (runtime / "tasks").mkdir()
            index_path = runtime / "src" / "index.ts"
            index_path.write_text(
                Path("third_party/pi-bench/src/index.ts").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            task_id = "django__django-timeout"
            task_file = runtime / "tasks" / f"{task_id}.json"
            task_file.write_text(
                json.dumps({"id": task_id, "repo": "django/django"}),
                encoding="utf-8",
            )
            bin_dir = root / "bin"
            bin_dir.mkdir()
            fake_docker = bin_dir / "docker"
            fake_docker.write_text(
                """#!/usr/bin/env bash
if [ "${1:-}" = "volume" ]; then
    exit 0
fi
echo 'Error: Inference backend is unreachable: The operation timed out.'
exit 2
""",
                encoding="utf-8",
            )
            fake_docker.chmod(0o755)
            output_dir = runtime / "results"
            env = os.environ.copy()
            env.update(
                {
                    "PATH": f"{bin_dir}:{env['PATH']}",
                    "PI_BENCH_DIR": str(runtime),
                    "PI_BENCH_RUN_DIR": str(runtime),
                    "PI_BENCH_SKIP_CHOWN": "1",
                    "SWE_MINI_OUTPUT_PATH": str(output_dir),
                    "LMEVAL_WEBUI_JOB_ID": "timeout-test",
                    "LMEVAL_WEBUI_SWE_PROVIDER_TIMEOUT_MS": "900000",
                }
            )

            completed = subprocess.run(
                ["bash", "scripts/run-swe-mini.sh", str(task_file)],
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            patched_index = index_path.read_text(encoding="utf-8")

        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertIn("scoring this attempt zero and continuing", completed.stdout)
        self.assertEqual(summary["totalTasks"], 1)
        self.assertEqual(summary["passedTasks"], 0)
        self.assertEqual(summary["infrastructureFailedTasks"], 1)
        self.assertEqual(
            summary["results"][0]["infrastructureError"], "inference_timeout"
        )
        self.assertIn("LMEVAL_WEBUI_EXPLICIT_PROVIDER_TIMEOUT_V1", patched_index)
        self.assertIn("SettingsManager.inMemory", patched_index)
        self.assertIn("enabled: false", patched_index)
        self.assertIn("maxRetries: 0", patched_index)

    def test_wrapper_labels_and_cleans_up_cancelled_containers(self):
        script = Path("scripts/run-swe-mini.sh").read_text(encoding="utf-8")

        self.assertIn("LMEVAL_WEBUI_JOB_ID", script)
        self.assertIn("lm-eval-webui.job-id=$JOB_ID", script)
        self.assertIn("trap 'cancel_run 143' TERM", script)
        self.assertIn('docker rm -f "$ACTIVE_CONTAINER"', script)

    def test_wrapper_switches_pin_to_judge_and_restores_candidate(self):
        script = Path("scripts/run-swe-mini.sh").read_text(encoding="utf-8")

        self.assertIn("LMEVAL_WEBUI_LEMONADE_MODEL_PIN_LIFECYCLE_V2", script)
        self.assertIn("switchPinnedCandidateToJudge", script)
        self.assertIn("restorePinnedCandidate", script)
        self.assertIn("waitForModelIdle(candidate)", script)
        self.assertIn("loadAndConfirmPinnedModel", script)
        self.assertIn("task will be scored zero", script)
        self.assertIn("Judge infrastructure failure; task scored zero", script)
        self.assertIn("write_lifecycle_failure_result", script)
        self.assertIn("scoring this attempt zero and continuing", script)
        self.assertIn('"/internal/pin"', script)
        self.assertIn('"/api/v1/load"', script)
        self.assertIn("MODEL_LIFECYCLE_ENV_ARGS+=(--env", script)
        self.assertIn("LMEVAL_WEBUI_EXPLICIT_PROVIDER_TIMEOUT_V1", script)
        self.assertIn("LMEVAL_WEBUI_SWE_PROVIDER_TIMEOUT_MS", script)
        self.assertIn("write_inference_timeout_result", script)
        self.assertIn("wait_for_candidate_idle", script)
        self.assertIn("generate_aggregate_summary", script)


class TaskCompatibilityTests(unittest.TestCase):
    def test_malformed_generate_until_group_is_marked_incompatible(self):
        annotate_task_compatibility = symbol(
            "lm_eval_webui.server", "annotate_task_compatibility"
        )
        config_text = """
group: t0_eval
task:
  - dataset_path: aps/super_glue
    dataset_name: wsc.fixed
    output_type: generate_until
"""

        task = annotate_task_compatibility(
            {"name": "t0_eval", "description": "t0_eval.yaml"},
            lambda _path: config_text,
        )

        self.assertEqual(task["compatibility"], "incompatible")

    def test_dataset_script_tasks_are_marked_incompatible(self):
        annotate_task_compatibility = symbol(
            "lm_eval_webui.server", "annotate_task_compatibility"
        )
        for task_name, dataset_path in (
            ("anagrams1", "EleutherAI/unscramble"),
            ("french_bench_orangesum_title", "orange_sum"),
            ("ja_leaderboard_jaqket_v2", "kumapo/JAQKET"),
            ("logieval", "baber/logiqa2"),
            ("mlqa_en_en", "facebook/mlqa"),
            ("qasper_freeform", "allenai/qasper"),
            ("xlsum_es", "csebuetnlp/xlsum"),
        ):
            with self.subTest(task_name=task_name):
                config_text = f"""
task: {task_name}
dataset_path: {dataset_path}
output_type: generate_until
"""

                task = annotate_task_compatibility(
                    {"name": task_name, "description": f"{task_name}.yaml"},
                    lambda _path, text=config_text: text,
                )

                self.assertEqual(task["compatibility"], "incompatible")

    def test_bleurt_metric_tasks_are_marked_incompatible(self):
        annotate_task_compatibility = symbol(
            "lm_eval_webui.server", "annotate_task_compatibility"
        )
        config_text = """
task: careqa_open
output_type: generate_until
metric_list:
  - metric: bleurt
"""

        task = annotate_task_compatibility(
            {"name": "careqa_open", "description": "careqa_open.yaml"},
            lambda _path: config_text,
        )

        self.assertEqual(task["compatibility"], "incompatible")

    def test_code_eval_metric_tasks_are_marked_incompatible(self):
        annotate_task_compatibility = symbol(
            "lm_eval_webui.server", "annotate_task_compatibility"
        )
        config_text = """
task: humaneval
output_type: generate_until
metric_list:
  - metric: !function utils.pass_at_k
"""

        task = annotate_task_compatibility(
            {"name": "humaneval", "description": "humaneval.yaml"},
            lambda _path: config_text,
        )

        self.assertEqual(task["compatibility"], "incompatible")

    def test_unavailable_metric_tasks_are_marked_incompatible(self):
        annotate_task_compatibility = symbol(
            "lm_eval_webui.server", "annotate_task_compatibility"
        )
        config_text = """
task: wmt-ro-en-t5-prompt
output_type: generate_until
metric_list:
  - metric: wer
"""

        task = annotate_task_compatibility(
            {"name": "wmt-ro-en-t5-prompt", "description": "wmt-ro-en-t5-prompt.yaml"},
            lambda _path: config_text,
        )

        self.assertEqual(task["compatibility"], "incompatible")

    def test_openai_judged_process_result_tasks_require_openai_api_key(self):
        annotate_task_compatibility = symbol(
            "lm_eval_webui.server", "annotate_task_compatibility"
        )
        config_text = """
task: pisa_en_llm_judged
output_type: generate_until
process_results: !function utils.pisa_process_results_llm_judged
"""

        with mock.patch.dict(os.environ, {}, clear=True):
            task = annotate_task_compatibility(
                {
                    "name": "pisa_en_llm_judged",
                    "description": "pisa_en_llm_judged.yaml",
                },
                lambda _path: config_text,
            )

        self.assertEqual(task["compatibility"], "incompatible")

    def test_openai_judged_process_result_tasks_are_available_with_openai_api_key(self):
        annotate_task_compatibility = symbol(
            "lm_eval_webui.server", "annotate_task_compatibility"
        )
        config_text = """
task: pisa_en_llm_judged
output_type: generate_until
process_results: !function utils.pisa_process_results_llm_judged
"""

        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            task = annotate_task_compatibility(
                {
                    "name": "pisa_en_llm_judged",
                    "description": "pisa_en_llm_judged.yaml",
                },
                lambda _path: config_text,
            )

        self.assertEqual(task["compatibility"], "compatible")

    def test_gated_dataset_tasks_are_marked_gated(self):
        annotate_task_compatibility = symbol(
            "lm_eval_webui.server", "annotate_task_compatibility"
        )
        config_text = """
task: cocoteros_va
dataset_path: gplsi/cocoteros_va
output_type: generate_until
"""

        task = annotate_task_compatibility(
            {"name": "cocoteros_va", "description": "cocoteros_va.yaml"},
            lambda _path: config_text,
        )

        self.assertEqual(task["compatibility"], "gated")

        truthfulqa_task = annotate_task_compatibility(
            {"name": "truthfulqa_va", "description": "truthfulqa_va.yaml"},
            lambda _path: (
                """
task: truthfulqa_va
dataset_path: gplsi/truthfulqa_va
output_type: generate_until
"""
            ),
        )

        self.assertEqual(truthfulqa_task["compatibility"], "gated")

        gpqa_task = annotate_task_compatibility(
            {"name": "gpqa_main_generative_n_shot", "description": "gpqa.yaml"},
            lambda _path: (
                """
task: gpqa_main_generative_n_shot
dataset_path: Idavidrein/gpqa
output_type: generate_until
"""
            ),
        )

        self.assertEqual(gpqa_task["compatibility"], "gated")

        salt_task = annotate_task_compatibility(
            {"name": "salt_eng-swa_prompt_1", "description": "salt.yaml"},
            lambda _path: (
                """
task: salt_eng-swa_prompt_1
dataset_path: Sunbird/salt
output_type: generate_until
"""
            ),
        )

        self.assertEqual(salt_task["compatibility"], "gated")

    def test_unavailable_dataset_tasks_are_marked_incompatible(self):
        annotate_task_compatibility = symbol(
            "lm_eval_webui.server", "annotate_task_compatibility"
        )
        for task_name, dataset_path in (
            ("common_voice_en", "fixie-ai/endpointing-audio"),
            ("ja_leaderboard_jsquad", "Rakuten/JGLUE"),
            ("summarization_gl", "proxectonos/summarization_gl"),
        ):
            with self.subTest(task_name=task_name):
                config_text = f"""
task: {task_name}
dataset_path: {dataset_path}
output_type: generate_until
"""

                task = annotate_task_compatibility(
                    {"name": task_name, "description": f"{task_name}.yaml"},
                    lambda _path, text=config_text: text,
                )

                self.assertEqual(task["compatibility"], "incompatible")

    def test_multilingual_ifeval_tasks_are_marked_incompatible(self):
        annotate_task_compatibility = symbol(
            "lm_eval_webui.server", "annotate_task_compatibility"
        )
        config_text = """
task: ifeval_ca
output_type: generate_until
process_results: !function utils.process_results
"""

        task = annotate_task_compatibility(
            {"name": "ifeval_ca", "description": "ifeval_ca.yaml"},
            lambda _path: config_text,
        )

        self.assertEqual(task["compatibility"], "incompatible")

    def test_metadata_dependent_tasks_are_marked_incompatible(self):
        annotate_task_compatibility = symbol(
            "lm_eval_webui.server", "annotate_task_compatibility"
        )
        for task_name in (
            "niah_single_1",
            "niah_single_2",
            "niah_multikey_1",
            "niah_multiquery",
            "niah_multivalue",
        ):
            with self.subTest(task_name=task_name):
                config_text = f"""
task: {task_name}
dataset_path: ""
output_type: generate_until
"""

                task = annotate_task_compatibility(
                    {"name": task_name, "description": f"{task_name}.yaml"},
                    lambda _path, text=config_text: text,
                )

                self.assertEqual(task["compatibility"], "incompatible")

    def test_smoked_coding_tasks_are_marked_compatible(self):
        annotate_task_compatibility = symbol(
            "lm_eval_webui.server", "annotate_task_compatibility"
        )
        for task_name in (
            "jsonschema_bench",
            "bigbench_bbq_lite_json_generate_until",
            "bigbench_code_line_description_generate_until",
            "bigbench_codenames_generate_until",
            "bigbench_simple_arithmetic_json_generate_until",
            "bigbench_simple_arithmetic_json_subtasks_generate_until",
            "code2text_go",
            "code2text_java",
            "code2text_javascript",
            "code2text_php",
            "code2text_python",
            "code2text_ruby",
            "jsonschema_bench_hard",
            "jsonschema_bench_medium",
        ):
            with self.subTest(task_name=task_name):
                config_text = f"""
task: {task_name}
"""

                task = annotate_task_compatibility(
                    {"name": task_name, "description": f"{task_name}.yaml"},
                    lambda _path, text=config_text: text,
                )

                self.assertEqual(task["compatibility"], "compatible")

    def test_smoked_coding_failures_are_marked_incompatible(self):
        annotate_task_compatibility = symbol(
            "lm_eval_webui.server", "annotate_task_compatibility"
        )
        for task_name in (
            "bigbench_bbq_lite_json_multiple_choice",
            "bigbench_code_line_description_multiple_choice",
            "bigbench_simple_arithmetic_json_multiple_choice_generate_until",
            "bigbench_simple_arithmetic_multiple_targets_json_generate_until",
            "humaneval_64_instruct",
            "humaneval_instruct",
            "humaneval_plus",
            "humaneval_random_span_infilling",
            "humaneval_single_line_infilling",
            "humaneval_single_line_infilling_light",
            "infinitebench_code_debug",
            "infinitebench_code_run",
            "longbench_code_tasks",
            "longbench_code_tasks_e",
            "longbench_lcc",
            "longbench_lcc_e",
            "longbench_repobench-p",
            "longbench_repobench-p_e",
            "longbench2_code",
            "mbpp_plus",
            "mbpp_plus_instruct",
            "toksuite_chinese_code_language_script_switching",
            "toksuite_farsi_code_language_script_switching",
            "toksuite_italian_code_language_script_switching",
            "toksuite_stem_unicode_formatting",
            "toksuite_turkish_code_language_script_switching",
        ):
            with self.subTest(task_name=task_name):
                config_text = f"""
task: {task_name}
output_type: generate_until
"""

                task = annotate_task_compatibility(
                    {"name": task_name, "description": f"{task_name}.yaml"},
                    lambda _path, text=config_text: text,
                )

                self.assertEqual(task["compatibility"], "incompatible")

    def test_aggregate_groups_and_tags_are_marked_with_kind(self):
        annotate_task_compatibility = symbol(
            "lm_eval_webui.server", "annotate_task_compatibility"
        )
        for task, expected_kind in (
            ({"name": "bbh", "description": "bbh.yaml", "kind": "group"}, "group"),
            (
                {
                    "name": "bbh_cot_fewshot",
                    "description": "bbh_cot_fewshot.yaml",
                    "kind": "group",
                },
                "group",
            ),
            (
                {
                    "name": "mmlu_cot_llama_humanities_tasks",
                    "description": "",
                    "kind": "tag",
                },
                "tag",
            ),
        ):
            with self.subTest(task_name=task["name"]):
                classified = annotate_task_compatibility(
                    task,
                    lambda _path: (
                        """
group: aggregate
task:
  - child_task
"""
                    ),
                )

                self.assertEqual(classified["kind"], expected_kind)
                self.assertEqual(classified["compatibility"], "compatible")

    def test_lm_eval_task_table_parser_records_row_kind(self):
        parse_lm_eval_task_table = symbol(
            "lm_eval_webui.server", "parse_lm_eval_task_table"
        )
        output = """
| Group | Config Location |
|-------|-----------------|
| bbh | lm_eval/tasks/bbh/_bbh.yaml |

| Tag |
|-----|
| mmlu_cot_llama_humanities_tasks |

| Task | Config Location | Output Type |
|------|-----------------|-------------|
| bbh_cot_fewshot_boolean_expressions | lm_eval/tasks/bbh/boolean_expressions.yaml | generate_until |
"""

        rows = parse_lm_eval_task_table(output)

        self.assertEqual(
            [(row["name"], row["kind"]) for row in rows],
            [
                ("bbh", "group"),
                ("mmlu_cot_llama_humanities_tasks", "tag"),
                ("bbh_cot_fewshot_boolean_expressions", "task"),
            ],
        )

    def test_common_task_aggregate_entries_do_not_mask_discovered_kind(self):
        load_available_tasks = symbol("lm_eval_webui.server", "load_available_tasks")

        class Completed:
            returncode = 0
            stdout = """
| Group | Config Location |
|-------|-----------------|
| bbh_cot_zeroshot | lm_eval/tasks/bbh/cot_zeroshot/_bbh_cot_zeroshot.yaml |
"""

        tasks = load_available_tasks(
            "/home/iain/.venv/lm-eval/bin/python",
            run_command=lambda *_args, **_kwargs: Completed(),
            config_reader=lambda _path: (
                """
group: bbh_cot_zeroshot
task:
  - bbh_cot_zeroshot_boolean_expressions
"""
            ),
        )
        by_name = {task["name"]: task for task in tasks}

        self.assertEqual(by_name["bbh_cot_zeroshot"]["kind"], "group")
        self.assertEqual(by_name["bbh_cot_zeroshot"]["compatibility"], "compatible")

    def test_smoked_reasoning_instruction_math_tasks_are_marked_compatible(self):
        annotate_task_compatibility = symbol(
            "lm_eval_webui.server", "annotate_task_compatibility"
        )
        for task_name in (
            "bigbench_natural_instructions_generate_until",
            "bigbench_elementary_math_qa_generate_until",
            "hendrycks_math500",
            "bbh_cot_fewshot_boolean_expressions",
            "truthfulqa-multi_gen_en",
            "mmlu_cot_llama_abstract_algebra",
            "mmlu_prox_en_biology",
            "mmlu_prox_lite_en_biology",
            "metabench_gsm8k_subset",
            "score_prompt_robustness_math",
            "score_non_greedy_robustness_math",
            "score_robustness_math",
            "score_robustness_mmlu_pro",
            "leaderboard_instruction_following",
            "leaderboard_math_hard",
            "minerva_math",
            "mmlu_college_mathematics_generative",
            "mmlu_llama_college_mathematics",
            "mmlu_pro_biology",
        ):
            with self.subTest(task_name=task_name):
                config_text = f"""
task: {task_name}
"""

                task = annotate_task_compatibility(
                    {"name": task_name, "description": f"{task_name}.yaml"},
                    lambda _path, text=config_text: text,
                )

                self.assertEqual(task["compatibility"], "compatible")

    def test_smoked_reasoning_instruction_math_failures_are_marked_incompatible(self):
        annotate_task_compatibility = symbol(
            "lm_eval_webui.server", "annotate_task_compatibility"
        )
        for task_name in (
            "tinyGSM8k",
            "bigbench_elementary_math_qa_multiple_choice",
            "agieval_sat_math",
            "pile_dm-mathematics",
            "truthfulqa-multi_mc1_en",
            "afrimmlu_direct_eng_prompt_1",
            "tmmluplus_logic_reasoning",
            "global_mmlu_full_en_abstract_algebra",
            "toksuite_math_canonical",
            "math_word_problems",
            "m_mmlu_en",
            "acp_app_gen",
            "acp_app_gen_with_pddl",
            "acp_reach_mcq",
            "infinitebench_kv_retrieval",
            "infinitebench_longbook_choice_en",
            "infinitebench_passkey",
            "ruler_cwe",
            "ruler_qa_squad",
            "ruler_vt",
            "cmmlu_college_mathematics",
            "arc_multilingual",
            "metabench_mmlu_subset",
            "metabench_arc_subset",
            "AraDiCE_ArabicMMLU_egy",
            "uhura-arc-easy_en_prompt_1",
            "naijarc_yor_prompt_1",
            "openai_mmlu_yor_prompt_1",
            "openai_mmlu",
            "mmmlu_zh_cn_abstract_algebra",
            "nortruthfulqa_mc_nno",
            "nortruthfulqa_mc_nob",
            "truthfulqa",
            "truthfulqa-multi",
            "truthfulqa_gl",
            "truthfulqa_multi",
            "truthfulqa_multilingual",
            "libra_complex_reasoning_and_mathematical_problems",
            "leaderboard_bbh",
            "mmlu_college_mathematics",
            "mmlu_flan_n_shot_loglikelihood_college_mathematics",
            "mmlu_humanities_continuation",
        ):
            with self.subTest(task_name=task_name):
                config_text = f"""
task: {task_name}
output_type: generate_until
"""

                task = annotate_task_compatibility(
                    {"name": task_name, "description": f"{task_name}.yaml"},
                    lambda _path, text=config_text: text,
                )

                self.assertEqual(task["compatibility"], "incompatible")

    def test_lm_eval_config_reader_expands_simple_includes(self):
        read_lm_eval_config = symbol("lm_eval_webui.server", "read_lm_eval_config")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tasks_dir = root / "tasks"
            tasks_dir.mkdir()
            (tasks_dir / "generate_until_template.yaml").write_text(
                "output_type: generate_until\n", encoding="utf-8"
            )
            task_path = tasks_dir / "example.yaml"
            task_path.write_text(
                "include: generate_until_template_yaml\ntask: example\n",
                encoding="utf-8",
            )

            config_text = read_lm_eval_config(str(task_path), root)

        self.assertIsNotNone(config_text)
        self.assertIn("output_type: generate_until", config_text or "")
        self.assertIn("task: example", config_text or "")

    def test_task_language_scope_marks_non_english_tasks(self):
        annotate_task_compatibility = symbol(
            "lm_eval_webui.server", "annotate_task_compatibility"
        )
        for task_name in (
            "mmmlu_zh_cn_abstract_algebra",
            "truthfulqa-multi_gen_es",
            "openai_mmlu_yor_prompt_1",
            "include_base_44_arabic",
            "flores_afr-eng",
            "toksuite_turkish_web_search_query",
            "mmlu_redux_spanish_generative",
            "mmlu_high_school_mathematics_generative_spanish",
            "global_piqa_prompted_spa_latn_spai",
            "global_piqa_prompted_deu_latn",
            "global_piqa_prompted_jpn_jpan",
            "pisa_ch",
            "pisa_de",
            "pisa_es",
            "pisa_fr",
            "pisa_it",
            "bigbench_kanji_ascii_generate_until",
            "bigbench_hinglish_toxicity_generate_until",
            "polemo2_in",
            "polemo2_out",
            "jfinqa_ja",
            "jfinqa_zh",
            "jfinqa_out",
            "xquad_ar",
            "xquad_de",
            "xquad_es",
            "xquad_zh",
            "librusec_history",
            "librusec_mhqa",
        ):
            with self.subTest(task_name=task_name):
                config_text = f"task: {task_name}\n"
                task = annotate_task_compatibility(
                    {"name": task_name, "description": f"{task_name}.yaml"},
                    lambda _path, config_text=config_text: config_text,
                )

                self.assertEqual(task["language_scope"], "non_english")

    def test_task_language_scope_keeps_english_tasks(self):
        annotate_task_compatibility = symbol(
            "lm_eval_webui.server", "annotate_task_compatibility"
        )
        for task_name in (
            "gsm8k",
            "mmlu_prox_en_biology",
            "truthfulqa-multi_gen_en",
            "code2text_python",
            "global_piqa_prompted_eng_latn",
            "pisa_en",
            "pisa_en_llm_judged",
            "xquad_en",
        ):
            with self.subTest(task_name=task_name):
                config_text = f"task: {task_name}\n"
                task = annotate_task_compatibility(
                    {"name": task_name, "description": f"{task_name}.yaml"},
                    lambda _path, config_text=config_text: config_text,
                )

                self.assertEqual(task["language_scope"], "english")

    def test_remaining_smoked_tasks_are_marked_compatible(self):
        annotate_task_compatibility = symbol(
            "lm_eval_webui.server", "annotate_task_compatibility"
        )
        for task_name in (
            "graphwalks_128k",
            "code2text",
            "ntrex_afr-eng",
            "ntrex_eng-afr_prompt_3",
            "adr_prompt_1",
            "adr_tasks",
            "jfinqa",
        ):
            with self.subTest(task_name=task_name):
                config_text = f"task: {task_name}\n"
                task = annotate_task_compatibility(
                    {"name": task_name, "description": f"{task_name}.yaml"},
                    lambda _path, text=config_text: text,
                )

                self.assertEqual(task["compatibility"], "compatible")

    def test_remaining_smoked_failures_are_marked_incompatible(self):
        annotate_task_compatibility = symbol(
            "lm_eval_webui.server", "annotate_task_compatibility"
        )
        for task_name in (
            "graphwalks_1M",
            "graphwalks",
            "meddialog_qsumm",
            "humaneval_infilling",
            "longbench",
            "longbench_2wikimqa",
            "longbench2_single",
            "scrolls_qasper",
            "agieval",
            "leaderboard",
            "leaderboard_gpqa",
            "leaderboard_musr",
            "tinyBenchmarks",
            "openllm",
            "pythia",
            "afrimgsm-irokobench",
            "afrixnli_en_direct",
            "african_flores",
            "flores_afr-eng",
            "mafand_afr-eng",
            "afriqa_prompt_1",
            "afrisenti_prompt_1",
            "masakhanews_prompt_1",
            "masakhaner_prompt_1",
            "masakhapos_prompt_1",
            "nollysenti_prompt_1",
            "sib_prompt_1",
            "injongointent_prompt_1",
            "include_base_44_arabic",
            "20_newsgroups",
            "ag_news",
            "cnn_dailymail",
            "doc_vqa",
            "bigbench_list_functions_generate_until",
            "stsb",
            "med_concepts_qa_atc",
            "multimedqa",
            "japanese_leaderboard",
            "wmdp",
            "pawsx",
            "xcopa",
            "xnli",
            "xstorycloze",
            "xwinograd",
            "blimp",
            "lambada",
            "lambada_cloze",
        ):
            with self.subTest(task_name=task_name):
                config_text = f"task: {task_name}\n"
                task = annotate_task_compatibility(
                    {"name": task_name, "description": f"{task_name}.yaml"},
                    lambda _path, text=config_text: text,
                )

                self.assertEqual(task["compatibility"], "incompatible")

    def test_unclassified_no_output_tasks_default_incompatible(self):
        annotate_task_compatibility = symbol(
            "lm_eval_webui.server", "annotate_task_compatibility"
        )

        task = annotate_task_compatibility(
            {"name": "new_unclassified_group", "description": "new_group.yaml"},
            lambda _path: "group: new_unclassified_group\ntask:\n  - child_task\n",
        )

        self.assertEqual(task["compatibility"], "incompatible")


class JobManagerTelemetryTests(unittest.TestCase):
    def test_job_persists_requested_llamacpp_backend(self):
        JobManager = symbol("lm_eval_webui.jobs", "JobManager")

        with tempfile.TemporaryDirectory() as tmp:
            manager = JobManager(
                data_dir=Path(tmp) / "data",
                project_root=Path("/repo"),
                run_async=False,
                launcher=lambda _command, _env, _log_path: 0,
            )

            created = manager.create_jobs(
                {
                    "model_ids": ["Model-A"],
                    "tasks": ["gsm8k"],
                    "llamacpp_backend": "vulkan",
                }
            )
            job = manager.get_job(created[0]["id"])

        self.assertIn("llamacpp_backend=vulkan", job["command"])
        self.assertEqual(job["requested_llamacpp_backend"], "vulkan")
        self.assertEqual(job["provider_backend"], "vulkan")

    def test_successful_job_persists_runtime_backend_metadata(self):
        JobManager = symbol("lm_eval_webui.jobs", "JobManager")

        def launcher(command, _env, _log_path):
            output_path = Path(command[command.index("--output_path") + 1])
            result_dir = output_path / "Model-A"
            result_dir.mkdir(parents=True)
            (result_dir / "results_2026-06-21T00-00-00.json").write_text(
                json.dumps(
                    {
                        "config": {
                            "model": "openai-compatible-chat-completions",
                            "model_args": {"model": "Model-A"},
                            "limit": 1,
                        },
                        "results": {
                            "gsm8k": {
                                "exact_match,strict-match": 1.0,
                                "exact_match,flexible-extract": 1.0,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            return 0

        with tempfile.TemporaryDirectory() as tmp:
            manager = JobManager(
                data_dir=Path(tmp) / "data",
                project_root=Path("/repo"),
                launcher=launcher,
                run_async=False,
                model_metadata_probe=lambda _base_url, _model_id: {
                    "recipe": "llamacpp",
                    "llamacpp_backend": "vulkan",
                    "runtime_backend": "vulkan",
                    "device": "gpu",
                },
            )

            created = manager.create_jobs(
                {"model_ids": ["Model-A"], "tasks": ["gsm8k"]}
            )
            job = manager.get_job(created[0]["id"])
            leaderboard = manager.leaderboard_entries()

        self.assertEqual(job["model_metadata"]["runtime_backend"], "vulkan")
        self.assertEqual(job["provider_backend"], "vulkan")
        self.assertEqual(leaderboard[0]["provider_backend"], "vulkan")

    def test_successful_job_falls_back_to_backend_when_metadata_probe_misses(self):
        JobManager = symbol("lm_eval_webui.jobs", "JobManager")

        def launcher(command, _env, _log_path):
            output_path = Path(command[command.index("--output_path") + 1])
            result_dir = output_path / "Model-A"
            result_dir.mkdir(parents=True)
            (result_dir / "results_2026-06-21T00-00-00.json").write_text(
                json.dumps(
                    {
                        "config": {
                            "model": "openai-compatible-chat-completions",
                            "model_args": {"model": "Model-A"},
                            "limit": 1,
                        },
                        "results": {
                            "gsm8k": {
                                "exact_match,strict-match": 1.0,
                                "exact_match,flexible-extract": 1.0,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            return 0

        with tempfile.TemporaryDirectory() as tmp:
            manager = JobManager(
                data_dir=Path(tmp) / "data",
                project_root=Path("/repo"),
                launcher=launcher,
                run_async=False,
                model_metadata_probe=lambda _base_url, _model_id: {},
            )

            created = manager.create_jobs(
                {"model_ids": ["Model-A"], "tasks": ["gsm8k"]}
            )
            job = manager.get_job(created[0]["id"])
            leaderboard = manager.leaderboard_entries()

        self.assertEqual(job["runtime_backend"], "openai-compatible-chat-completions")
        self.assertEqual(job["provider_backend"], "openai-compatible-chat-completions")
        self.assertEqual(
            leaderboard[0]["provider_backend"], "openai-compatible-chat-completions"
        )

    def test_probe_is_skipped_when_benchmark_ttft_exists(self):
        JobManager = symbol("lm_eval_webui.jobs", "JobManager")
        probe_called = False

        with tempfile.TemporaryDirectory() as tmp:
            telemetry_path = Path(tmp) / "telemetry.jsonl"
            telemetry_path.write_text(
                json.dumps(
                    {
                        "timings": {
                            "predicted_n": 2,
                            "predicted_ms": 100,
                            "ttft_s": 0.25,
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            def telemetry_probe(_base_url, _model_id):
                nonlocal probe_called
                probe_called = True
                return {"ttft_s": 10.0}

            manager = JobManager(
                data_dir=Path(tmp) / "data",
                project_root=Path("/repo"),
                run_async=False,
                telemetry_probe=telemetry_probe,
            )

            telemetry = manager._collect_telemetry(
                {
                    "telemetry_path": str(telemetry_path),
                    "openai_base_url": "http://example.test",
                    "model_id": "Model-A",
                },
                0,
            )

        self.assertEqual(telemetry["ttft_s"], 0.25)
        self.assertFalse(probe_called)
        self.assertNotIn("probe_ttft_s", telemetry)
        self.assertNotIn("error", telemetry)

    def test_probe_rate_is_used_when_benchmark_has_ttft_but_no_rate(self):
        JobManager = symbol("lm_eval_webui.jobs", "JobManager")

        with tempfile.TemporaryDirectory() as tmp:
            telemetry_path = Path(tmp) / "telemetry.jsonl"
            telemetry_path.write_text(
                json.dumps(
                    {
                        "timings": {"ttft_s": 0.25},
                        "usage": {"completion_tokens": 2},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            manager = JobManager(
                data_dir=Path(tmp) / "data",
                project_root=Path("/repo"),
                run_async=False,
                telemetry_probe=lambda _base_url, _model_id: {
                    "ttft_s": 10.0,
                    "generation_tok_s": 12.5,
                },
            )

            telemetry = manager._collect_telemetry(
                {
                    "telemetry_path": str(telemetry_path),
                    "openai_base_url": "http://example.test",
                    "model_id": "Model-A",
                },
                0,
            )

        self.assertEqual(telemetry["ttft_s"], 0.25)
        self.assertEqual(telemetry["probe_generation_tok_s"], 12.5)

    def test_probe_ttft_is_used_when_benchmark_ttft_is_missing(self):
        JobManager = symbol("lm_eval_webui.jobs", "JobManager")

        with tempfile.TemporaryDirectory() as tmp:
            telemetry_path = Path(tmp) / "telemetry.jsonl"
            telemetry_path.write_text(
                json.dumps({"timings": {"predicted_n": 2, "predicted_ms": 100}}) + "\n",
                encoding="utf-8",
            )
            manager = JobManager(
                data_dir=Path(tmp) / "data",
                project_root=Path("/repo"),
                run_async=False,
                telemetry_probe=lambda _base_url, _model_id: {
                    "ttft_s": 10.0,
                    "time_to_headers_s": 9.0,
                },
            )

            telemetry = manager._collect_telemetry(
                {
                    "telemetry_path": str(telemetry_path),
                    "openai_base_url": "http://example.test",
                    "model_id": "Model-A",
                },
                0,
            )

        self.assertEqual(telemetry["ttft_s"], 10.0)
        self.assertEqual(telemetry["probe_ttft_s"], 10.0)
        self.assertEqual(telemetry["probe_time_to_headers_s"], 9.0)


class JobManagerModelProtectionTests(unittest.TestCase):
    def test_model_is_pinned_across_all_task_batches_and_released_afterward(self):
        JobManager = symbol("lm_eval_webui.jobs", "JobManager")
        events = []

        def pin_model(base_url, model_id, timeout):
            events.append(("pin", base_url, model_id, timeout))
            return {}

        def unpin_model(base_url, model_id, timeout):
            events.append(("unpin", base_url, model_id, timeout))
            return {}

        def launcher(command, _env, _log_path):
            tasks = command[
                command.index("--tasks") + 1 : command.index("--output_path")
            ]
            events.append(("launch", tasks))
            return 0

        with tempfile.TemporaryDirectory() as tmp:
            manager = JobManager(
                data_dir=Path(tmp) / "data",
                project_root=Path("/repo"),
                launcher=launcher,
                run_async=False,
                protect_models=True,
                model_pin_loader=pin_model,
                model_unpinner=unpin_model,
            )

            created = manager.create_jobs(
                {
                    "model_ids": ["Model-A"],
                    "tasks": ["task_a", "task_b", "task_c"],
                    "task_batch_size": 1,
                    "openai_base_url": "https://llm.example.test/v1",
                }
            )
            job = manager.get_job(created[0]["id"])

        self.assertEqual(
            events[0][:3], ("pin", "https://llm.example.test/v1", "Model-A")
        )
        self.assertEqual(
            [event for event in events if event[0] == "launch"],
            [
                ("launch", ["task_a"]),
                ("launch", ["task_b"]),
                ("launch", ["task_c"]),
            ],
        )
        self.assertEqual(
            events[-1][:3], ("unpin", "https://llm.example.test/v1", "Model-A")
        )
        self.assertEqual(job["status"], "succeeded")
        self.assertEqual(job["model_protection"]["state"], "released")

    def test_restart_releases_a_pin_left_by_an_interrupted_job(self):
        JobManager = symbol("lm_eval_webui.jobs", "JobManager")
        unpinned = []

        def unpin_model(base_url, model_id, _timeout):
            unpinned.append((base_url, model_id))
            return {}

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            jobs_dir = data_dir / "jobs"
            jobs_dir.mkdir(parents=True)
            (jobs_dir / "interrupted.json").write_text(
                json.dumps(
                    {
                        "id": "interrupted",
                        "model_id": "Model-A",
                        "tasks": ["gsm8k"],
                        "status": "running",
                        "created_at": 1,
                        "updated_at": 1,
                        "openai_base_url": "https://llm.example.test/v1",
                        "output_path": str(data_dir / "runs" / "interrupted"),
                        "log_path": str(data_dir / "logs" / "interrupted.log"),
                        "model_protection": {
                            "state": "pinned",
                            "model_id": "Model-A",
                        },
                    }
                ),
                encoding="utf-8",
            )

            manager = JobManager(
                data_dir=data_dir,
                project_root=Path("/repo"),
                run_async=False,
                protect_models=False,
                model_unpinner=unpin_model,
            )
            interrupted = manager.get_job("interrupted")

        self.assertEqual(unpinned, [("https://llm.example.test/v1", "Model-A")])
        self.assertEqual(interrupted["status"], "failed")
        self.assertEqual(
            interrupted["model_protection"]["state"], "released_after_restart"
        )


class JobManagerBatchTests(unittest.TestCase):
    @staticmethod
    def _command_tasks(command: list[str]) -> list[str]:
        return command[command.index("--tasks") + 1 : command.index("--output_path")]

    @staticmethod
    def _write_result(command: list[str], score: float = 1.0) -> None:
        tasks = JobManagerBatchTests._command_tasks(command)
        output_path = Path(command[command.index("--output_path") + 1])
        result_dir = output_path / "Model-A"
        result_dir.mkdir(parents=True)
        (result_dir / "results_2026-06-30T00-00-00.json").write_text(
            json.dumps(
                {
                    "model_name": "Model-A",
                    "config": {
                        "model": "openai-compatible-chat-completions",
                        "model_args": {"model": "Model-A"},
                        "limit": 1,
                    },
                    "results": {
                        task: {"acc,none": score, "sample_len": 1} for task in tasks
                    },
                }
            ),
            encoding="utf-8",
        )

    def test_lm_eval_job_runs_task_batches_sequentially(self):
        JobManager = symbol("lm_eval_webui.jobs", "JobManager")
        commands = []

        def launcher(command, _env, _log_path):
            commands.append(command)
            self._write_result(command)
            return 0

        with tempfile.TemporaryDirectory() as tmp:
            manager = JobManager(
                data_dir=Path(tmp) / "data",
                project_root=Path("/repo"),
                launcher=launcher,
                run_async=False,
                lm_eval_python="/venv/bin/python",
            )

            created = manager.create_jobs(
                {
                    "model_ids": ["Model-A"],
                    "tasks": ["task_a", "task_b", "task_c", "task_d", "task_e"],
                    "task_batch_size": 2,
                }
            )
            job = manager.get_job(created[0]["id"])
            leaderboard = manager.leaderboard_entries()

        self.assertEqual(
            [self._command_tasks(command) for command in commands],
            [["task_a", "task_b"], ["task_c", "task_d"], ["task_e"]],
        )
        self.assertEqual(job["status"], "succeeded")
        self.assertEqual(job["returncode"], 0)
        self.assertEqual(job["task_batch_size"], 2)
        self.assertEqual(job["eval_options"]["task_batch_size"], 2)
        self.assertEqual(job["batch_progress"]["total"], 3)
        self.assertEqual(job["batch_progress"]["completed"], 3)
        self.assertEqual(len(job["result_files"]), 3)
        self.assertGreaterEqual(job["finished_at"], job["started_at"])
        self.assertGreaterEqual(job["runtime_seconds"], 0)
        self.assertEqual(len(leaderboard), 1)
        self.assertEqual(leaderboard[0]["runtime_seconds"], job["runtime_seconds"])
        self.assertEqual(
            sorted(score["task"] for score in leaderboard[0]["task_scores"]),
            ["task_a", "task_b", "task_c", "task_d", "task_e"],
        )

    def test_lm_eval_running_progress_reports_requests_inside_current_task(self):
        JobManager = symbol("lm_eval_webui.jobs", "JobManager")

        with tempfile.TemporaryDirectory() as tmp:
            manager = JobManager(
                data_dir=Path(tmp) / "data",
                project_root=Path("/repo"),
                launcher=lambda _command, _env, _log_path: 0,
                run_async=False,
            )
            created = manager._create_job(
                "Model-A",
                ["gsm8k", "ifeval", "mmlu"],
                {"task_batch_size": 1},
            )
            job = manager.get_job(created["id"])
            job["status"] = "running"
            job["batch_progress"] = {
                "task_batch_size": 1,
                "total": 3,
                "completed": 0,
                "current": 1,
                "current_tasks": ["gsm8k"],
                "failed": None,
            }
            manager._write_job(job)
            Path(job["log_path"]).write_text(
                "=== lm-eval task batch 1/3 (1 task) ===\n"
                "Requesting API:  46%|#### | 607/1319 [7:15:24<4:54:20]\r",
                encoding="utf-8",
            )

            running = manager.get_job(created["id"])

        self.assertEqual(running["progress"]["current"], 1)
        self.assertEqual(running["progress"]["total"], 3)
        self.assertEqual(running["progress"]["completed"], 0)
        self.assertEqual(running["request_progress"]["current"], 607)
        self.assertEqual(running["request_progress"]["total"], 1319)
        self.assertEqual(running["request_progress"]["unit"], "requests")
        self.assertEqual(running["request_progress"]["batch"], 1)

    def test_lm_eval_request_progress_works_without_task_batching(self):
        JobManager = symbol("lm_eval_webui.jobs", "JobManager")

        with tempfile.TemporaryDirectory() as tmp:
            manager = JobManager(
                data_dir=Path(tmp) / "data",
                project_root=Path("/repo"),
                launcher=lambda _command, _env, _log_path: 0,
                run_async=False,
            )
            created = manager._create_job(
                "Model-A",
                ["gsm8k", "ifeval", "truthfulqa_gen"],
                {"task_batch_size": None},
            )
            job = manager.get_job(created["id"])
            job["status"] = "running"
            manager._write_job(job)
            Path(job["log_path"]).write_text(
                "Running generate_until requests\n"
                "Requesting API:  10%|# | 129/1319 [1:12:14<11:00:00]\r",
                encoding="utf-8",
            )

            running = manager.get_job(created["id"])

        self.assertNotIn("progress", running)
        self.assertEqual(running["request_progress"]["current"], 129)
        self.assertEqual(running["request_progress"]["total"], 1319)
        self.assertAlmostEqual(running["request_progress"]["percent"], 9.7801, places=3)
        self.assertNotIn("batch", running["request_progress"])

    def test_lm_eval_job_without_task_batching_tracks_its_single_process(self):
        JobManager = symbol("lm_eval_webui.jobs", "JobManager")
        commands = []

        def launcher(command, _env, _log_path):
            commands.append(command)
            return 0

        with tempfile.TemporaryDirectory() as tmp:
            manager = JobManager(
                data_dir=Path(tmp) / "data",
                project_root=Path("/repo"),
                launcher=launcher,
                run_async=False,
            )
            created = manager.create_jobs(
                {
                    "model_ids": ["Model-A"],
                    "tasks": ["gsm8k", "ifeval", "truthfulqa_gen"],
                    "task_batch_size": None,
                }
            )
            job = manager.get_job(created[0]["id"])
            log = Path(job["log_path"]).read_text(encoding="utf-8")

        self.assertEqual(len(commands), 1)
        self.assertEqual(job["status"], "succeeded")
        self.assertEqual(job["batch_progress"]["total"], 1)
        self.assertEqual(job["batch_progress"]["completed"], 1)
        self.assertIsNone(job["batch_progress"]["current"])
        self.assertEqual(job["batch_progress"]["task_batch_size"], 3)
        self.assertIn("=== lm-eval task batch 1/1 (3 tasks) ===", log)

    def test_lm_eval_job_stops_after_failed_task_batch(self):
        JobManager = symbol("lm_eval_webui.jobs", "JobManager")
        commands = []

        def launcher(command, _env, _log_path):
            commands.append(command)
            return 7 if len(commands) == 2 else 0

        with tempfile.TemporaryDirectory() as tmp:
            manager = JobManager(
                data_dir=Path(tmp) / "data",
                project_root=Path("/repo"),
                launcher=launcher,
                run_async=False,
            )

            created = manager.create_jobs(
                {
                    "model_ids": ["Model-A"],
                    "tasks": ["task_a", "task_b", "task_c", "task_d", "task_e"],
                    "task_batch_size": 2,
                }
            )
            job = manager.get_job(created[0]["id"])

        self.assertEqual(len(commands), 2)
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["returncode"], 7)
        self.assertEqual(job["batch_progress"]["completed"], 1)
        self.assertEqual(job["batch_progress"]["failed"], 2)


class JobManagerSweMiniTests(unittest.TestCase):
    def _write_swe_task(self, pi_bench_dir: Path, task_id: str) -> None:
        task_dir = pi_bench_dir / "tasks" / "verified-mini"
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / f"{task_id}.json").write_text(
            json.dumps(
                {
                    "id": task_id,
                    "repo": "django/django",
                    "prompt": "Fix the regression.",
                }
            ),
            encoding="utf-8",
        )

    def test_swe_mini_running_progress_is_parsed_from_log(self):
        JobManager = symbol("lm_eval_webui.jobs", "JobManager")

        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "repo"
            pi_bench_dir = project_root / "third_party" / "pi-bench"
            scripts_dir = project_root / "scripts"
            pi_bench_dir.mkdir(parents=True)
            scripts_dir.mkdir()
            (scripts_dir / "run-swe-mini.sh").write_text(
                "#!/bin/sh\n", encoding="utf-8"
            )
            self._write_swe_task(pi_bench_dir, "django__django-11790")
            self._write_swe_task(pi_bench_dir, "django__django-11815")
            self._write_swe_task(pi_bench_dir, "django__django-11848")
            manager = JobManager(
                data_dir=Path(tmp) / "data",
                project_root=project_root,
                run_async=True,
                pi_bench_dir=pi_bench_dir,
            )
            created = manager._create_swe_mini_job(
                "Model-A",
                [
                    "django__django-11790",
                    "django__django-11815",
                    "django__django-11848",
                ],
                {"suite": "swe_mini"},
            )
            job = manager.get_job(created["id"])
            job["status"] = "running"
            manager._write_job(job)
            Path(job["log_path"]).write_text(
                "[1/3] Task: django__django-11790\n[2/3] Task: django__django-11815\n",
                encoding="utf-8",
            )

            listed = manager.list_jobs()[0]

        self.assertEqual(job["swe_options"]["timeout_minutes"], 60)
        self.assertEqual(job["swe_options"]["context_window"], 65536)
        self.assertEqual(job["swe_options"]["max_output_tokens"], 16384)
        self.assertEqual(job["swe_options"]["provider_timeout_minutes"], 15)
        self.assertEqual(job["swe_options"]["provider_max_retries"], 0)
        self.assertEqual(job["swe_options"]["recipe_policy"], "lemonade_unchanged")
        self.assertEqual(listed["progress"]["current"], 2)
        self.assertEqual(listed["progress"]["total"], 3)
        self.assertAlmostEqual(listed["progress"]["percent"], 66.6666666667)

    def test_swe_mini_job_uses_suite_command_and_parses_summary(self):
        JobManager = symbol("lm_eval_webui.jobs", "JobManager")
        commands = []
        envs = []
        pin_events = []

        def pin_model(base_url, model_id, _timeout):
            pin_events.append(("pin", base_url, model_id))
            return {}

        def unpin_model(base_url, model_id, _timeout):
            pin_events.append(("unpin", base_url, model_id))
            return {}

        def launcher(command, env, _log_path):
            commands.append(command)
            envs.append(env)
            output_path = Path(env["SWE_MINI_OUTPUT_PATH"])
            output_path.mkdir(parents=True)
            (output_path / "summary.json").write_text(
                json.dumps(
                    {
                        "totalTasks": 1,
                        "passedTasks": 1,
                        "passRate": 1.0,
                        "averageDurationMs": 1000,
                        "results": [
                            {
                                "task": "django__django-12209",
                                "durationMs": 1000,
                                "judgeScore": 1,
                                "judgeRationale": "fixed",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            return 0

        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "repo"
            pi_bench_dir = project_root / "third_party" / "pi-bench"
            scripts_dir = project_root / "scripts"
            pi_bench_dir.mkdir(parents=True)
            scripts_dir.mkdir()
            (scripts_dir / "run-swe-mini.sh").write_text(
                "#!/bin/sh\n", encoding="utf-8"
            )
            self._write_swe_task(pi_bench_dir, "django__django-12209")
            manager = JobManager(
                data_dir=Path(tmp) / "data",
                project_root=project_root,
                launcher=launcher,
                run_async=False,
                pi_bench_dir=pi_bench_dir,
                protect_models=True,
                model_pin_loader=pin_model,
                model_unpinner=unpin_model,
            )

            created = manager.create_jobs(
                {
                    "suite": "swe_mini",
                    "model_ids": ["Gemma-4-26B-A4B-it-GGUF"],
                    "tasks": ["django__django-12209"],
                    "judge_model": "lemonade/gpt-oss-120b-mxfp-GGUF",
                    "openai_base_url": "https://llm.savagelands.net",
                    "swe_timeout": 45,
                    "pass_count": 2,
                    "platform": "lemonade-swe",
                    "context_window": 65536,
                    "max_output_tokens": 16384,
                    "swe_provider_timeout": 15,
                    "llamacpp_backend": "vulkan",
                }
            )
            job = manager.get_job(created[0]["id"])
            rows = manager.result_rows()
            leaderboard = manager.leaderboard_entries()
            try:
                models_json = json.loads(
                    Path(envs[0]["PI_BENCH_MODELS_JSON"]).read_text(encoding="utf-8")
                )
            except json.JSONDecodeError as exc:
                self.fail(f"invalid generated models.json: {exc}")

        self.assertEqual(job["suite"], "swe_mini")
        self.assertEqual(
            job["swe_options"]["judge_model"],
            "lemonade/gpt-oss-120b-mxfp-GGUF",
        )
        self.assertEqual(job["swe_options"]["pass_count"], 2)
        self.assertEqual(job["swe_options"]["timeout_minutes"], 45)
        self.assertEqual(job["swe_options"]["context_window"], 65536)
        self.assertEqual(job["swe_options"]["max_output_tokens"], 16384)
        self.assertEqual(job["swe_options"]["provider_timeout_minutes"], 15)
        self.assertEqual(job["swe_options"]["provider_max_retries"], 0)
        self.assertEqual(job["swe_options"]["recipe_policy"], "lemonade_unchanged")
        self.assertNotIn("requested_llamacpp_backend", job)
        self.assertEqual(
            job["result_files"], [str(Path(job["output_path"]) / "summary.json")]
        )
        self.assertEqual(
            commands[0][0], str(project_root / "scripts" / "run-swe-mini.sh")
        )
        self.assertIn("--judge-model", commands[0])
        self.assertIn("lemonade/gpt-oss-120b-mxfp-GGUF", commands[0])
        self.assertIn("--pass", commands[0])
        self.assertIn("2", commands[0])
        self.assertEqual(envs[0]["PI_BENCH_DIR"], str(pi_bench_dir))
        self.assertEqual(envs[0]["LMEVAL_WEBUI_LAUNCH_CWD"], str(project_root))
        self.assertEqual(envs[0]["LMEVAL_WEBUI_SWE_PROVIDER_TIMEOUT_MS"], "900000")
        self.assertEqual(
            envs[0]["LMEVAL_WEBUI_LEMONADE_BASE_URL"],
            "https://llm.savagelands.net",
        )
        self.assertEqual(
            envs[0]["LMEVAL_WEBUI_CANDIDATE_MODEL_ID"],
            "Gemma-4-26B-A4B-it-GGUF",
        )
        self.assertEqual(
            envs[0]["LMEVAL_WEBUI_JUDGE_MODEL_ID"],
            "gpt-oss-120b-mxfp-GGUF",
        )
        self.assertEqual(
            pin_events,
            [
                (
                    "pin",
                    "https://llm.savagelands.net",
                    "Gemma-4-26B-A4B-it-GGUF",
                ),
                (
                    "unpin",
                    "https://llm.savagelands.net",
                    "gpt-oss-120b-mxfp-GGUF",
                ),
                (
                    "unpin",
                    "https://llm.savagelands.net",
                    "Gemma-4-26B-A4B-it-GGUF",
                ),
            ],
        )
        self.assertEqual(
            models_json["providers"]["lemonade"]["baseUrl"],
            "https://llm.savagelands.net/v1",
        )
        self.assertEqual(
            [model["id"] for model in models_json["providers"]["lemonade"]["models"]],
            ["Gemma-4-26B-A4B-it-GGUF", "gpt-oss-120b-mxfp-GGUF"],
        )
        self.assertTrue(
            all(
                model["contextWindow"] == 65536 and model["maxTokens"] == 16384
                for model in models_json["providers"]["lemonade"]["models"]
            )
        )
        self.assertEqual(rows[0]["suite"], "swe_mini")
        self.assertEqual(leaderboard[0]["suite"], "swe_mini")
        self.assertEqual(leaderboard[0]["overall_score"], 100.0)

    def test_rerun_jobs_preserves_swe_mini_options(self):
        JobManager = symbol("lm_eval_webui.jobs", "JobManager")
        commands = []

        def launcher(command, env, _log_path):
            commands.append(command)
            output_path = Path(env["SWE_MINI_OUTPUT_PATH"])
            output_path.mkdir(parents=True)
            (output_path / "summary.json").write_text(
                json.dumps({"totalTasks": 0, "passedTasks": 0, "passRate": 0}),
                encoding="utf-8",
            )
            return 0

        with tempfile.TemporaryDirectory() as tmp:
            project_root = Path(tmp) / "repo"
            pi_bench_dir = project_root / "third_party" / "pi-bench"
            scripts_dir = project_root / "scripts"
            pi_bench_dir.mkdir(parents=True)
            scripts_dir.mkdir()
            (scripts_dir / "run-swe-mini.sh").write_text(
                "#!/bin/sh\n", encoding="utf-8"
            )
            self._write_swe_task(pi_bench_dir, "django__django-12209")
            manager = JobManager(
                data_dir=Path(tmp) / "data",
                project_root=project_root,
                launcher=launcher,
                run_async=False,
                pi_bench_dir=pi_bench_dir,
            )
            original = manager.create_jobs(
                {
                    "suite": "swe_mini",
                    "model_ids": ["Model-A"],
                    "tasks": ["django__django-12209"],
                    "judge_model": "lemonade/gpt-oss-120b-mxfp-GGUF",
                    "pass_count": 3,
                    "swe_timeout": 60,
                    "platform": "lemonade-swe",
                    "context_window": 65536,
                    "max_output_tokens": 16384,
                    "swe_provider_timeout": 15,
                }
            )[0]

            rerun = manager.rerun_jobs([original["id"]])[0]
            original_job = manager.get_job(original["id"])
            rerun_job = manager.get_job(rerun["id"])

        self.assertEqual(rerun_job["suite"], "swe_mini")
        self.assertEqual(rerun_job["rerun_of"], original_job["id"])
        self.assertEqual(
            rerun_job["swe_options"]["judge_model"], "lemonade/gpt-oss-120b-mxfp-GGUF"
        )
        self.assertEqual(rerun_job["swe_options"]["pass_count"], 3)
        self.assertEqual(rerun_job["swe_options"]["timeout_minutes"], 60)
        self.assertEqual(rerun_job["swe_options"]["context_window"], 65536)
        self.assertEqual(rerun_job["swe_options"]["max_output_tokens"], 16384)
        self.assertEqual(rerun_job["swe_options"]["provider_timeout_minutes"], 15)
        self.assertEqual(rerun_job["swe_options"]["provider_max_retries"], 0)
        self.assertEqual(
            rerun_job["swe_options"]["recipe_policy"], "lemonade_unchanged"
        )
        self.assertNotEqual(rerun_job["output_path"], original_job["output_path"])
        self.assertIn("--pass", commands[1])
        self.assertIn("3", commands[1])
        self.assertIn("--model-tag", commands[1])
        self.assertIn(rerun_job["id"], commands[1])


class JobManagerRerunTests(unittest.TestCase):
    def test_rerun_jobs_creates_fresh_jobs_from_saved_settings(self):
        JobManager = symbol("lm_eval_webui.jobs", "JobManager")
        commands = []

        def launcher(command, _env, _log_path):
            commands.append(command)
            return 0

        with tempfile.TemporaryDirectory() as tmp:
            manager = JobManager(
                data_dir=Path(tmp) / "data",
                project_root=Path("/repo"),
                launcher=launcher,
                run_async=False,
            )
            original = manager.create_jobs(
                {
                    "model_ids": ["Model-A"],
                    "tasks": ["gsm8k", "ifeval"],
                    "openai_base_url": "http://example.test",
                    "backend": "openai-compatible-chat-completions",
                    "llamacpp_backend": "rocm",
                    "limit": "2",
                    "num_fewshot": 3,
                    "batch_size": "4",
                    "max_gen_toks": 128,
                    "num_concurrent": 2,
                    "timeout": 45,
                    "apply_chat_template": False,
                    "fewshot_as_multiturn": True,
                    "log_samples": True,
                    "predict_only": True,
                    "task_batch_size": 25,
                }
            )[0]

            rerun = manager.rerun_jobs([original["id"]])[0]
            original_job = manager.get_job(original["id"])
            rerun_job = manager.get_job(rerun["id"])

        self.assertNotEqual(rerun_job["id"], original_job["id"])
        self.assertEqual(rerun_job["rerun_of"], original_job["id"])
        self.assertEqual(rerun_job["model_id"], "Model-A")
        self.assertEqual(rerun_job["tasks"], ["gsm8k", "ifeval"])
        self.assertEqual(rerun_job["openai_base_url"], "http://example.test")
        self.assertEqual(rerun_job["requested_llamacpp_backend"], "rocm")
        self.assertEqual(rerun_job["eval_options"]["limit"], "2")
        self.assertEqual(rerun_job["eval_options"]["num_fewshot"], 3)
        self.assertEqual(rerun_job["eval_options"]["batch_size"], "4")
        self.assertEqual(rerun_job["eval_options"]["max_gen_toks"], 128)
        self.assertEqual(rerun_job["eval_options"]["num_concurrent"], 2)
        self.assertEqual(rerun_job["eval_options"]["timeout"], 45)
        self.assertFalse(rerun_job["eval_options"]["apply_chat_template"])
        self.assertTrue(rerun_job["eval_options"]["fewshot_as_multiturn"])
        self.assertTrue(rerun_job["eval_options"]["log_samples"])
        self.assertTrue(rerun_job["eval_options"]["predict_only"])
        self.assertEqual(rerun_job["eval_options"]["task_batch_size"], 25)
        self.assertEqual(rerun_job["task_batch_size"], 25)
        self.assertEqual(len(commands), 2)
        self.assertIn("--limit", commands[1])
        self.assertIn("2", commands[1])
        self.assertIn("llamacpp_backend=rocm", commands[1])
        self.assertNotEqual(
            commands[0][commands[0].index("--output_path") + 1],
            commands[1][commands[1].index("--output_path") + 1],
        )

    def test_rerun_jobs_skips_missing_ids(self):
        JobManager = symbol("lm_eval_webui.jobs", "JobManager")

        with tempfile.TemporaryDirectory() as tmp:
            manager = JobManager(
                data_dir=Path(tmp) / "data",
                project_root=Path("/repo"),
                run_async=False,
            )

            reruns = manager.rerun_jobs(["missing-job"])

        self.assertEqual(reruns, [])

    def test_rerun_jobs_skips_jobs_without_model_or_tasks(self):
        JobManager = symbol("lm_eval_webui.jobs", "JobManager")

        with tempfile.TemporaryDirectory() as tmp:
            manager = JobManager(
                data_dir=Path(tmp) / "data",
                project_root=Path("/repo"),
                run_async=False,
            )
            (Path(tmp) / "data" / "jobs" / "legacy.json").write_text(
                json.dumps(
                    {
                        "id": "legacy",
                        "model_id": "",
                        "tasks": [],
                        "created_at": 1,
                        "updated_at": 1,
                    }
                ),
                encoding="utf-8",
            )

            reruns = manager.rerun_jobs(["legacy"])

        self.assertEqual(reruns, [])


class JobManagerConcurrencyTests(unittest.TestCase):
    def test_async_jobs_are_serialized_by_default(self):
        JobManager = symbol("lm_eval_webui.jobs", "JobManager")
        active = 0
        max_active = 0
        first_started = threading.Event()
        release = threading.Event()
        lock = threading.Lock()

        def blocking_launcher(_command, _env, log_path):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            Path(log_path).write_text("started\n", encoding="utf-8")
            first_started.set()
            release.wait(2)
            with lock:
                active -= 1
            return 0

        with tempfile.TemporaryDirectory() as tmp:
            manager = JobManager(
                data_dir=Path(tmp),
                project_root=Path("/repo"),
                launcher=blocking_launcher,
                run_async=True,
                lm_eval_python="/venv/bin/python",
            )
            manager.create_jobs(
                {"model_ids": ["Model-A", "Model-B"], "tasks": ["gsm8k"]}
            )
            self.assertTrue(first_started.wait(1))
            time.sleep(0.05)
            statuses = sorted(job["status"] for job in manager.list_jobs())
            self.assertEqual(statuses, ["queued", "running"])
            self.assertEqual(max_active, 1)
            release.set()
            deadline = time.time() + 2
            while time.time() < deadline:
                if all(job["status"] == "succeeded" for job in manager.list_jobs()):
                    break
                time.sleep(0.02)
            self.assertTrue(
                all(job["status"] == "succeeded" for job in manager.list_jobs())
            )
            self.assertEqual(max_active, 1)

    def test_max_concurrent_jobs_option_allows_parallel_launches(self):
        JobManager = symbol("lm_eval_webui.jobs", "JobManager")
        active = 0
        max_active = 0
        started = 0
        both_started = threading.Event()
        release = threading.Event()
        lock = threading.Lock()

        def blocking_launcher(_command, _env, log_path):
            nonlocal active, max_active, started
            with lock:
                active += 1
                started += 1
                max_active = max(max_active, active)
                if started == 2:
                    both_started.set()
            Path(log_path).write_text("started\n", encoding="utf-8")
            release.wait(2)
            with lock:
                active -= 1
            return 0

        with tempfile.TemporaryDirectory() as tmp:
            manager = JobManager(
                data_dir=Path(tmp),
                project_root=Path("/repo"),
                launcher=blocking_launcher,
                run_async=True,
                lm_eval_python="/venv/bin/python",
            )
            manager.create_jobs(
                {
                    "model_ids": ["Model-A", "Model-B"],
                    "tasks": ["gsm8k"],
                    "max_concurrent_jobs": 2,
                }
            )
            self.assertTrue(both_started.wait(1))
            self.assertEqual(max_active, 2)
            release.set()
            deadline = time.time() + 2
            while time.time() < deadline:
                if all(job["status"] == "succeeded" for job in manager.list_jobs()):
                    break
                time.sleep(0.02)
            self.assertTrue(
                all(job["status"] == "succeeded" for job in manager.list_jobs())
            )


class JobManagerCancellationTests(unittest.TestCase):
    def test_queued_job_can_be_cancelled_without_starting(self):
        JobManager = symbol("lm_eval_webui.jobs", "JobManager")
        first_started = threading.Event()
        release = threading.Event()
        launched_models = []

        def blocking_launcher(command, _env, log_path):
            model_args = command[command.index("--model_args") + 1 :]
            launched_models.append(
                next(
                    value.split("=", 1)[1]
                    for value in model_args
                    if value.startswith("model=")
                )
            )
            Path(log_path).write_text("started\n", encoding="utf-8")
            first_started.set()
            release.wait(2)
            return 0

        with tempfile.TemporaryDirectory() as tmp:
            manager = JobManager(
                data_dir=Path(tmp),
                project_root=Path("/repo"),
                launcher=blocking_launcher,
                run_async=True,
            )
            jobs = manager.create_jobs(
                {"model_ids": ["Model-A", "Model-B"], "tasks": ["gsm8k"]}
            )
            self.assertTrue(first_started.wait(1))

            self.assertEqual(manager.cancel_jobs([jobs[1]["id"]]), 1)
            queued_job = manager.get_job(jobs[1]["id"])
            release.set()
            deadline = time.time() + 2
            while time.time() < deadline:
                if manager.get_job(jobs[0]["id"])["status"] == "succeeded":
                    break
                time.sleep(0.02)

            self.assertEqual(queued_job["status"], "cancelled")
            self.assertEqual(launched_models, ["Model-A"])

    def test_running_default_process_is_terminated_and_marked_cancelled(self):
        JobManager = symbol("lm_eval_webui.jobs", "JobManager")

        with tempfile.TemporaryDirectory() as tmp:
            manager = JobManager(
                data_dir=Path(tmp),
                project_root=Path("/repo"),
                run_async=True,
                protect_models=False,
            )
            created = manager._create_job("Model-A", ["gsm8k"], {})
            job = manager.get_job(created["id"])
            job["command"] = [
                __import__("sys").executable,
                "-c",
                "import time; time.sleep(60)",
            ]
            job["eval_options"]["task_batch_size"] = None
            manager._write_job(job)
            manager._enqueue_job(job["id"])
            deadline = time.time() + 2
            while time.time() < deadline:
                if manager.get_job(job["id"])["status"] == "running":
                    break
                time.sleep(0.02)

            self.assertEqual(manager.cancel_jobs([job["id"]]), 1)
            deadline = time.time() + 5
            while time.time() < deadline:
                if manager.get_job(job["id"])["status"] == "cancelled":
                    break
                time.sleep(0.02)
            cancelled = manager.get_job(job["id"])

        self.assertEqual(cancelled["status"], "cancelled")
        self.assertEqual(cancelled["returncode"], -15)
        self.assertGreaterEqual(cancelled["finished_at"], cancelled["started_at"])
        self.assertGreaterEqual(cancelled["runtime_seconds"], 0)

    def test_active_job_cannot_be_cleared_or_rerun(self):
        ActiveJobError = symbol("lm_eval_webui.jobs", "ActiveJobError")
        JobManager = symbol("lm_eval_webui.jobs", "JobManager")
        started = threading.Event()
        release = threading.Event()

        def blocking_launcher(_command, _env, _log_path):
            started.set()
            release.wait(2)
            return 0

        with tempfile.TemporaryDirectory() as tmp:
            manager = JobManager(
                data_dir=Path(tmp),
                project_root=Path("/repo"),
                launcher=blocking_launcher,
                run_async=True,
            )
            job = manager.create_jobs({"model_ids": ["Model-A"], "tasks": ["gsm8k"]})[0]
            self.assertTrue(started.wait(1))

            with self.assertRaises(ActiveJobError):
                manager.clear_jobs([job["id"]])
            with self.assertRaises(ActiveJobError):
                manager.rerun_jobs([job["id"]])
            release.set()
            deadline = time.time() + 2
            while time.time() < deadline:
                if manager.get_job(job["id"])["status"] == "succeeded":
                    break
                time.sleep(0.02)
            self.assertEqual(manager.get_job(job["id"])["status"], "succeeded")

    def test_startup_marks_running_jobs_failed_and_requeues_queued_jobs(self):
        JobManager = symbol("lm_eval_webui.jobs", "JobManager")

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            jobs_dir = data_dir / "jobs"
            jobs_dir.mkdir()
            base_job = {
                "model_id": "Model-A",
                "tasks": ["gsm8k"],
                "created_at": 1,
                "updated_at": 1,
                "command": [__import__("sys").executable, "-c", "pass"],
                "output_path": str(data_dir / "runs" / "job"),
                "log_path": str(data_dir / "logs" / "job.log"),
                "telemetry_path": str(data_dir / "telemetry" / "job.jsonl"),
                "backend": "openai-compatible-chat-completions",
                "eval_options": {},
                "result_files": [],
                "returncode": None,
                "error": None,
            }
            for job_id, status, created_at in (
                ("running-job", "running", 1),
                ("queued-job", "queued", 2),
            ):
                payload = {
                    **base_job,
                    "id": job_id,
                    "status": status,
                    "created_at": created_at,
                    "output_path": str(data_dir / "runs" / job_id),
                    "log_path": str(data_dir / "logs" / f"{job_id}.log"),
                    "telemetry_path": str(data_dir / "telemetry" / f"{job_id}.jsonl"),
                }
                (jobs_dir / f"{job_id}.json").write_text(
                    json.dumps(payload), encoding="utf-8"
                )

            manager = JobManager(
                data_dir=data_dir,
                project_root=Path("/repo"),
                run_async=True,
                protect_models=False,
            )
            deadline = time.time() + 3
            while time.time() < deadline:
                if manager.get_job("queued-job")["status"] == "succeeded":
                    break
                time.sleep(0.02)
            running = manager.get_job("running-job")
            queued = manager.get_job("queued-job")

        self.assertEqual(running["status"], "failed")
        self.assertTrue(running["interrupted"])
        self.assertIn("application restart", running["error"])
        self.assertEqual(queued["status"], "succeeded")


class ResultSnapshotTests(unittest.TestCase):
    def test_result_files_are_parsed_once_and_summary_survives_restart(self):
        jobs_module = import_module("lm_eval_webui.jobs")
        JobManager = jobs_module.JobManager

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            jobs_dir = data_dir / "jobs"
            result_dir = data_dir / "runs" / "job-1"
            jobs_dir.mkdir(parents=True)
            result_dir.mkdir(parents=True)
            result_files = []
            for index, task in enumerate(("task_a", "task_b"), start=1):
                path = result_dir / f"results_{index}.json"
                path.write_text(
                    json.dumps(
                        {
                            "model_name": "Model-A",
                            "config": {"model": "openai-compatible-chat-completions"},
                            "results": {task: {"acc,none": 0.5}},
                        }
                    ),
                    encoding="utf-8",
                )
                result_files.append(str(path))
            (jobs_dir / "job-1.json").write_text(
                json.dumps(
                    {
                        "id": "job-1",
                        "model_id": "Model-A",
                        "tasks": ["task_a", "task_b"],
                        "status": "succeeded",
                        "created_at": 1,
                        "updated_at": 2,
                        "command": [],
                        "output_path": str(result_dir),
                        "log_path": str(data_dir / "logs" / "job-1.log"),
                        "result_files": result_files,
                        "returncode": 0,
                        "error": None,
                    }
                ),
                encoding="utf-8",
            )
            manager = JobManager(
                data_dir=data_dir, project_root=Path("/repo"), run_async=False
            )
            with mock.patch.object(
                jobs_module,
                "load_result_file",
                wraps=jobs_module.load_result_file,
            ) as load_result:
                snapshot = manager.results_snapshot()
                manager.result_rows()
                manager.leaderboard_entries()
            self.assertEqual(load_result.call_count, 2)
            self.assertEqual(len(snapshot["rows"]), 2)
            self.assertEqual(len(snapshot["leaderboard"]), 1)
            self.assertEqual(snapshot["rows"][0]["profile_id"], "custom")
            self.assertEqual(
                snapshot["leaderboard"][0]["benchmark_profile"]["label"],
                "Custom (legacy)",
            )
            self.assertFalse(snapshot["leaderboard"][0]["rank_eligible"])
            self.assertTrue((data_dir / "result-summaries" / "job-1.json").exists())

            restarted = JobManager(
                data_dir=data_dir, project_root=Path("/repo"), run_async=False
            )
            with mock.patch.object(
                jobs_module,
                "load_result_file",
                wraps=jobs_module.load_result_file,
            ) as load_result:
                restarted.results_snapshot()
            self.assertEqual(load_result.call_count, 0)

    def test_job_summaries_omit_commands_and_full_task_lists(self):
        JobManager = symbol("lm_eval_webui.jobs", "JobManager")

        with tempfile.TemporaryDirectory() as tmp:
            manager = JobManager(
                data_dir=Path(tmp), project_root=Path("/repo"), run_async=False
            )
            manager._create_job("Model-A", ["a", "b", "c", "d"], {})
            summary = manager.list_job_summaries()[0]

        self.assertNotIn("command", summary)
        self.assertNotIn("tasks", summary)
        self.assertNotIn("result_files", summary)
        self.assertEqual(summary["task_count"], 4)
        self.assertEqual(summary["task_preview"], ["a", "b", "c"])
        self.assertEqual(summary["benchmark_profile"]["label"], "Custom")


class JobManagerDeletionTests(unittest.TestCase):
    def test_clear_jobs_ignores_missing_empty_artifact_paths(self):
        JobManager = symbol("lm_eval_webui.jobs", "JobManager")

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            manager = JobManager(
                data_dir=data_dir, project_root=Path("/repo"), run_async=False
            )
            job = {
                "id": "legacy-job",
                "model_id": "Legacy",
                "tasks": ["gsm8k"],
                "status": "succeeded",
                "created_at": 1,
                "updated_at": 1,
                "command": [],
                "output_path": "",
                "log_path": "",
                "result_files": [],
                "returncode": 0,
                "error": None,
            }
            (data_dir / "jobs" / "legacy-job.json").write_text(
                __import__("json").dumps(job), encoding="utf-8"
            )
            sentinel = data_dir / "sentinel.txt"
            sentinel.write_text("do not delete", encoding="utf-8")

            cleared = manager.clear_jobs(["legacy-job"])

            self.assertEqual(cleared, 1)
            self.assertTrue(sentinel.exists())
            self.assertEqual(manager.list_jobs(), [])

    def test_clear_jobs_removes_only_selected_jobs(self):
        JobManager = symbol("lm_eval_webui.jobs", "JobManager")

        def fake_launcher(_command, _env, log_path):
            Path(log_path).write_text("job log\n", encoding="utf-8")
            return 0

        with tempfile.TemporaryDirectory() as tmp:
            manager = JobManager(
                data_dir=Path(tmp),
                project_root=Path("/repo"),
                launcher=fake_launcher,
                run_async=False,
                lm_eval_python="/venv/bin/python",
            )
            manager.create_jobs(
                {"model_ids": ["Model-A", "Model-B", "Model-C"], "tasks": ["gsm8k"]}
            )
            by_model = {job["model_id"]: job for job in manager.list_jobs()}
            for job in by_model.values():
                Path(job["output_path"]).mkdir(parents=True)
                Path(job["telemetry_path"]).write_text("{}\n", encoding="utf-8")

            cleared = manager.clear_jobs(
                [by_model["Model-A"]["id"], by_model["Model-C"]["id"]]
            )
            remaining = manager.list_jobs()

            self.assertEqual(cleared, 2)
            self.assertEqual([job["model_id"] for job in remaining], ["Model-B"])
            self.assertFalse(Path(by_model["Model-A"]["log_path"]).exists())
            self.assertFalse(Path(by_model["Model-A"]["output_path"]).exists())
            self.assertFalse(Path(by_model["Model-A"]["telemetry_path"]).exists())
            self.assertTrue(Path(by_model["Model-B"]["log_path"]).exists())
            self.assertTrue(Path(by_model["Model-B"]["output_path"]).exists())
            self.assertTrue(Path(by_model["Model-B"]["telemetry_path"]).exists())


class BenchmarkProfileTests(unittest.TestCase):
    def test_balanced_profiles_are_versioned_and_keep_32k_generation_limit(self):
        lm_eval_profiles = symbol("lm_eval_webui.results", "lm_eval_profiles")

        profiles = lm_eval_profiles()

        self.assertEqual(
            [profile["id"] for profile in profiles],
            [
                "strix-balanced-quick-v1",
                "strix-balanced-standard-v1",
                "strix-balanced-full-v1",
            ],
        )
        self.assertEqual(
            [profile["settings"]["limit"] for profile in profiles],
            [50, 200, None],
        )
        self.assertEqual(
            [profile["maximum_samples"] for profile in profiles],
            [400, 1600, None],
        )
        self.assertTrue(
            all(profile["settings"]["max_gen_toks"] == 32768 for profile in profiles)
        )
        self.assertTrue(all(len(profile["tasks"]) == 8 for profile in profiles))

    def test_profile_matching_ignores_task_order_and_detects_deviations(self):
        classify_lm_eval_profile = symbol(
            "lm_eval_webui.results", "classify_lm_eval_profile"
        )
        profile = symbol("lm_eval_webui.results", "lm_eval_profiles")()[0]

        matched = classify_lm_eval_profile(
            list(reversed(profile["tasks"])),
            {**profile["settings"], "limit": "50"},
        )
        changed = classify_lm_eval_profile(
            profile["tasks"],
            {**profile["settings"], "num_concurrent": 4},
        )

        self.assertEqual(matched["id"], "strix-balanced-quick-v1")
        self.assertFalse(matched["custom"])
        self.assertEqual(changed["id"], "custom")
        self.assertTrue(changed["custom"])

    def test_job_persists_server_verified_profile(self):
        JobManager = symbol("lm_eval_webui.jobs", "JobManager")
        profile = symbol("lm_eval_webui.results", "lm_eval_profiles")()[1]

        with tempfile.TemporaryDirectory() as tmp:
            manager = JobManager(
                data_dir=Path(tmp), project_root=Path("/repo"), run_async=False
            )
            created = manager._create_job(
                "Model-A", profile["tasks"], profile["settings"]
            )
            loaded = manager.get_job(created["id"])

        self.assertEqual(
            loaded["benchmark_profile"]["id"],
            "strix-balanced-standard-v1",
        )
        self.assertEqual(loaded["max_concurrent_jobs"], 1)

    def test_result_rows_include_profile_identity(self):
        extract_result_rows = symbol("lm_eval_webui.results", "extract_result_rows")
        profile = {
            "id": "strix-balanced-quick-v1",
            "label": "Quick Screen",
            "version": 1,
            "custom": False,
        }

        rows = extract_result_rows(
            "job-1",
            {"model_name": "Model-A", "results": {"gsm8k": {"acc,none": 1}}},
            benchmark_profile=profile,
            runtime_seconds=125.5,
        )

        self.assertEqual(rows[0]["profile_id"], "strix-balanced-quick-v1")
        self.assertEqual(rows[0]["profile_label"], "Quick Screen")
        self.assertEqual(rows[0]["profile_version"], 1)
        self.assertEqual(rows[0]["runtime_seconds"], 125.5)


class LeaderboardScoringTests(unittest.TestCase):
    def test_gsm8k_score_uses_flexible_extract_metric(self):
        extract_leaderboard_entry = symbol(
            "lm_eval_webui.results", "extract_leaderboard_entry"
        )
        result_json = {
            "model_name": "Model-A",
            "config": {"model": "lemonade-chat-completions", "limit": 1.0},
            "results": {
                "gsm8k": {
                    "sample_len": 1,
                    "exact_match,strict-match": 0.0,
                    "exact_match,flexible-extract": 1.0,
                    "exact_match_stderr,strict-match": "N/A",
                    "exact_match_stderr,flexible-extract": "N/A",
                }
            },
        }
        job = {"id": "job-1", "model_id": "Model-A", "status": "succeeded"}

        entry = extract_leaderboard_entry(job, result_json)

        self.assertEqual(entry["overall_score"], 100.0)
        self.assertEqual(entry["task_scores"][0]["score"], 100.0)
        self.assertEqual(
            entry["task_scores"][0]["metrics"],
            ["exact_match,flexible-extract"],
        )
        self.assertEqual(entry["category_scores"][0]["category"], "Math")
        self.assertEqual(entry["category_scores"][0]["score"], 100.0)

    def test_balanced_overall_equally_weights_categories(self):
        extract_leaderboard_entry = symbol(
            "lm_eval_webui.results", "extract_leaderboard_entry"
        )
        entry = extract_leaderboard_entry(
            {"id": "job-1", "model_id": "Model-A", "status": "succeeded"},
            {
                "model_name": "Model-A",
                "results": {
                    "mmlu_pro_biology": {"exact_match,none": 1.0},
                    "arc_challenge_chat": {"exact_match,remove_whitespace": 1.0},
                    "gsm8k": {"exact_match,flexible-extract": 0.0},
                    "ifeval": {"prompt_level_strict_acc,none": 0.5},
                },
            },
        )

        self.assertEqual(entry["overall_score"], 50.0)
        self.assertEqual(entry["score_method"], "category-balanced-v1")

    def test_result_runtime_prefers_job_wall_time_and_falls_back_to_eval_time(self):
        extract_leaderboard_entry = symbol(
            "lm_eval_webui.results", "extract_leaderboard_entry"
        )
        result_json = {
            "model_name": "Model-A",
            "total_evaluation_time_seconds": 100,
            "results": {"gsm8k": {"exact_match,flexible-extract": 1.0}},
        }

        timed = extract_leaderboard_entry(
            {
                "id": "job-1",
                "model_id": "Model-A",
                "status": "succeeded",
                "started_at": 1000,
                "finished_at": 1125.5,
            },
            result_json,
        )
        fallback = extract_leaderboard_entry(
            {"id": "legacy", "model_id": "Model-A", "status": "succeeded"},
            result_json,
        )

        self.assertEqual(timed["runtime_seconds"], 125.5)
        self.assertEqual(fallback["runtime_seconds"], 100.0)

    def test_builtin_profile_uses_canonical_metrics_and_is_rank_eligible(self):
        extract_leaderboard_entry = symbol(
            "lm_eval_webui.results", "extract_leaderboard_entry"
        )
        classify_lm_eval_profile = symbol(
            "lm_eval_webui.results", "classify_lm_eval_profile"
        )
        profile_definition = symbol("lm_eval_webui.results", "lm_eval_profiles")()[0]
        benchmark_profile = classify_lm_eval_profile(
            profile_definition["tasks"], profile_definition["settings"]
        )
        results = {
            "ifeval": {
                "prompt_level_strict_acc,none": 0.4,
                "prompt_level_loose_acc,none": 1.0,
            },
            "gsm8k": {
                "exact_match,strict-match": 0.0,
                "exact_match,flexible-extract": 0.8,
            },
            "minerva_math500": {
                "exact_match,none": 0.0,
                "math_verify,none": 0.6,
            },
            "mmlu_pro_computer_science": {"exact_match,custom-extract": 0.9},
            "mmlu_pro_engineering": {"exact_match,custom-extract": 0.7},
            "bbh_cot_zeroshot_logical_deduction_five_objects": {
                "exact_match,flexible-extract": 0.5
            },
            "arc_challenge_chat": {"exact_match,remove_whitespace": 0.7},
            "jsonschema_bench_easy": {
                "json_validity,none": 0.0,
                "schema_compliance,none": 1.0,
            },
        }
        job = {
            "id": "job-1",
            "model_id": "Model-A",
            "status": "succeeded",
            "tasks": profile_definition["tasks"],
            "benchmark_profile": benchmark_profile,
        }

        entry = extract_leaderboard_entry(
            job, {"model_name": "Model-A", "results": results}
        )

        category_scores = {
            category["category"]: category["score"]
            for category in entry["category_scores"]
        }
        self.assertEqual(category_scores["Reasoning"], 70.0)
        self.assertEqual(category_scores["Math"], 70.0)
        self.assertEqual(category_scores["Instruction Following"], 40.0)
        self.assertEqual(category_scores["Coding / Structured Output"], 100.0)
        self.assertEqual(entry["overall_score"], 70.0)
        self.assertTrue(entry["profile_complete"])
        self.assertTrue(entry["rank_eligible"])
        self.assertFalse(entry["partial"])
        by_task = {task["task"]: task for task in entry["task_scores"]}
        self.assertEqual(by_task["ifeval"]["metrics"], ["prompt_level_strict_acc,none"])
        self.assertEqual(
            by_task["jsonschema_bench_easy"]["metrics"],
            ["schema_compliance,none"],
        )

        del results["jsonschema_bench_easy"]
        incomplete = extract_leaderboard_entry(
            job, {"model_name": "Model-A", "results": results}
        )
        self.assertIsNone(incomplete["overall_score"])
        self.assertFalse(incomplete["profile_complete"])
        self.assertFalse(incomplete["rank_eligible"])
        self.assertTrue(incomplete["partial"])

    def test_failed_leaderboard_score_reports_partial_task_coverage(self):
        extract_leaderboard_entry = symbol(
            "lm_eval_webui.results", "extract_leaderboard_entry"
        )
        result_json = {
            "model_name": "Model-A",
            "results": {"gsm8k": {"exact_match,strict-match": 1.0}},
        }
        job = {
            "id": "job-1",
            "model_id": "Model-A",
            "status": "failed",
            "tasks": ["gsm8k", "ifeval"],
        }

        entry = extract_leaderboard_entry(job, result_json)

        self.assertEqual(entry["status"], "failed")
        self.assertTrue(entry["partial"])
        self.assertEqual(entry["result_task_count"], 1)
        self.assertEqual(entry["requested_task_count"], 2)
        self.assertEqual(entry["overall_score"], 100.0)

    def test_leaderboard_falls_back_to_job_backend_when_runtime_metadata_missing(self):
        extract_leaderboard_entry = symbol(
            "lm_eval_webui.results", "extract_leaderboard_entry"
        )

        entry = extract_leaderboard_entry(
            {
                "id": "job-1",
                "model_id": "Model-A",
                "status": "succeeded",
                "backend": "openai-compatible-chat-completions",
            },
            {
                "model_name": "Model-A",
                "results": {"gsm8k": {"exact_match,strict-match": 1.0}},
            },
        )

        self.assertEqual(
            entry["provider_backend"], "openai-compatible-chat-completions"
        )

    def test_leaderboard_reports_system_not_llamacpp_for_recipe_only_metadata(self):
        extract_leaderboard_entry = symbol(
            "lm_eval_webui.results", "extract_leaderboard_entry"
        )

        entry = extract_leaderboard_entry(
            {
                "id": "job-1",
                "model_id": "Model-A",
                "status": "succeeded",
                "backend": "openai-compatible-chat-completions",
                "model_metadata": {"recipe": "llamacpp"},
            },
            {
                "model_name": "Model-A",
                "results": {"gsm8k": {"exact_match,strict-match": 1.0}},
            },
        )

        self.assertEqual(entry["provider_backend"], "system")

    def test_coding_results_use_coding_category_not_other(self):
        extract_leaderboard_entry = symbol(
            "lm_eval_webui.results", "extract_leaderboard_entry"
        )
        entry = extract_leaderboard_entry(
            {"id": "job-1", "model_id": "Model-A", "status": "succeeded"},
            {
                "model_name": "Model-A",
                "results": {
                    "bigbench_simple_arithmetic_json_generate_until": {
                        "exact_match,none": 1.0,
                    },
                    "code2text_python": {
                        "smoothed_bleu_4,none": 1.25,
                    },
                    "jsonschema_bench_medium": {
                        "schema_compliance,none": 0.0,
                    },
                },
            },
        )

        categories = {score["category"] for score in entry["category_scores"]}
        self.assertIn("Coding / Structured Output", categories)
        self.assertNotIn("Other", categories)

    def test_code2text_bleu_scores_are_not_ratio_scaled(self):
        extract_leaderboard_entry = symbol(
            "lm_eval_webui.results", "extract_leaderboard_entry"
        )
        entry = extract_leaderboard_entry(
            {"id": "job-1", "model_id": "Model-A", "status": "succeeded"},
            {
                "model_name": "Model-A",
                "results": {
                    "code2text_ruby": {
                        "smoothed_bleu_4,none": 0.9130374376116006,
                    }
                },
            },
        )

        self.assertEqual(entry["task_scores"][0]["score"], 0.9130374376116006)


class ResultJsonEncodingTests(unittest.TestCase):
    def test_result_rows_skip_non_finite_metric_values(self):
        extract_result_rows = symbol("lm_eval_webui.results", "extract_result_rows")

        rows = extract_result_rows(
            "job-1",
            {
                "model_name": "Model-A",
                "results": {
                    "bbq_generate": {
                        "acc,none": 0.5,
                        "accuracy_disamb,none": math.nan,
                        "amb_bias_score,none": math.inf,
                    }
                },
            },
        )

        self.assertEqual([row["metric"] for row in rows], ["acc,none"])
        self.assertEqual(rows[0]["value"], 0.5)

    def test_leaderboard_ignores_non_finite_scores(self):
        extract_leaderboard_entry = symbol(
            "lm_eval_webui.results", "extract_leaderboard_entry"
        )

        entry = extract_leaderboard_entry(
            {"id": "job-1", "model_id": "Model-A", "status": "succeeded"},
            {
                "model_name": "Model-A",
                "results": {
                    "gsm8k": {
                        "exact_match,strict-match": math.nan,
                        "exact_match,flexible-extract": 1.0,
                    }
                },
            },
        )

        self.assertEqual(entry["overall_score"], 100.0)
        self.assertEqual(entry["task_scores"][0]["score"], 100.0)

    def test_json_responses_replace_non_finite_numbers_with_null(self):
        make_handler = symbol("lm_eval_webui.server", "make_handler")
        Handler = make_handler(object(), "static")
        handler = Handler.__new__(Handler)
        handler.headers = []
        handler.body = b""

        class Writer:
            def write(self, body):
                handler.body += body

        def send_response(self, status):
            self.status = status

        def send_header(self, name, value):
            self.headers.append((name, value))

        def end_headers(self):
            return None

        handler.wfile = Writer()
        handler.send_response = types.MethodType(send_response, handler)
        handler.send_header = types.MethodType(send_header, handler)
        handler.end_headers = types.MethodType(end_headers, handler)

        handler._json({"value": math.nan, "nested": {"rate": math.inf}})

        self.assertNotIn(b"NaN", handler.body)
        self.assertNotIn(b"Infinity", handler.body)
        try:
            payload = json.loads(handler.body)
        except json.JSONDecodeError as exc:
            self.fail(f"invalid JSON response: {exc}")
        self.assertEqual(payload, {"value": None, "nested": {"rate": None}})


class BrokenPipeResponseTests(unittest.TestCase):
    def test_write_response_ignores_disconnect_during_headers(self):
        write_response = symbol("lm_eval_webui.server", "write_response")

        class Handler:
            def __init__(self):
                self.headers = []
                self.wfile = self
                self.body = b""

            def send_response(self, status):
                self.status = status

            def send_header(self, name, value):
                self.headers.append((name, value))

            def end_headers(self):
                raise BrokenPipeError("client disconnected")

            def write(self, body):
                self.body += body

        handler = Handler()

        write_response(handler, 200, "application/json", b"{}")

        self.assertEqual(handler.status, 200)
        self.assertEqual(handler.body, b"")


class ServerEfficiencyTests(unittest.TestCase):
    def test_serialized_result_cache_is_generation_scoped_and_gzipped(self):
        JsonResponseCache = symbol("lm_eval_webui.server", "JsonResponseCache")
        cache = JsonResponseCache(max_entries=2)

        body, compressed, etag = cache.get_or_create(
            1, "results", {"rows": [{"value": "x" * 2000}]}
        )
        repeated = cache.get_or_create(1, "results", {"rows": [{"value": "different"}]})
        replacement = cache.get_or_create(2, "results", {"rows": []})

        self.assertEqual(repeated, (body, compressed, etag))
        self.assertEqual(__import__("gzip").decompress(compressed), body)
        self.assertNotEqual(replacement[2], etag)

    def test_task_catalog_load_is_shared_by_concurrent_callers(self):
        TaskCatalogCache = symbol("lm_eval_webui.server", "TaskCatalogCache")
        loader_started = threading.Event()
        release_loader = threading.Event()
        loader_calls = []
        results = []

        def loader(suite):
            loader_calls.append(suite)
            loader_started.set()
            release_loader.wait(2)
            return [{"name": "gsm8k"}]

        catalog = TaskCatalogCache(loader)
        threads = [
            threading.Thread(target=lambda: results.append(catalog.get("lm_eval")))
            for _index in range(2)
        ]
        threads[0].start()
        self.assertTrue(loader_started.wait(1))
        threads[1].start()
        release_loader.set()
        for thread in threads:
            thread.join(2)

        self.assertTrue(all(not thread.is_alive() for thread in threads))
        self.assertEqual(loader_calls, ["lm_eval"])
        self.assertEqual(results, [[{"name": "gsm8k"}], [{"name": "gsm8k"}]])
        self.assertIs(catalog.get("lm_eval"), results[0])

    def test_compact_leaderboard_omits_per_task_payloads(self):
        compact_leaderboard_entry = symbol(
            "lm_eval_webui.server", "compact_leaderboard_entry"
        )

        compact = compact_leaderboard_entry(
            {
                "job_id": "job-1",
                "task_scores": [{"task": "a", "score": 1}],
                "category_scores": [
                    {"category": "Math", "score": 100, "tasks": ["a", "b"]}
                ],
            }
        )

        self.assertNotIn("task_scores", compact)
        self.assertNotIn("tasks", compact["category_scores"][0])
        self.assertEqual(compact["category_scores"][0]["task_count"], 2)

    def test_results_handler_paginates_and_filters_by_suite(self):
        make_handler = symbol("lm_eval_webui.server", "make_handler")

        class Manager:
            def results_snapshot(self):
                return {
                    "version": 3,
                    "rows": [
                        {"job_id": "a", "suite": "swe_mini"},
                        {"job_id": "b"},
                        {"job_id": "c"},
                    ],
                    "leaderboard": [],
                }

        Handler = make_handler(Manager(), "static")
        handler = Handler.__new__(Handler)
        captured = {}

        def cached_json(self, payload, generation, cache_key):
            captured.update(payload=payload, generation=generation, cache_key=cache_key)

        handler._cached_json = types.MethodType(cached_json, handler)
        handler._handle_results("suite=lm_eval&offset=1&limit=1")

        self.assertEqual(captured["generation"], 3)
        self.assertEqual(captured["payload"]["rows"], [{"job_id": "c"}])
        self.assertIsNone(captured["payload"]["next_offset"])
        self.assertEqual(captured["payload"]["total"], 2)


class SmokeTests(unittest.TestCase):
    def test_github_workflow_pins_actions_and_does_not_persist_credentials(self):
        workflow = Path(".github/workflows/docker-image.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn("persist-credentials: false", workflow)
        self.assertIn(
            "group: docker-${{ github.workflow }}-${{ github.ref }}", workflow
        )
        self.assertIn("cancel-in-progress: true", workflow)
        self.assertNotRegex(workflow, r"uses: [^\n]+@v\d")
        self.assertRegex(workflow, r"uses: actions/checkout@[0-9a-f]{40}")
        self.assertRegex(workflow, r"uses: docker/login-action@[0-9a-f]{40}")
        self.assertRegex(workflow, r"uses: docker/build-push-action@[0-9a-f]{40}")

    def test_kubernetes_manifest_uses_statefulset_with_data_volume(self):
        statefulset = Path("deploy/k8s/statefulset.yaml").read_text(encoding="utf-8")

        self.assertIn("kind: StatefulSet", statefulset)
        self.assertIn("serviceName: lm-eval-webui", statefulset)
        self.assertIn("name: data", statefulset)
        self.assertIn("claimName: lm-eval-data", statefulset)
        self.assertIn("mountPath: /data", statefulset)
        self.assertFalse(Path("deploy/k8s/deployment.yaml").exists())

    def test_kubernetes_manifest_limits_webui_privilege_escalation(self):
        statefulset = Path("deploy/k8s/statefulset.yaml").read_text(encoding="utf-8")
        webui_container = statefulset[
            statefulset.index("        - name: webui") : statefulset.index(
                "        - name: docker"
            )
        ]

        self.assertIn("securityContext:", webui_container)
        self.assertIn("allowPrivilegeEscalation: false", webui_container)
        self.assertIn("capabilities:", webui_container)
        self.assertIn("drop:", webui_container)
        self.assertIn("- ALL", webui_container)
        self.assertNotIn("privileged: true", webui_container)

    def test_kubernetes_manifest_sets_webui_memory_limit(self):
        statefulset = Path("deploy/k8s/statefulset.yaml").read_text(encoding="utf-8")
        webui_container = statefulset[
            statefulset.index("        - name: webui") : statefulset.index(
                "        - name: docker"
            )
        ]

        self.assertIn("resources:", webui_container)
        self.assertIn("requests:", webui_container)
        self.assertIn("memory: 2Gi", webui_container)
        self.assertIn("limits:", webui_container)
        self.assertIn("memory: 12Gi", webui_container)

    def test_kubernetes_manifest_persists_huggingface_cache_on_data_volume(self):
        statefulset = Path("deploy/k8s/statefulset.yaml").read_text(encoding="utf-8")

        self.assertIn("name: HF_HOME", statefulset)
        self.assertIn("value: /data/huggingface", statefulset)
        self.assertIn("name: HF_DATASETS_CACHE", statefulset)
        self.assertIn("value: /data/huggingface/datasets", statefulset)
        self.assertIn("name: LMEVAL_WEBUI_HF_RETRIES", statefulset)
        self.assertIn('value: "5"', statefulset)
        self.assertIn("name: LMEVAL_WEBUI_HF_RETRY_DELAY", statefulset)
        self.assertIn('value: "10"', statefulset)
        self.assertIn("name: LMEVAL_WEBUI_HF_RETRY_MAX_DELAY", statefulset)
        self.assertIn('value: "120"', statefulset)

    def test_kubernetes_manifest_supports_optional_huggingface_token(self):
        statefulset = Path("deploy/k8s/statefulset.yaml").read_text(encoding="utf-8")
        secret_example = Path("deploy/k8s/huggingface-secret.example.yaml").read_text(
            encoding="utf-8"
        )

        self.assertIn("name: HF_TOKEN", statefulset)
        self.assertIn("secretKeyRef:", statefulset)
        self.assertIn("name: huggingface-token", statefulset)
        self.assertIn("key: token", statefulset)
        self.assertIn("optional: true", statefulset)
        self.assertIn("name: HUGGING_FACE_HUB_TOKEN", statefulset)
        self.assertIn("name: huggingface-token", secret_example)
        self.assertIn("token:", secret_example)

    def test_job_log_css_cannot_force_page_horizontal_scroll(self):
        styles = Path("static/styles.css").read_text(encoding="utf-8")
        log_rule = styles[
            styles.index(".log {") : styles.index("}\n", styles.index(".log {"))
        ]

        self.assertIn("max-width: 100%", log_rule)
        self.assertIn("min-width: 0", log_rule)
        self.assertIn("overflow-wrap: anywhere", log_rule)
        self.assertIn(".badge.progress.live::before", styles)
        self.assertIn("job-activity-pulse", styles)
        self.assertIn("prefers-reduced-motion: reduce", styles)

    def test_static_ui_defaults_to_full_quality_benchmarks(self):
        index = Path("static/index.html").read_text(encoding="utf-8")
        script = Path("static/app.js").read_text(encoding="utf-8")
        limit_control = index[
            index.index('id="limit"') - 50 : index.index('id="limit"') + 100
        ]
        fewshot_control = index[
            index.index('id="numFewshot"') - 80 : index.index('id="numFewshot"') + 120
        ]
        swe_context_control = index[
            index.index('id="sweContextWindow"') - 80 : index.index(
                'id="sweContextWindow"'
            )
            + 160
        ]

        self.assertNotIn("value=", limit_control)
        self.assertNotIn("value=", fewshot_control)
        self.assertIn('id="maxGenToks" type="number" value="32768"', index)
        self.assertIn('id="timeout" type="number" value="7200"', index)
        self.assertIn('id="sweTimeout" type="number" min="1" value="60"', index)
        self.assertIn('value="65536"', swe_context_control)
        self.assertIn('id="sweMaxOutputTokens"', index)
        self.assertIn('value="16384"', index)
        self.assertIn('id="sweProviderTimeout"', index)
        self.assertIn('value="15"', index)
        self.assertIn('id="sweProviderRetries" type="number" value="0" disabled', index)
        self.assertIn('id="sweRecipePolicy" type="checkbox" checked disabled', index)
        self.assertIn("const DEFAULT_SWE_TIMEOUT_MINUTES = 60", script)
        self.assertIn("const DEFAULT_SWE_CONTEXT_WINDOW = 65536", script)
        self.assertIn("const DEFAULT_SWE_MAX_OUTPUT_TOKENS = 16384", script)
        self.assertIn('recipe_policy: "lemonade_unchanged"', script)
        self.assertIn('if (suite === "lm_eval")', script)
        self.assertIn("Limit (blank = all)", index)
        self.assertIn("Few-shot (blank = task default)", index)

    def test_static_ui_exposes_lemonade_bench_as_first_suite(self):
        index = Path("static/index.html").read_text(encoding="utf-8")
        script = Path("static/app.js").read_text(encoding="utf-8")
        dockerfile = Path("deploy/Dockerfile").read_text(encoding="utf-8")

        self.assertLess(
            index.index('id="suiteLemonadeBench"'), index.index('id="suiteLmEval"')
        )
        self.assertLess(
            index.index('id="suiteLmEval"'), index.index('id="suiteSweMini"')
        )
        self.assertLess(
            index.index('id="leaderboardLemonadeBench"'),
            index.index('id="leaderboardLmEval"'),
        )
        self.assertLess(
            index.index('id="leaderboardLmEval"'),
            index.index('id="leaderboardSweMini"'),
        )
        self.assertIn('id="lemonadeBenchOptions"', index)
        self.assertIn('id="benchBackends"', index)
        self.assertIn("blank = each model's configured backend", index)
        self.assertIn('id="benchContextSizes"', index)
        self.assertIn('id="benchRuns" type="number" min="1" value="3"', index)
        self.assertIn('id="benchWarmup" type="number" min="0" value="0"', index)
        self.assertIn('id="benchTimeout" type="number" min="1" value="1800"', index)
        self.assertIn('id="benchMemoryTracking" type="checkbox" checked', index)
        self.assertIn('id="benchReloadBetweenRuns" type="checkbox" checked', index)
        self.assertIn('activeSuite: "lemonade_bench"', script)
        self.assertIn('resultSuite: "lemonade_bench"', script)
        self.assertIn("function renderLemonadeBenchLeaderboard", script)
        self.assertIn("const DEFAULT_LEMONADE_BENCH_TIMEOUT = 1800", script)
        self.assertIn("lemonade_model_backends: configuredModelBackends", script)
        self.assertIn('model.recipe === "llamacpp"', script)
        self.assertIn(
            'renderLeaderboardTable(list, rows, columns, "lemonade_bench")', script
        )
        self.assertIn("function resultConfigurationLabel", script)
        self.assertIn("LEMONADE_CLI_VERSION=11.6.0", dockerfile)
        self.assertIn("LEMONADE_CLI_SHA256_AMD64=", dockerfile)
        self.assertIn("LEMONADE_CLI_SHA256_ARM64=", dockerfile)
        self.assertIn("ARG TARGETARCH\n", dockerfile)
        self.assertNotIn("ARG TARGETARCH=", dockerfile)
        self.assertIn('case "${TARGETARCH}"', dockerfile)

    def test_static_ui_exposes_balanced_profile_controls_and_results(self):
        index = Path("static/index.html").read_text(encoding="utf-8")
        script = Path("static/app.js").read_text(encoding="utf-8")
        styles = Path("static/styles.css").read_text(encoding="utf-8")
        server = Path("lm_eval_webui/server.py").read_text(encoding="utf-8")

        self.assertIn('id="lmEvalProfilePicker"', index)
        self.assertIn('id="lmEvalProfileButtons"', index)
        self.assertIn('id="activeBenchmarkProfile"', index)
        self.assertIn('id="resultProfileFilter"', index)
        self.assertIn('id="leaderboardDescription"', index)
        self.assertIn("Balanced Overall", script)
        self.assertIn("function applyBenchmarkProfile", script)
        self.assertIn("function activeBenchmarkProfile", script)
        self.assertIn("function renderResultProfileFilter", script)
        self.assertIn('"Balanced Overall"', script)
        self.assertIn('"Profile"', script)
        self.assertIn("rank_eligible", script)
        self.assertIn('"Runtime"', script)
        self.assertIn("function completedJobRuntime", script)
        self.assertIn("function formatRuntimeSeconds", script)
        self.assertIn("Runtime ${runtime}", script)
        self.assertIn(".profile-picker", styles)
        self.assertIn(".runtime-cell", styles)
        self.assertIn(".badge.profile", styles)
        self.assertIn('"benchmark_profiles": lm_eval_profiles()', server)

    def test_leaderboard_columns_are_sortable(self):
        index = Path("static/index.html").read_text(encoding="utf-8")
        script = Path("static/app.js").read_text(encoding="utf-8")
        styles = Path("static/styles.css").read_text(encoding="utf-8")

        self.assertIn("Select any", index)
        self.assertIn("column heading to sort", index)
        self.assertIn("leaderboardSort: {}", script)
        self.assertIn("function renderLeaderboardTable", script)
        self.assertIn("function sortLeaderboardRows", script)
        self.assertIn("function compareLeaderboardValues", script)
        self.assertIn('button.className = "leaderboard-sort"', script)
        self.assertIn('"aria-sort"', script)
        self.assertIn('renderLeaderboardTable(list, rows, columns, "lm_eval")', script)
        self.assertIn('renderLeaderboardTable(list, rows, columns, "swe_mini")', script)
        prompt_rate_column = script.index('key: "prompt-rate"')
        generation_rate_column = script.index('key: "generation-rate"')
        self.assertLess(prompt_rate_column, generation_rate_column)
        self.assertIn('label: "Prompt tok/s"', script)
        self.assertIn("numberOrNull(row.entry.prompt_tok_s)", script)
        self.assertIn("formatRate(row.entry.prompt_tok_s)", script)
        self.assertEqual(script.count('label: "Backend"'), 3)
        self.assertEqual(script.count('sortLabel: "Runtime backend"'), 3)
        self.assertIn('label: "Overall"', script)
        self.assertIn('sortLabel: "Balanced Overall"', script)
        self.assertIn(
            'label: category === "Coding / Structured Output" ? "Coding" : category',
            script,
        )
        self.assertIn("sortLabel: category", script)
        self.assertIn(".leaderboard-sort {", styles)
        self.assertIn(".leaderboard-sort:focus-visible", styles)

    def test_static_ui_uses_centered_overflow_safe_page_layout(self):
        styles = Path("static/styles.css").read_text(encoding="utf-8")

        self.assertIn("--page-content-width: 2000px", styles)
        self.assertIn("--page-gutter: 16px", styles)
        self.assertIn("overflow-x: clip", styles)
        self.assertIn("margin-inline: auto", styles)
        self.assertIn("padding-inline: var(--page-gutter)", styles)
        self.assertIn("main > *,\n.card {\n\tmin-width: 0", styles)
        self.assertIn(
            ".leaderboard,\n.table-wrap {\n\tmax-width: 100%;\n\tmin-width: 0",
            styles,
        )

    def test_detailed_results_have_cascading_multi_select_filters(self):
        index = Path("static/index.html").read_text(encoding="utf-8")
        script = Path("static/app.js").read_text(encoding="utf-8")
        styles = Path("static/styles.css").read_text(encoding="utf-8")

        for control_id in (
            "detailModelSummary",
            "detailModelOptions",
            "detailTaskSummary",
            "detailTaskOptions",
            "detailMetricSummary",
            "detailMetricOptions",
        ):
            self.assertIn(f'id="{control_id}"', index)
        self.assertIn('aria-label="Models tested"', index)
        self.assertIn('aria-label="Scenarios or tasks run"', index)
        self.assertIn('aria-label="Metrics for selected tasks"', index)
        self.assertNotIn('id="metricSelect"', index)
        self.assertIn("detailFilters: new Map()", script)
        self.assertIn("function handleDetailFilterChange", script)
        self.assertIn("function renderDetailFilter", script)
        self.assertIn('allLabel: "All models"', script)
        self.assertIn('allLabel: "All tasks"', script)
        self.assertIn('allLabel: "All metrics"', script)
        self.assertIn("metricsInitialized: false", script)
        self.assertIn('checkbox.dataset.selectAll = "true"', script)
        self.assertIn("const metricRows = suiteRows.filter", script)
        self.assertIn("filterState.tasks.has(String(row.task))", script)
        self.assertIn("filterState.models.has(String(row.model))", script)
        self.assertIn("filterState.metrics.has(String(row.metric))", script)
        self.assertIn("function appendMetricChart", script)
        self.assertNotIn('$("metricSelect")', script)
        self.assertIn(".detail-filter-grid {", styles)
        self.assertIn(".detail-filter-menu {", styles)
        self.assertIn(".detail-filter-option.all {", styles)

    def test_detailed_chart_measures_labels_before_drawing_bars(self):
        script = Path("static/app.js").read_text(encoding="utf-8")
        styles = Path("static/styles.css").read_text(encoding="utf-8")

        self.assertIn("function svgTextWidth", script)
        self.assertIn("text.getComputedTextLength()", script)
        self.assertIn("const longestLabelWidth", script)
        self.assertIn("const barStart = Math.ceil(10 + longestLabelWidth + 24)", script)
        self.assertIn("svgRect(barStart", script)
        self.assertIn("svg.style.width = `${width}px`", script)
        self.assertIn(".chart {\n\tmax-width: 100%;\n\toverflow-x: auto", styles)
        self.assertIn("font-family: inherit", styles)
        self.assertIn("font-size: 0.86rem", styles)
        self.assertIn("overflow-x: auto", styles)

    def test_task_catalog_request_uses_extended_timeout(self):
        script = Path("static/app.js").read_text(encoding="utf-8")

        self.assertIn("const TASK_REQUEST_TIMEOUT_MS = 120000", script)
        self.assertIn("timeoutMs: TASK_REQUEST_TIMEOUT_MS", script)

    def test_all_model_smoke_script_is_safe_by_default(self):
        script = Path("scripts/smoke-all-models.py").read_text(encoding="utf-8")

        self.assertIn('SMOKE_TASKS = ("gsm8k",', script)
        self.assertIn('"mmlu_abstract_algebra_generative"', script)
        self.assertIn('"ifeval"', script)
        self.assertIn('"log_samples": True', script)
        self.assertIn('"max_concurrent_jobs": 1', script)
        self.assertIn("without this flag the script is read-only", script)
        script_symbols = __import__("runpy").run_path("scripts/smoke-all-models.py")
        is_chat_model = script_symbols["is_chat_model"]
        self.assertTrue(is_chat_model({"labels": ["llm", "reasoning"]}))
        self.assertFalse(is_chat_model({"labels": ["tts"]}))
        self.assertFalse(is_chat_model({"labels": ["transcription", "hot"]}))

    def test_static_ui_exposes_selected_job_controls(self):
        index = Path("static/index.html").read_text(encoding="utf-8")
        script = Path("static/app.js").read_text(encoding="utf-8")

        self.assertIn('id="cancelSelectedJobs"', index)
        self.assertIn('id="clearSelectedJobs"', index)
        self.assertIn('id="selectAllJobs"', index)
        self.assertIn("Select all", index)
        self.assertIn("jobs</label", index)
        self.assertIn('id="selectedJobCount"', index)
        self.assertIn('id="maxConcurrentJobs"', index)
        self.assertIn('id="llamacppBackend"', index)
        self.assertIn("Model runtime options", index)
        self.assertIn("Benchmark options", index)
        self.assertIn('value="vulkan"', index)
        self.assertIn('value="rocm"', index)
        self.assertIn('id="hideGatedTasks"', index)
        self.assertIn("gated</label", index)
        self.assertIn('id="taskViewMode"', index)
        self.assertIn('value="leaves" selected', index)
        self.assertIn('value="groups"', index)
        self.assertIn("Groups / tags", index)
        list_actions = index[
            index.index('class="row list-actions"') : index.index('id="taskSpinner"')
        ]
        self.assertIn('id="taskViewMode"', list_actions)
        task_filter_rows = index[
            index.index('class="row task-filters"') : index.index('id="taskHint"')
        ]
        self.assertNotIn('id="taskViewMode"', task_filter_rows)
        self.assertNotIn('id="leafTasksOnly"', index)
        self.assertNotIn("leafTasksOnly", script)
        self.assertIn('id="hideNonEnglishTasks"', index)
        self.assertIn("non-English</label", index)
        self.assertIn("hideNonEnglishTasks", script)
        self.assertIn('task.language_scope === "non_english"', script)
        self.assertIn("taskViewMode", script)
        self.assertIn('taskViewMode === "leaves"', script)
        self.assertIn('taskViewMode === "groups"', script)
        self.assertIn('(task.kind || "task") !== "task"', script)
        self.assertIn('(task.kind || "task") === "task"', script)
        self.assertIn("function pruneSelectedTasksForViewMode", script)
        self.assertIn("state.selectedTasks = new Set", script)
        self.assertIn('id="suiteSweMini"', index)
        self.assertIn('id="sweJudgeModel"', index)
        self.assertIn("gpt-oss-120b-mxfp-GGUF", index)
        self.assertIn('id="sweMiniJudgeHint"', index)
        self.assertIn("DEFAULT_SWE_JUDGE_MODEL", script)
        self.assertIn("function renderSweJudgeModels", script)
        self.assertIn("const suite = state.activeSuite", script)
        self.assertIn("kindBadge(task.kind)", script)
        self.assertIn('"Status"', script)
        self.assertIn('"Tasks"', script)
        self.assertIn("formatTaskCoverage", script)
        self.assertIn("entry.status", script)
        self.assertNotIn('id="hideUnknownTasks"', index)
        self.assertNotIn("hideUnknownTasks", script)
        self.assertIn('value="1"', index)
        task_batch_control = index[
            index.index('id="taskBatchSize"') - 80 : index.index('id="taskBatchSize"')
            + 80
        ]
        self.assertIn('id="taskBatchSize"', task_batch_control)
        self.assertIn('value="1"', task_batch_control)
        self.assertIn("task_batch_size", script)
        self.assertIn("taskBatchSize", script)
        self.assertIn("Task batch size", script)
        self.assertIn("batch_progress", script)
        self.assertIn("request_progress", script)
        self.assertIn("Current task batch", script)
        self.assertIn("Current batch requests", script)
        self.assertIn("Completed task batches", script)
        self.assertIn("Model protection", script)
        self.assertIn("Lemonade Benchmark WebUI", index)
        self.assertNotIn("Local lm-eval Benchmark WebUI", index)
        self.assertIn("OpenAI-compatible base URL", index)
        self.assertIn('id="openaiBaseUrl"', index)
        self.assertIn('value="http://localhost:11434/v1"', index)
        self.assertIn("async function loadConfig", script)
        self.assertIn('api("/api/config")', script)
        self.assertIn("await loadConfig()", script)
        self.assertNotIn('id="lemonadeUrl"', index)
        self.assertIn("selectedJobs", script)
        self.assertIn("visibleTaskNames", script)
        self.assertIn('id="selectVisibleTasks"', index)
        self.assertIn("function selectVisibleTasks", script)
        self.assertIn("job-select", script)
        self.assertIn("job-summary-actions", script)
        self.assertIn("function progressBadge", script)
        self.assertIn("function progressText", script)
        self.assertIn("function requestProgressFromLog", script)
        self.assertIn("function activeJobElapsed", script)
        self.assertIn("includeActive: true", script)
        self.assertIn("Live job activity", script)
        self.assertIn("summaryActions.append(progress)", script)
        self.assertIn('button("Rerun", "job-rerun")', script)
        self.assertIn("function rerunJobs", script)
        self.assertIn("displayJudgeModel(row.entry.judge_model)", script)
        self.assertIn('replace(/^lemonade\\//, "")', script)
        self.assertIn(
            'leaderboardCell(row.modelName, "model-cell", row.modelName)', script
        )
        self.assertIn("summaryActions.append(", script)
        self.assertIn('checkbox.addEventListener("click"', script)
        self.assertIn("job-details", script)
        self.assertIn("job-summary", script)
        self.assertIn("job-expanded", script)
        self.assertNotIn("job-expanded-header", script)
        self.assertIn("job-task-list", script)
        self.assertIn("(job.tasks || []).forEach", script)
        self.assertIn("function loadJobDetails", script)
        self.assertIn("jobDetails", script)
        self.assertIn("expandedJobs", script)
        self.assertIn("details.open = state.expandedJobs.has(job.id)", script)
        self.assertIn('details.addEventListener("toggle"', script)
        self.assertIn("selectAllJobs", script)
        self.assertIn("function toggleAllJobs", script)
        self.assertIn("function syncSelectAllJobs", script)
        self.assertIn("cancelSelectedJobs", script)
        self.assertIn("function cancelJobs", script)
        self.assertIn("/api/jobs/cancel", script)
        self.assertIn("clearSelectedJobs", script)
        self.assertIn("rerunSelectedJobs", script)
        self.assertIn('id="rerunSelectedJobs"', index)
        self.assertIn("/api/jobs/rerun", script)
        self.assertIn("function rerunSelectedJobs", script)
        self.assertIn("max_concurrent_jobs", script)
        self.assertIn("llamacpp_backend", script)
        self.assertIn("llamacppBackend", script)
        self.assertIn("openai_base_url", script)
        self.assertIn("openaiBaseUrl", script)
        self.assertIn("function modelForEntry", script)
        self.assertIn("runtime_backend", script)
        self.assertIn("entry.backend", script)
        self.assertIn("function isClientBackend", script)
        self.assertIn("isClientBackend(backend)", script)
        self.assertIn("model?.llamacpp_backend", script)
        self.assertNotIn("specificRuntimeBackend(model?.recipe)", script)
        self.assertIn("Other", script)
        self.assertIn("categoryBadge", script)
        self.assertNotIn("compatibility: ${compatibility}", script)
        self.assertIn("task.category", script)
        self.assertIn('task.compatibility === "gated"', script)
        self.assertIn("hideGatedTasks", script)
        self.assertIn("Jobs", index)
        self.assertIn("<summary>Jobs", index)
        self.assertIn("Could not load results", script)
        self.assertIn("setTaskLoading", script)
        self.assertNotIn('id="jobLog"', index)
        self.assertNotIn("selectedJobId", script)
        self.assertIn("function loadJobLog", script)
        self.assertIn("function loadExpandedJobLogs", script)
        self.assertIn("jobLogElements", script)
        self.assertIn('log.className = "log job-log"', script)
        self.assertIn("log.dataset.jobId = summaryJob.id", script)
        self.assertIn("function shouldAutoScrollLog", script)
        self.assertIn("function scrollLogToBottom", script)
        self.assertIn("/log${query}", script)
        self.assertIn("shouldAutoScrollLog(log)", script)
        self.assertIn("scrollLogToBottom(log)", script)
        self.assertIn("spinner", index)
        styles = Path("static/styles.css").read_text(encoding="utf-8")
        self.assertIn("selector-panel", styles)
        self.assertIn("list-header", styles)
        self.assertIn("spinner", styles)
        self.assertIn("job-summary", styles)
        self.assertIn("job-expanded", styles)
        self.assertNotIn("job-expanded-header", styles)
        self.assertIn("job-task-list", styles)
        self.assertIn("job-summary-actions", styles)
        self.assertIn("job-log-label", styles)
        self.assertIn("job-log", styles)
        self.assertNotIn(".job-row:has(.job-details:not([open]))", styles)
        server = Path("lm_eval_webui/server.py").read_text(encoding="utf-8")
        self.assertIn("Cache-Control", server)
        self.assertIn("BoundedThreadPoolHTTPServer", server)
        self.assertIn("/api/jobs/rerun", server)
        self.assertIn("/api/jobs/cancel", server)
        self.assertIn("/api/leaderboard", server)
        self.assertIn("/api/config", server)
        self.assertIn('"openai_base_url": openai_base_url', server)
        self.assertIn("no-store", server)
        self.assertIn("BrokenPipeError", server)
        self.assertIn("Content-Encoding", server)
        self.assertNotIn("setInterval(", script)
        self.assertIn("function singleFlight", script)
        self.assertIn("function scheduleJobPoll", script)
        self.assertIn('id="resultDetails"', index)

    def test_job_suite_filter_scopes_bulk_selection_to_visible_jobs(self):
        index = Path("static/index.html").read_text(encoding="utf-8")
        script = Path("static/app.js").read_text(encoding="utf-8")
        styles = Path("static/styles.css").read_text(encoding="utf-8")

        filter_markup = index[
            index.index('id="jobSuiteFilter"') : index.index('id="visibleJobCount"')
        ]
        self.assertLess(
            filter_markup.index('value="all"'),
            filter_markup.index('value="lemonade_bench"'),
        )
        self.assertLess(
            filter_markup.index('value="lemonade_bench"'),
            filter_markup.index('value="lm_eval"'),
        )
        self.assertLess(
            filter_markup.index('value="lm_eval"'),
            filter_markup.index('value="swe_mini"'),
        )
        self.assertIn("All jobs", filter_markup)
        normalized_index = " ".join(index.split())
        self.assertIn("Select all visible jobs", normalized_index)
        self.assertIn('jobSuiteFilter: "all"', script)
        self.assertIn("function visibleJobs()", script)
        self.assertIn("function selectedVisibleJobs()", script)
        self.assertIn("jobSuite(job) === state.jobSuiteFilter", script)
        self.assertIn('$("jobSuiteFilter").addEventListener("change"', script)

        toggle_start = script.index("function toggleAllJobs()")
        toggle_end = script.index("async function loadJobLog", toggle_start)
        toggle = script[toggle_start:toggle_end]
        self.assertIn("const jobs = visibleJobs()", toggle)
        self.assertIn("state.selectedJobs.add(job.id)", toggle)
        self.assertIn("state.selectedJobs.delete(job.id)", toggle)
        self.assertNotIn("state.jobs.map", toggle)

        for function_name in (
            "cancelSelectedJobs",
            "clearSelectedJobs",
            "rerunSelectedJobs",
        ):
            function_start = script.index(f"async function {function_name}()")
            function_end = script.index("\n}", function_start)
            self.assertIn("selectedVisibleJobs()", script[function_start:function_end])
        self.assertIn(".job-suite-filter", styles)

    def test_requirements_include_libra_scoring_dependency(self):
        requirements = Path("requirements.txt").read_text(encoding="utf-8")

        self.assertIn("pymorphy3", requirements)

    def test_static_ui_exposes_task_category_filters(self):
        index = Path("static/index.html").read_text(encoding="utf-8")
        script = Path("static/app.js").read_text(encoding="utf-8")

        for checkbox_id, label in (
            ("taskCategoryReasoning", "Reasoning"),
            ("taskCategoryMath", "Math"),
            ("taskCategoryCoding", "Coding / Structured Output"),
            ("taskCategoryInstruction", "Instruction Following"),
            ("taskCategoryOther", "Other"),
        ):
            with self.subTest(checkbox_id=checkbox_id):
                self.assertIn(f'id="{checkbox_id}"', index)
                if label == "Coding / Structured Output":
                    self.assertIn("Coding /", index)
                    self.assertIn("Structured Output", index)
                else:
                    self.assertIn(label, index)
                self.assertIn(checkbox_id, script)

        self.assertLess(
            index.index('id="taskCategoryReasoning"'),
            index.index('id="hideIncompatibleTasks"'),
        )
        self.assertIn("TASK_CATEGORY_FILTERS", script)
        self.assertIn("function selectedTaskCategories", script)
        self.assertIn("selectedCategories.has", script)

    def test_static_ui_exposes_visible_task_bulk_controls(self):
        index = Path("static/index.html").read_text(encoding="utf-8")
        script = Path("static/app.js").read_text(encoding="utf-8")

        self.assertIn('id="selectVisibleTasks"', index)
        self.assertIn('id="unselectVisibleTasks"', index)
        self.assertIn("Select visible", index)
        self.assertIn("Unselect visible", index)
        self.assertIn("function selectVisibleTasks", script)
        self.assertIn("function unselectVisibleTasks", script)
        self.assertIn("state.selectedTasks.delete(taskName)", script)
        self.assertIn("hasAutoSelectedTask", script)
        self.assertIn("!state.hasAutoSelectedTask", script)
        self.assertIn(
            '$("selectVisibleTasks").addEventListener("click", selectVisibleTasks)',
            script,
        )
        self.assertIn(
            '$("unselectVisibleTasks").addEventListener("click", unselectVisibleTasks)',
            script,
        )

    def test_common_tasks_have_categories(self):
        common_tasks = symbol("lm_eval_webui.server", "COMMON_TASKS")
        by_name = {task["name"]: task for task in common_tasks}

        self.assertEqual(by_name["gsm8k"]["category"], "Math")
        self.assertEqual(by_name["ifeval"]["category"], "Instruction Following")
        self.assertEqual(by_name["truthfulqa_gen"]["category"], "Reasoning")


if __name__ == "__main__":
    unittest.main()
