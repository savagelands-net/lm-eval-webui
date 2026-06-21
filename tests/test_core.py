import json
import math
import tempfile
import threading
import time
import types
import unittest
from importlib import import_module
from pathlib import Path
from typing import Any


def symbol(module_name, attribute):
    return import_module(module_name).__dict__[attribute]


class OpenAICompatibleEndpointTests(unittest.TestCase):
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


class LemonadeModelTests(unittest.TestCase):
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
        self.assertEqual(output["timings"]["time_to_headers_s"], 1.0)
        self.assertEqual(output["timings"]["time_to_first_event_s"], 2.0)
        self.assertEqual(output["timings"]["ttft_s"], 3.0)
        self.assertEqual(output["timings"]["predicted_n"], 1)
        self.assertEqual(output["usage"], {"completion_tokens": 2})

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
        self.assertEqual(metadata["device"], "gpu")


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
            lambda _path: """
task: truthfulqa_va
dataset_path: gplsi/truthfulqa_va
output_type: generate_until
""",
        )

        self.assertEqual(truthfulqa_task["compatibility"], "gated")

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
        config_text = """
task: niah_single_1
dataset_path: ""
output_type: generate_until
"""

        task = annotate_task_compatibility(
            {"name": "niah_single_1", "description": "niah_single_1.yaml"},
            lambda _path: config_text,
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

    def test_slow_unverified_tasks_are_marked_unknown(self):
        annotate_task_compatibility = symbol(
            "lm_eval_webui.server", "annotate_task_compatibility"
        )
        for task_name in (
            "meddialog_qsumm",
            "graphwalks_128k",
            "graphwalks_1M",
            "tinyGSM8k",
        ):
            with self.subTest(task_name=task_name):
                config_text = f"""
task: {task_name}
dataset_path: lighteval/med_dialog
output_type: generate_until
"""

                task = annotate_task_compatibility(
                    {"name": task_name, "description": f"{task_name}.yaml"},
                    lambda _path, text=config_text: text,
                )

                self.assertEqual(task["compatibility"], "unknown")


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


class LeaderboardScoringTests(unittest.TestCase):
    def test_gsm8k_score_averages_strict_and_flexible_extract_metrics(self):
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

        self.assertEqual(entry["overall_score"], 50.0)
        self.assertEqual(entry["task_scores"][0]["score"], 50.0)
        self.assertEqual(
            entry["task_scores"][0]["metrics"],
            ["exact_match,strict-match", "exact_match,flexible-extract"],
        )
        self.assertEqual(entry["category_scores"][0]["category"], "Math")
        self.assertEqual(entry["category_scores"][0]["score"], 50.0)


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
        self.assertEqual(
            json.loads(handler.body), {"value": None, "nested": {"rate": None}}
        )


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


class SmokeTests(unittest.TestCase):
    def test_static_ui_exposes_selected_job_controls(self):
        index = Path("static/index.html").read_text(encoding="utf-8")
        script = Path("static/app.js").read_text(encoding="utf-8")

        self.assertIn('id="clearSelectedJobs"', index)
        self.assertIn('id="selectAllJobs"', index)
        self.assertIn('Select all jobs', index)
        self.assertIn('id="selectedJobCount"', index)
        self.assertIn('id="maxConcurrentJobs"', index)
        self.assertIn('id="llamacppBackend"', index)
        self.assertIn("Model runtime options", index)
        self.assertIn("Benchmark options", index)
        self.assertIn('value="vulkan"', index)
        self.assertIn('value="rocm"', index)
        self.assertIn('id="hideGatedTasks"', index)
        self.assertIn("gated</label", index)
        self.assertIn('value="1"', index)
        self.assertIn("lm-eval Benchmark WebUI", index)
        self.assertNotIn("Local lm-eval Benchmark WebUI", index)
        self.assertIn("OpenAI-compatible base URL", index)
        self.assertIn('id="openaiBaseUrl"', index)
        self.assertNotIn('id="lemonadeUrl"', index)
        self.assertIn("selectedJobs", script)
        self.assertIn("visibleTaskNames", script)
        self.assertIn('id="selectVisibleTasks"', index)
        self.assertIn("function selectVisibleTasks", script)
        self.assertIn("job-select", script)
        self.assertIn("selectAllJobs", script)
        self.assertIn("function toggleAllJobs", script)
        self.assertIn("function syncSelectAllJobs", script)
        self.assertIn("clearSelectedJobs", script)
        self.assertIn("max_concurrent_jobs", script)
        self.assertIn("llamacpp_backend", script)
        self.assertIn("llamacppBackend", script)
        self.assertIn("openai_base_url", script)
        self.assertIn("openaiBaseUrl", script)
        self.assertIn("function modelForEntry", script)
        self.assertIn("runtime_backend", script)
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
        self.assertIn("spinner", index)
        styles = Path("static/styles.css").read_text(encoding="utf-8")
        self.assertIn("selector-panel", styles)
        self.assertIn("list-header", styles)
        self.assertIn("spinner", styles)
        server = Path("lm_eval_webui/server.py").read_text(encoding="utf-8")
        self.assertIn("Cache-Control", server)
        self.assertIn("no-store", server)
        self.assertIn("BrokenPipeError", server)

    def test_common_tasks_have_categories(self):
        common_tasks = symbol("lm_eval_webui.server", "COMMON_TASKS")
        by_name = {task["name"]: task for task in common_tasks}

        self.assertEqual(by_name["gsm8k"]["category"], "Math")
        self.assertEqual(by_name["ifeval"]["category"], "Instruction Following")
        self.assertEqual(by_name["truthfulqa_gen"]["category"], "Reasoning")


if __name__ == "__main__":
    unittest.main()
