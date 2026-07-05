import json
import math
import os
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


class SweMiniRunnerTests(unittest.TestCase):
    def test_swe_mini_command_uses_repo_owned_wrapper_for_codex_judge(self):
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
                    judge_model="openai-codex/gpt-5.5",
                    model_tag="job123",
                    platform="lemonade-swe",
                    timeout_minutes=45,
                    pass_count=2,
                    context_window=131072,
                    require_pi_auth=True,
                )
            )
            models_path_exists = Path(env["PI_BENCH_MODELS_JSON"]).is_file()

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
        self.assertIn("openai-codex/gpt-5.5", command)
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
        self.assertEqual(env["PI_BENCH_REQUIRE_PI_AUTH"], "1")
        self.assertEqual(env["SWE_MINI_OUTPUT_PATH"], str(output_path))
        self.assertEqual(env["LMEVAL_WEBUI_LAUNCH_CWD"], str(project_root))
        self.assertEqual(env["PI_BENCH_DIR"], str(pi_bench_dir))
        self.assertTrue(models_path_exists)

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
            )
            try:
                payload = json.loads(Path(models_path).read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                self.fail(f"invalid generated models.json: {exc}")

        lemonade = payload["providers"]["lemonade"]
        self.assertEqual(lemonade["baseUrl"], "https://llm.savagelands.net/v1")
        self.assertEqual(lemonade["api"], "openai-completions")
        self.assertEqual(lemonade["apiKey"], "lemonade")
        self.assertEqual(lemonade["models"][0]["id"], "Gemma-4-26B-A4B-it-GGUF")
        self.assertEqual(lemonade["models"][0]["contextWindow"], 131072)

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
            "provider_backend": "rocm",
            "swe_options": {
                "judge_model": "openai-codex/gpt-5.5",
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
        self.assertEqual(entry["suite"], "swe_mini")
        self.assertEqual(entry["overall_score"], 50.0)
        self.assertEqual(entry["total_tasks"], 2)
        self.assertEqual(entry["passed_tasks"], 1)
        self.assertEqual(entry["judge_model"], "openai-codex/gpt-5.5")
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

    def test_parse_generations_preserves_empty_choice_responses(self):
        OpenAICompatibleChatCompletion = symbol(
            "lm_eval_webui.lemonade_model", "OpenAICompatibleChatCompletion"
        )

        generations = OpenAICompatibleChatCompletion.parse_generations(
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

    def test_wrapper_fails_fast_when_docker_run_produces_no_result(self):
        script = Path("scripts/run-swe-mini.sh").read_text(encoding="utf-8")

        self.assertIn("No result file produced", script)
        self.assertIn('exit "$EXIT_CODE"', script)


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
        self.assertEqual(len(leaderboard), 1)
        self.assertEqual(
            sorted(score["task"] for score in leaderboard[0]["task_scores"]),
            ["task_a", "task_b", "task_c", "task_d", "task_e"],
        )

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

    def test_swe_mini_job_uses_suite_command_and_parses_summary(self):
        JobManager = symbol("lm_eval_webui.jobs", "JobManager")
        commands = []
        envs = []

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
            )

            created = manager.create_jobs(
                {
                    "suite": "swe_mini",
                    "model_ids": ["Gemma-4-26B-A4B-it-GGUF"],
                    "tasks": ["django__django-12209"],
                    "judge_model": "openai-codex/gpt-5.5",
                    "openai_base_url": "https://llm.savagelands.net",
                    "swe_timeout": 45,
                    "pass_count": 2,
                    "platform": "lemonade-swe",
                    "context_window": 131072,
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
        self.assertEqual(job["swe_options"]["judge_model"], "openai-codex/gpt-5.5")
        self.assertEqual(job["swe_options"]["pass_count"], 2)
        self.assertEqual(job["swe_options"]["timeout_minutes"], 45)
        self.assertEqual(
            job["result_files"], [str(Path(job["output_path"]) / "summary.json")]
        )
        self.assertEqual(
            commands[0][0], str(project_root / "scripts" / "run-swe-mini.sh")
        )
        self.assertIn("--judge-model", commands[0])
        self.assertIn("openai-codex/gpt-5.5", commands[0])
        self.assertIn("--pass", commands[0])
        self.assertIn("2", commands[0])
        self.assertEqual(envs[0]["PI_BENCH_REQUIRE_PI_AUTH"], "1")
        self.assertEqual(envs[0]["PI_BENCH_DIR"], str(pi_bench_dir))
        self.assertEqual(envs[0]["LMEVAL_WEBUI_LAUNCH_CWD"], str(project_root))
        self.assertEqual(
            models_json["providers"]["lemonade"]["baseUrl"],
            "https://llm.savagelands.net/v1",
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
                    "judge_model": "openai-codex/gpt-5.5",
                    "pass_count": 3,
                    "swe_timeout": 60,
                    "platform": "lemonade-swe",
                    "require_pi_auth": True,
                }
            )[0]

            rerun = manager.rerun_jobs([original["id"]])[0]
            original_job = manager.get_job(original["id"])
            rerun_job = manager.get_job(rerun["id"])

        self.assertEqual(rerun_job["suite"], "swe_mini")
        self.assertEqual(rerun_job["rerun_of"], original_job["id"])
        self.assertEqual(
            rerun_job["swe_options"]["judge_model"], "openai-codex/gpt-5.5"
        )
        self.assertEqual(rerun_job["swe_options"]["pass_count"], 3)
        self.assertEqual(rerun_job["swe_options"]["timeout_minutes"], 60)
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


class SmokeTests(unittest.TestCase):
    def test_github_workflow_pins_actions_and_does_not_persist_credentials(self):
        workflow = Path(".github/workflows/docker-image.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn("persist-credentials: false", workflow)
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

    def test_static_ui_exposes_selected_job_controls(self):
        index = Path("static/index.html").read_text(encoding="utf-8")
        script = Path("static/app.js").read_text(encoding="utf-8")

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
        self.assertIn("openai-codex/gpt-5.5", index)
        self.assertIn('id="sweUsePiAuth"', index)
        self.assertIn("suite: state.activeSuite", script)
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
        self.assertIn("Lemonade lm-eval Benchmark WebUI", index)
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
        self.assertIn(
            "summaryActions.append(suiteBadge(job), statusBadge(job), checkbox)", script
        )
        self.assertIn('checkbox.addEventListener("click"', script)
        self.assertIn("job-details", script)
        self.assertIn("job-summary", script)
        self.assertIn("job-expanded", script)
        self.assertNotIn("job-expanded-header", script)
        self.assertIn("job-task-list", script)
        self.assertIn("job.tasks.forEach", script)
        self.assertIn("expandedJobs", script)
        self.assertIn("details.open = state.expandedJobs.has(job.id)", script)
        self.assertIn('details.addEventListener("toggle"', script)
        self.assertIn("selectAllJobs", script)
        self.assertIn("function toggleAllJobs", script)
        self.assertIn("function syncSelectAllJobs", script)
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
        self.assertIn("function loadSelectedJobLog", script)
        self.assertIn("function shouldAutoScrollLog", script)
        self.assertIn("function scrollLogToBottom", script)
        self.assertIn("loadSelectedJobLog()", script)
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
        self.assertNotIn(".job-row:has(.job-details:not([open]))", styles)
        server = Path("lm_eval_webui/server.py").read_text(encoding="utf-8")
        self.assertIn("Cache-Control", server)
        self.assertIn("/api/jobs/rerun", server)
        self.assertIn("/api/config", server)
        self.assertIn('"openai_base_url": openai_base_url', server)
        self.assertIn("no-store", server)
        self.assertIn("BrokenPipeError", server)

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
