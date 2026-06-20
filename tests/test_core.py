import tempfile
import threading
import time
import unittest
from importlib import import_module
from pathlib import Path


def symbol(module_name, attribute):
    return import_module(module_name).__dict__[attribute]


class LemonadeModelTests(unittest.TestCase):
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
        config_text = """
task: anagrams1
dataset_path: EleutherAI/unscramble
dataset_name: mid_word_1_anagrams
output_type: generate_until
"""

        task = annotate_task_compatibility(
            {"name": "anagrams1", "description": "anagrams1.yaml"},
            lambda _path: config_text,
        )

        self.assertEqual(task["compatibility"], "incompatible")


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
        self.assertIn('id="selectedJobCount"', index)
        self.assertIn('id="maxConcurrentJobs"', index)
        self.assertIn('value="1"', index)
        self.assertIn("selectedJobs", script)
        self.assertIn("visibleTaskNames", script)
        self.assertIn('id="selectVisibleTasks"', index)
        self.assertIn("function selectVisibleTasks", script)
        self.assertIn("job-select", script)
        self.assertIn("clearSelectedJobs", script)
        self.assertIn("max_concurrent_jobs", script)
        self.assertIn("function modelForEntry", script)
        self.assertIn("runtime_backend", script)
        self.assertIn("Other", script)
        self.assertIn("categoryBadge", script)
        self.assertNotIn("compatibility: ${compatibility}", script)
        self.assertIn("task.category", script)
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
