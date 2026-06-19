import tempfile
import unittest
from importlib import import_module
from pathlib import Path


def symbol(module_name, attribute):
    return import_module(module_name).__dict__[attribute]


class JobManagerDeletionTests(unittest.TestCase):
    def test_clear_jobs_ignores_missing_empty_artifact_paths(self):
        JobManager = symbol("lm_eval_webui.jobs", "JobManager")

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp)
            manager = JobManager(data_dir=data_dir, project_root=Path("/repo"), run_async=False)
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


class SmokeTests(unittest.TestCase):
    def test_static_ui_exposes_selected_job_controls(self):
        index = Path("static/index.html").read_text(encoding="utf-8")
        script = Path("static/app.js").read_text(encoding="utf-8")

        self.assertIn('id="clearSelectedJobs"', index)
        self.assertIn('id="selectedJobCount"', index)
        self.assertIn("selectedJobs", script)
        self.assertIn("job-select", script)
        self.assertIn("clearSelectedJobs", script)


if __name__ == "__main__":
    unittest.main()
