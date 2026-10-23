"""Static/offline smoke tests; no GPU, tunnel, or large download required."""
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

colab = types.ModuleType("google.colab")
colab.drive = types.SimpleNamespace(mount=lambda *_: None)
colab.userdata = types.SimpleNamespace(get=lambda *_: None)
sys.modules["google.colab"] = colab

PATH = Path(__file__).resolve().parents[1] / "studio" / "launcher.py"
spec = importlib.util.spec_from_file_location("anime_studio_launcher", PATH)
studio = importlib.util.module_from_spec(spec)
spec.loader.exec_module(studio)


class StudioTests(unittest.TestCase):
    def test_models_have_safetensors_and_unique_names(self):
        self.assertGreaterEqual(len(studio.MODELS), 5)
        for name, (repo, file, family) in studio.MODELS.items():
            self.assertTrue(file.endswith(".safetensors"), name)
            self.assertEqual(repo.count("/"), 1)
            self.assertIn(family, ("pony", "illustrious", "noob", "sdxl"))

    def test_lora_model_family_filter(self):
        self.assertTrue(any("pony" in v[2] for v in studio.LORAS.values()))
        self.assertTrue(any("illustrious" in v[2] for v in studio.LORAS.values()))

    def test_custom_lora_urls(self):
        self.assertEqual(studio.parse_custom_lora("hf://alice/repo/my.safetensors"),
                         ("hf", "alice/repo|my.safetensors"))
        self.assertEqual(studio.parse_custom_lora(
            "https://huggingface.co/alice/models/resolve/main/sub/test.safetensors"),
            ("hf", "alice/models|sub/test.safetensors|main"))
        with self.assertRaises(ValueError):
            studio.parse_custom_lora("http://127.0.0.1/admin.safetensors")
        with self.assertRaises(ValueError):
            studio.parse_custom_lora("https://example.com/model.safetensors")

    def test_workflow_multiple_loras_and_pony(self):
        names = ["a.safetensors", "b.safetensors", "c.safetensors"]
        for pony in (False, True):
            wf = studio.comfy_workflow("checkpoint.safetensors", names, "Balanced", pony=pony)
            self.assertEqual(sum(n["type"] == "LoraLoader" for n in wf["nodes"]), 3)
            self.assertEqual(sum(n["type"] == "CLIPSetLastLayer" for n in wf["nodes"]), int(pony))
            self.assertEqual(wf["last_node_id"], len(wf["nodes"]))
            self.assertEqual(wf["last_link_id"], len(wf["links"]))
            json.dumps(wf)
            by_id = {n["id"]: n for n in wf["nodes"]}
            for link_id, source, output, dest, input_index, datatype in wf["links"]:
                self.assertIn(link_id, by_id[source]["outputs"][output]["links"])
                self.assertEqual(by_id[dest]["inputs"][input_index]["link"], link_id)

    def test_settings_round_trip_and_no_unknown_fields(self):
        with tempfile.TemporaryDirectory() as td:
            settings = Path(td) / "settings.json"
            with patch.object(studio, "SETTINGS", settings):
                data = {**studio.DEFAULTS, "model": "Pony Diffusion V6 XL"}
                studio.save_settings(data)
                self.assertEqual(studio.load_settings()["model"], data["model"])
                self.assertNotIn("NGROK_AUTHTOKEN", settings.read_text())

    def test_lossless_png_setting_and_link(self):
        with tempfile.TemporaryDirectory() as td:
            app = Path(td)
            src = app / "model.safetensors"
            src.write_bytes(b"sample")
            dest = app / "models" / "checkpoints" / src.name
            studio.safe_link(src, dest)
            studio.safe_link(src, dest)
            self.assertTrue(dest.is_symlink())
            studio.write_a1111_settings(app, app / "Drive" / "outputs", "Balanced")
            config = json.loads((app / "config.json").read_text())
            self.assertEqual(config["samples_format"], "png")
            self.assertIn("Drive", config["outdir_txt2img_samples"])


    def test_runtime_status_and_log_visibility(self):
        from studio import runtime
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with patch.object(runtime, "STATUS", root / "status.json"):
                with patch.object(runtime, "LOG", root / "runtime.log"):
                    runtime.stage("installing", "Descargando dependencias")
                    current = runtime.read_json(runtime.STATUS)
                    self.assertEqual(current["stage"], "installing")
                    self.assertEqual(current["percent"], 18)
                    runtime.LOG.write_text("checking GPU\\npip install started\\n")
                    self.assertIn("pip install", runtime.recent_log())
                    self.assertIn("pip install", runtime.render_log())


    def test_detached_worker_bootstrap_reports_failure(self):
        import os
        import subprocess
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "runtime-config.json").write_text(json.dumps({
                "model": "INVALID-MODEL", "ui": "ComfyUI", "tunnel": "Local only"
            }))
            env = dict(os.environ, ANIME_STUDIO_RUNTIME_DIR=str(root))
            worker = Path(__file__).resolve().parents[1] / "studio" / "worker.py"
            result = subprocess.run([sys.executable, "-u", str(worker)],
                                    env=env, text=True, capture_output=True, timeout=35)
            self.assertIn("[BOOT]", result.stdout)
            self.assertIn("[BOOTSTRAP]", result.stdout)
            status = json.loads((root / "runtime-status.json").read_text())
            self.assertEqual(status["stage"], "error")
            self.assertIn("Modelo desconocido", status["message"])

    def test_stale_boot_detected(self):
        from studio import runtime
        from datetime import datetime, timezone, timedelta
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            state_file = root / "status.json"
            pids_file = root / "pids.json"
            state_file.write_text(json.dumps({
                "stage": "preparing",
                "updated_at": (datetime.now(timezone.utc) - timedelta(seconds=50)).isoformat()
            }))
            pids_file.write_text('{"worker": 12345}')
            with patch.object(runtime, "STATUS", state_file), patch.object(runtime, "PIDS", pids_file):
                with patch.object(runtime, "process_active", return_value=True):
                    out = runtime.snapshot()
                    self.assertEqual(out["stage"], "stalled")

    def test_notebook_parse_and_entrypoint(self):
        nb = json.loads((PATH.parent.parent / "Anime_Studio_Colab.ipynb").read_text())
        self.assertEqual(nb["nbformat"], 4)
        cells = nb["cells"]
        self.assertEqual(len(cells), 6)
        self.assertIn("studio.dashboard", "".join(cells[2]["source"]))
        self.assertIn("enable_custom_widget_manager", "".join(cells[2]["source"]))
        self.assertIn("runtime.recent_log", "".join(cells[3]["source"]))
        self.assertIn("launch_foreground", "".join(cells[5]["source"]))


if __name__ == "__main__":
    unittest.main()
