"""Offline tests for Colab video model setup and mobile notebook paths."""
import json
import ast
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from studio import video


class VideoTests(unittest.TestCase):
    def test_catalog_exposes_multiple_engines(self):
        self.assertGreaterEqual(len(video.VIDEO_PRESETS), 5)
        self.assertEqual(video.info("Wan 2.2 TI2V 5B · ComfyUI")["backend"], "ComfyUI")
        self.assertEqual(video.info("Wan 2.1 1.3B · WanGP")["backend"], "WanGP")
        self.assertTrue(all(x["recommended_vram"] > 0 for x in video.VIDEO_PRESETS.values()))
        with self.assertRaises(ValueError):
            video.info("invalid")

    def test_official_wan22_files_and_workflow(self):
        self.assertEqual(len(video.WAN22_FILES), 3)
        names = [v[0] for v in video.WAN22_FILES]
        self.assertTrue(any("diffusion_models" in x for x in names))
        self.assertTrue(any("text_encoders" in x for x in names))
        self.assertTrue(any("/vae/" in x for x in names))
        self.assertIn("workflow_templates", video.WORKFLOW_URL)

    def test_wangp_command_localhost_safe_and_drive_folders(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            app = root / "app"
            app.mkdir()
            py = root / "venv" / "bin" / "python"
            outputs = root / "videos"
            process = MagicMock()
            process.pid = 12345
            with patch.object(video, "WAN_PYTHON", py.parents[1]), \
                 patch("studio.launcher.LOCAL_ROOT", root), \
                 patch.object(video.subprocess, "Popen", return_value=process) as popen, \
                 patch("studio.runtime.save_pid") as save:
                port, log = video.launch_video_ui(
                    "Wan 2.2 T2V / I2V · WanGP", app, outputs, py)
            self.assertEqual(port, 7862)
            self.assertEqual(log.name, "wangp-server.log")
            self.assertIn("--server-name", popen.call_args.args[0])
            self.assertIn("127.0.0.1", popen.call_args.args[0])
            self.assertIn("--workspaces-dir", popen.call_args.args[0])
            self.assertIn(str(outputs / "workspaces"), popen.call_args.args[0])
            self.assertIn("--profile", popen.call_args.args[0])
            self.assertIn("4", popen.call_args.args[0])
            save.assert_called_once_with("server", 12345)

    def test_comfy_video_runs_without_looking_up_image_checkpoint(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            process = MagicMock()
            process.pid = 12346
            with patch("studio.launcher.LOCAL_ROOT", root), \
                 patch.object(video.subprocess, "Popen", return_value=process) as popen, \
                 patch("studio.runtime.save_pid"):
                port, _ = video.launch_video_ui(
                    "Wan 2.2 TI2V 5B · ComfyUI", root, root / "videos")
            self.assertEqual(port, 8188)
            self.assertEqual(popen.call_args.args[0][1], "main.py")
            self.assertIn("--output-directory", popen.call_args.args[0])

    def test_drive_storage_symlinks_and_preserves_existing_files(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "WanGP" / "ckpts"
            src.mkdir(parents=True)
            (src / "previous.safetensors").write_text("preserve")
            target = root / "MyDrive" / "video-cache" / "ckpts"
            video.connect_storage(src, target)
            self.assertTrue(src.is_symlink())
            self.assertEqual((target / "previous.safetensors").read_text(), "preserve")
            video.connect_storage(src, target)
            self.assertEqual((src / "previous.safetensors").read_text(), "preserve")

    def test_drive_storage_does_not_clobber_conflicting_file(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "src"
            dest = root / "dest"
            src.mkdir()
            dest.mkdir()
            (src / "model").write_text("new")
            (dest / "model").write_text("old")
            with self.assertRaises(FileExistsError):
                video.connect_storage(src, dest)
            self.assertEqual((dest / "model").read_text(), "old")

    def test_both_notebook_files_parse_and_mobile_has_forms(self):
        root = Path(__file__).resolve().parents[1]
        mobile = json.loads((root / "Anime_Studio_Mobile.ipynb").read_text())
        desktop = json.loads((root / "Anime_Studio_Colab.ipynb").read_text())
        self.assertEqual(mobile["nbformat"], 4)
        self.assertEqual(desktop["nbformat"], 4)
        code = "\n".join("".join(c.get("source", [])) for c in mobile["cells"])
        self.assertIn("# @param", code)
        self.assertIn("runtime.launch_worker", code)
        self.assertIn("monitor_active_session_async", code)
        self.assertNotIn("ipywidgets", code)
        self.assertIn("video_preset", code)
        for cell in mobile["cells"]:
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                if source.lstrip().startswith("%pip"):
                    continue
                # Parse Python portions only; magic cells are never compiled.
                if "%pip" not in source:
                    compile(source, "<mobile-cell>", "exec",
                            flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
