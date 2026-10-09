import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import check_wind_env as env


class EnvironmentTests(unittest.TestCase):
    def test_missing_torch(self):
        with patch.object(env.importlib, "import_module", side_effect=
                          ModuleNotFoundError("No torch", name="torch")):
            self.assertEqual(env.probe_torch()["status"], "TORCH_NOT_INSTALLED")

    def test_missing_torch_dependency_requires_review(self):
        with patch.object(env.importlib, "import_module", side_effect=
                          ModuleNotFoundError("No dependency", name="dependency")):
            self.assertEqual(env.probe_torch()["status"], "NEEDS_REVIEW")

    def test_native_import_error_is_reported(self):
        with patch.object(env.importlib, "import_module", side_effect=OSError("DLL failure")):
            report = env.probe_torch()
            self.assertEqual(report["status"], "NEEDS_REVIEW")
            self.assertIn("DLL failure", report["error"])

    def test_cuda_unavailable(self):
        torch = SimpleNamespace(__version__="test", version=SimpleNamespace(cuda=None),
                                cuda=SimpleNamespace(is_available=lambda: False))
        with patch.object(env.importlib, "import_module", return_value=torch):
            self.assertEqual(env.probe_torch()["status"], "CUDA_UNAVAILABLE")

    def test_invalid_visible_device(self):
        torch = SimpleNamespace(
            __version__="test", version=SimpleNamespace(cuda="test"),
            cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1),
        )
        with patch.object(env.importlib, "import_module", return_value=torch):
            report = env.probe_torch(1)
            self.assertEqual(report["status"], "NEEDS_REVIEW")
            self.assertIn("out of range", report["error"])

    def test_command_timeout(self):
        with patch.object(env.subprocess, "run", side_effect=
                          subprocess.TimeoutExpired(["probe"], 1)):
            self.assertIn("TimeoutExpired", env.run_command(["probe"], 1)["error"])

    def test_missing_command(self):
        with patch.object(env.subprocess, "run", side_effect=FileNotFoundError("missing")):
            self.assertIn("FileNotFoundError", env.run_command(["missing"])["error"])

    def test_crashed_torch_subprocess(self):
        with patch.object(env, "run_command", return_value={"returncode": -11}):
            report = env.isolated_torch_probe(0, 1)
            self.assertEqual(report["status"], "NEEDS_REVIEW")
            self.assertEqual(report["process"]["returncode"], -11)


if __name__ == "__main__":
    unittest.main()
