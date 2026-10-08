import contextlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


class DemoExitTests(unittest.TestCase):
    def load_demo(self):
        path = Path(__file__).resolve().parents[1] / "demo.py"
        spec = importlib.util.spec_from_file_location("casino_demo_test", path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"xgboost": Mock()}):
            spec.loader.exec_module(module)
        return module

    def test_model_section_failure_returns_nonzero(self):
        module = self.load_demo()
        with (
            patch.object(module, "tennis_demo", side_effect=ValueError("invalid model")),
            patch.object(module, "mlb_demo") as mlb,
            patch.object(module, "wc_demo") as wc,
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            self.assertEqual(module.main(), 1)
        mlb.assert_called_once()
        wc.assert_called_once()
        self.assertIn("Failed to run 1 model sections", output.getvalue())
        self.assertNotIn("Done.", output.getvalue())

    def test_all_sections_success_returns_zero(self):
        module = self.load_demo()
        with (
            patch.object(module, "tennis_demo"),
            patch.object(module, "mlb_demo"),
            patch.object(module, "wc_demo"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            self.assertEqual(module.main(), 0)


if __name__ == "__main__":
    unittest.main()
