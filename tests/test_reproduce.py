import contextlib
import io
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import explore


class ReproductionTests(unittest.TestCase):
    def test_child_failure_stops_before_dependent_reports(self):
        results = [
            subprocess.CompletedProcess([], 0, stdout="done\n", stderr=""),
            subprocess.CompletedProcess([], 7, stdout="", stderr="fixture failed\n"),
        ]
        with patch("explore.subprocess.run", side_effect=results) as run:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                with self.assertRaises(subprocess.CalledProcessError) as error:
                    explore.item_reproduce()
        self.assertEqual(run.call_count, 2)
        self.assertEqual(error.exception.returncode, 7)
        self.assertNotIn("rebuilt", output.getvalue())
        self.assertNotIn("Completed all five", output.getvalue())

    def test_empty_failure_output_still_returns_nonzero(self):
        result = subprocess.CompletedProcess([], 3, stdout="", stderr="")
        with patch("explore.subprocess.run", return_value=result) as run:
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                code = explore.main(["explore.py", "7"])
        self.assertEqual(code, 3)
        self.assertEqual(run.call_count, 1)

    def test_success_reports_completed_scripts(self):
        result = subprocess.CompletedProcess([], 0, stdout="done\n", stderr="")
        with patch("explore.subprocess.run", return_value=result) as run:
            with contextlib.redirect_stdout(io.StringIO()) as output:
                code = explore.main(["explore.py", "7"])
        self.assertEqual(code, 0)
        self.assertEqual(run.call_count, 5)
        self.assertIn("Completed all five analysis scripts.", output.getvalue())

    def test_child_scripts_read_and_write_unicode_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root / "casino_sim"
            scripts.mkdir()
            (root / "input.txt").write_text("Probability ≥ 50%", encoding="utf-8")
            for name in (
                "house_backtest",
                "house_backtest_mlb",
                "house_backtest_tennis",
                "book_compare",
                "three_model_summary",
            ):
                (scripts / f"{name}.py").write_text(
                    "from pathlib import Path\n"
                    "text = Path('input.txt').read_text()\n"
                    f"Path('{name}.txt').write_text(text)\n"
                    "print(text)\n",
                    encoding="utf-8",
                )
            with patch("explore.HERE", directory):
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(explore.main(["explore.py", "7"]), 0)
            reports = list(root.glob("*.txt"))
            self.assertEqual(len(reports), 6)
            self.assertTrue(
                all(
                    report.read_text(encoding="utf-8") == "Probability ≥ 50%"
                    for report in reports
                )
            )
            self.assertIn("Probability ≥ 50%", output.getvalue())


if __name__ == "__main__":
    unittest.main()
