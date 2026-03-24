#!/usr/bin/env python3
"""Unit tests for utils/latest_run.py (no pytest required)."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestLatestRunSnapshot(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.td = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def test_empty_when_no_artifacts(self):
        from utils import latest_run

        with patch.object(latest_run, "get_results_dir", return_value=self.td), patch.object(
            latest_run, "_optimized_json_search_dirs", return_value=[self.td]
        ):
            out = latest_run.get_latest_run_snapshot()
        self.assertEqual(out["status"], "empty")
        self.assertIn("message", out)
        self.assertIsNone(out["latest_csv"])
        self.assertIsNone(out["latest_json"])

    def test_json_latest_by_mtime(self):
        from utils import latest_run

        older = self.td / "optimized_results_20240101-120000.json"
        newer = self.td / "optimized_results_20240201-120000.json"
        older.write_text(json.dumps({"ranking": [{"name": "A", "pr_auc": 0.5}]}))
        newer.write_text(json.dumps({"ranking": [{"name": "B", "pr_auc": 0.9}]}))
        os.utime(older, (100, 100))
        os.utime(newer, (200, 200))

        with patch.object(latest_run, "get_results_dir", return_value=self.td), patch.object(
            latest_run, "_optimized_json_search_dirs", return_value=[self.td]
        ):
            out = latest_run.get_latest_run_snapshot()
        self.assertEqual(out["status"], "ok")
        assert out["latest_json"] is not None
        self.assertEqual(out["latest_json"]["path"], str(newer.resolve()))
        self.assertEqual(out["latest_json"]["ranking"][0]["name"], "B")

    def test_csv_row(self):
        from utils import latest_run

        csv_path = self.td / "latest_run.csv"
        csv_path.write_text("pr_auc,accuracy\n0.79,0.74\n")

        with patch.object(latest_run, "get_results_dir", return_value=self.td), patch.object(
            latest_run, "_optimized_json_search_dirs", return_value=[self.td]
        ):
            out = latest_run.get_latest_run_snapshot()
        self.assertEqual(out["status"], "ok")
        assert out["latest_csv"] is not None
        self.assertEqual(out["latest_csv"]["row"]["pr_auc"], "0.79")


if __name__ == "__main__":
    unittest.main()
