from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory
import unittest

from experiments_v2.adapters.base import run_logged


class ProcessMonitoringTests(unittest.TestCase):
    def _record(self, root: Path) -> dict:
        return json.loads((root / "process.json").read_text(encoding="utf-8"))

    def test_success_streams_output_and_records_lifecycle(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            log_path = root / "extraction.log"
            code = "\n".join(
                (
                    "import sys",
                    "print('Downloading model')",
                    "print('Applied providers: CPUExecutionProvider')",
                    "print('Extracting interaction')",
                    "print('stderr is streamed too', file=sys.stderr)",
                )
            )
            streamed = io.StringIO()
            with contextlib.redirect_stdout(streamed):
                result = run_logged(
                    [sys.executable, "-c", code], cwd=root, log_path=log_path
                )

            record = self._record(root)
            states = [event["state"] for event in record["states"]]
            self.assertEqual(result["returncode"], 0)
            self.assertIsNone(result["signal"])
            self.assertGreater(record["pid"], 0)
            self.assertEqual(record["state"], "complete")
            self.assertEqual(record["return_code"], 0)
            self.assertIsNone(record["signal"])
            self.assertIsNotNone(record["started_at"])
            self.assertIsNotNone(record["ended_at"])
            self.assertIn("-u", record["command"])
            self.assertEqual(
                states,
                [
                    "starting",
                    "downloading_model",
                    "initializing_model",
                    "extracting",
                    "finalizing",
                    "complete",
                ],
            )
            self.assertIn("stderr is streamed too", streamed.getvalue())
            self.assertIn("stderr is streamed too", log_path.read_text(encoding="utf-8"))
            self.assertTrue((root / "process_events.jsonl").is_file())

    def test_nonzero_exit_is_failure_with_return_code(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "exit code 7"):
                with contextlib.redirect_stdout(io.StringIO()):
                    run_logged(
                        [sys.executable, "-c", "raise SystemExit(7)"],
                        cwd=root,
                        log_path=root / "extraction.log",
                    )
            record = self._record(root)
            self.assertEqual(record["state"], "failed")
            self.assertEqual(record["return_code"], 7)
            self.assertIsNone(record["signal"])

    @unittest.skipIf(os.name == "nt", "POSIX signal semantics are required")
    def test_signal_exit_is_failure_with_signal_metadata(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            code = "import os, signal; os.kill(os.getpid(), signal.SIGTERM)"
            with self.assertRaisesRegex(RuntimeError, "signal SIGTERM"):
                with contextlib.redirect_stdout(io.StringIO()):
                    run_logged(
                        [sys.executable, "-c", code],
                        cwd=root,
                        log_path=root / "extraction.log",
                    )
            record = self._record(root)
            self.assertEqual(record["state"], "failed")
            self.assertEqual(record["return_code"], -signal.SIGTERM)
            self.assertEqual(record["signal"], "SIGTERM")


if __name__ == "__main__":
    unittest.main()
