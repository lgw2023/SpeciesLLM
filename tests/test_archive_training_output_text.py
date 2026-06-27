from __future__ import annotations

import contextlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "archive_training_output_text.py"
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))


def load_archive_module():
    spec = importlib.util.spec_from_file_location("archive_training_output_text", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


archive_tool = load_archive_module()


class ArchiveTrainingOutputTextTest(unittest.TestCase):
    def test_pack_unpack_roundtrip_verifies_selected_files_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "training_output_demo"
            nested = source / "rank0"
            nested.mkdir(parents=True)

            expected = {
                source / "log.0-0.txt": b"log line 1\nlog line 2\n",
                source / "loss_to_log.0-0.txt": b"1\t12.5\n2\t11.0\n",
                source / "metrics.0-0.jsonl": b'{"step":1,"loss":12.5}\n',
                nested / "metrics.0-1.jsonl": b'{"step":2,"loss":11.0}\n',
            }
            for path, data in expected.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            (source / "checkpoint.pt").write_bytes(b"not selected")
            (source / "training_curves.png").write_bytes(b"not selected")
            (source / "notes.txt").write_bytes(b"not selected")

            archive_path = tmp_path / "training_output_demo_text.tar.xz"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = archive_tool.main(
                    ["pack", str(source), "--output", str(archive_path), "--preset", "0"]
                )
            self.assertEqual(rc, 0, stdout.getvalue())
            self.assertTrue(archive_path.is_file())

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(archive_tool.main(["verify", str(archive_path)]), 0)

            listing = io.StringIO()
            with contextlib.redirect_stdout(listing):
                self.assertEqual(archive_tool.main(["list", str(archive_path)]), 0)
            self.assertIn("log.0-0.txt", listing.getvalue())
            self.assertNotIn("checkpoint.pt", listing.getvalue())

            restored_root = tmp_path / "restored"
            with contextlib.redirect_stdout(io.StringIO()):
                rc = archive_tool.main(
                    ["unpack", str(archive_path), "--output-dir", str(restored_root)]
                )
            self.assertEqual(rc, 0)

            for original_path, data in expected.items():
                rel = original_path.relative_to(source)
                restored_path = restored_root / source.name / rel
                self.assertEqual(restored_path.read_bytes(), data)
            self.assertFalse((restored_root / source.name / "checkpoint.pt").exists())
            self.assertFalse((restored_root / source.name / "training_curves.png").exists())
            self.assertFalse((restored_root / source.name / "notes.txt").exists())

    def test_single_stream_roundtrip_remains_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "training_output_demo"
            source.mkdir()
            (source / "log.0-0.txt").write_bytes(b"log\n")
            (source / "loss_to_log.0-0.txt").write_bytes(b"loss\n")
            (source / "metrics.0-0.jsonl").write_bytes(b'{"loss":1}\n')

            archive_path = tmp_path / "training_output_demo_text.tar.xz"
            with contextlib.redirect_stdout(io.StringIO()):
                rc = archive_tool.main(
                    [
                        "pack",
                        str(source),
                        "--output",
                        str(archive_path),
                        "--preset",
                        "0",
                        "--single-stream",
                    ]
                )
            self.assertEqual(rc, 0)

            restored_root = tmp_path / "restored"
            with contextlib.redirect_stdout(io.StringIO()):
                rc = archive_tool.main(
                    ["unpack", str(archive_path), "--output-dir", str(restored_root)]
                )
            self.assertEqual(rc, 0)
            self.assertEqual(
                (restored_root / source.name / "metrics.0-0.jsonl").read_bytes(),
                b'{"loss":1}\n',
            )

    def test_split_pack_unpack_roundtrip_respects_chunk_size(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "training_output_demo"
            source.mkdir()
            expected = {
                source / "log.0-0.txt": os.urandom(12_000),
                source / "loss_to_log.0-0.txt": os.urandom(9_000),
                source / "metrics.0-0.jsonl": os.urandom(15_000),
            }
            for path, data in expected.items():
                path.write_bytes(data)
            (source / "checkpoint.pt").write_bytes(os.urandom(10_000))

            split_dir = tmp_path / "split"
            chunk_size = 4096
            with contextlib.redirect_stdout(io.StringIO()):
                rc = archive_tool.main(
                    [
                        "split-pack",
                        str(source),
                        "--output-dir",
                        str(split_dir),
                        "--chunk-size",
                        str(chunk_size),
                        "--jobs",
                        "2",
                        "--preset",
                        "0",
                    ]
                )
            self.assertEqual(rc, 0)
            self.assertTrue(any(path.suffix == ".xzpart" for path in split_dir.iterdir()))
            for path in split_dir.iterdir():
                if path.is_file():
                    self.assertLessEqual(path.stat().st_size, chunk_size, path.name)

            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(archive_tool.main(["split-verify", str(split_dir)]), 0)

            listing = io.StringIO()
            with contextlib.redirect_stdout(listing):
                self.assertEqual(archive_tool.main(["split-list", str(split_dir)]), 0)
            self.assertIn("metrics.0-0.jsonl", listing.getvalue())
            self.assertNotIn("checkpoint.pt", listing.getvalue())

            restored_root = tmp_path / "restored"
            with contextlib.redirect_stdout(io.StringIO()):
                rc = archive_tool.main(
                    [
                        "split-unpack",
                        str(split_dir),
                        "--output-dir",
                        str(restored_root),
                    ]
                )
            self.assertEqual(rc, 0)

            for original_path, data in expected.items():
                restored_path = restored_root / source.name / original_path.name
                self.assertEqual(restored_path.read_bytes(), data)
            self.assertFalse((restored_root / source.name / "checkpoint.pt").exists())

    def test_structured_split_codecs_roundtrip_stable_formats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "training_output_demo"
            source.mkdir()
            (source / "metrics.0-0.jsonl").write_bytes(
                b'{"time":"2026-05-06T22:15:37","node_rank":"0","rank":0,'
                b'"local_rank":0,"epoch":1,"batch_index":1,"num_batches":2,'
                b'"update_step":1,"should_step":true,"lr":1e-06,"loss_total":1.5,'
                b'"target_mean_masked":2.0,"abs_error_p99":4.0,'
                b'"batch_metadata":{"assay":{"top":[{"id":2,"count":4}]}}}\r\n'
                b'{"time":"2026-05-06T22:15:38","node_rank":"0","rank":0,'
                b'"local_rank":0,"epoch":1,"batch_index":2,"num_batches":2,'
                b'"update_step":2,"should_step":true,"lr":2e-06}\r\n'
            )
            (source / "loss_to_log.0-0.txt").write_bytes(
                b"time,node_rank,rank,local_rank,epoch,batch_index,lr,loss_total\r\n"
                b"2026-05-06T22:15:37,0,0,0,1,1,1e-06,1.5\r\n"
                b"2026-05-06T22:15:38,0,0,0,1,2,2e-06,\r\n"
            )
            (source / "log.0-0.txt").write_bytes(
                b"2026-05-06 22:14:52 INFO node=0 rank=0 local_rank=0 pid=123 args={\r\n"
                b'  "batch_size": 512\r\n'
                b"}\r\n"
            )

            split_dir = tmp_path / "split"
            with contextlib.redirect_stdout(io.StringIO()):
                rc = archive_tool.main(
                    [
                        "split-pack",
                        str(source),
                        "--output-dir",
                        str(split_dir),
                        "--chunk-size",
                        "4096",
                        "--jobs",
                        "2",
                        "--preset",
                        "0",
                    ]
                )
            self.assertEqual(rc, 0)

            manifest = archive_tool.read_split_manifest(split_dir)
            codecs = {item["path"].split("/")[-1]: item["codec"] for item in manifest["files"]}
            self.assertEqual(codecs["metrics.0-0.jsonl"], "metrics-jsonl-v1")
            self.assertEqual(codecs["loss_to_log.0-0.txt"], "loss-csv-v1")
            self.assertEqual(codecs["log.0-0.txt"], "log-template-v1")

            restored_root = tmp_path / "restored"
            with contextlib.redirect_stdout(io.StringIO()):
                rc = archive_tool.main(
                    [
                        "split-unpack",
                        str(split_dir),
                        "--output-dir",
                        str(restored_root),
                    ]
                )
            self.assertEqual(rc, 0)
            for original_path in source.iterdir():
                restored_path = restored_root / source.name / original_path.name
                self.assertEqual(restored_path.read_bytes(), original_path.read_bytes())

    def test_split_pack_streams_raw_for_files_above_structured_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "training_output_demo"
            source.mkdir()
            metrics_payload = b"".join(
                [
                    b'{"update_step":1,"loss_total":1.5,'
                    b'"batch_metadata":{"species":{"top":[{"id":5,"count":4}]}}}\n',
                    b'{"update_step":2,"loss_total":1.4,'
                    b'"target_mean_masked":2.0,"abs_error_p99":3.0}\n',
                ]
            )
            metrics_payload *= 8
            (source / "metrics.0-0.jsonl").write_bytes(metrics_payload)

            split_dir = tmp_path / "split"
            with contextlib.redirect_stdout(io.StringIO()):
                rc = archive_tool.main(
                    [
                        "split-pack",
                        str(source),
                        "--output-dir",
                        str(split_dir),
                        "--chunk-size",
                        "4096",
                        "--structured-max-bytes",
                        "64",
                        "--preset",
                        "0",
                    ]
                )
            self.assertEqual(rc, 0)

            manifest = archive_tool.read_split_manifest(split_dir)
            codecs = {item["path"].split("/")[-1]: item["codec"] for item in manifest["files"]}
            self.assertEqual(codecs["metrics.0-0.jsonl"], "raw-bytes")

            restored_root = tmp_path / "restored"
            with contextlib.redirect_stdout(io.StringIO()):
                rc = archive_tool.main(
                    [
                        "split-unpack",
                        str(split_dir),
                        "--output-dir",
                        str(restored_root),
                    ]
                )
            self.assertEqual(rc, 0)
            restored = restored_root / source.name / "metrics.0-0.jsonl"
            self.assertEqual(restored.read_bytes(), metrics_payload)


if __name__ == "__main__":
    unittest.main()
