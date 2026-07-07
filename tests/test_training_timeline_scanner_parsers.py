from __future__ import annotations

import json
from pathlib import Path

from tests.training_timeline_fixtures import make_run_dir
from training_timeline.config import is_inside_source, load_source_roots
from training_timeline.parsers import collect_artifacts, parse_run_directory_name, parse_run_record, parse_summary
from training_timeline.scanner import discover_runs


def test_load_source_roots_uses_repo_and_extra_roots(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    extra = tmp_path / "extra"
    repo.mkdir()
    extra.mkdir()
    config = tmp_path / "sources.json"
    config.write_text(json.dumps({"extra_source_roots": [str(extra)]}), encoding="utf-8")

    roots = load_source_roots(config, repo)

    assert roots == [repo.resolve(), extra.resolve()]


def test_is_inside_source_blocks_outside_paths(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    inside = root / "training_output_demo"
    outside = tmp_path / "outside"

    assert is_inside_source(inside, [root]) is True
    assert is_inside_source(outside, [root]) is False


def test_discover_runs_skips_text_split_and_deduplicates_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    run = make_run_dir(
        root,
        "training_output_100m_data_1_2_3_stable_from_scratch_20260514_011839",
        metrics_rows=[{"update_step": 1, "loss_total": 10.0}],
    )
    make_run_dir(root, "training_output_empty")
    make_run_dir(root, "training_output_100m_data_1_2_3_stable_from_scratch_20260514_011839_text_split")
    symlink = root / "training_output_duplicate_link"
    symlink.symlink_to(run, target_is_directory=True)

    runs = discover_runs([root])

    assert [item.name for item in runs] == ["training_output_100m_data_1_2_3_stable_from_scratch_20260514_011839"]


def test_parse_run_directory_name_extracts_main_tags() -> None:
    parsed = parse_run_directory_name(
        "training_output_100m_data_1_3_E2_huber5_fp32_esm2_dnaseq_lossw_gepc01_shuffleall_from_scratch_20260622_162826"
    )

    assert parsed.model_size == "100m"
    assert parsed.data_recipe == "data_1_3"
    assert parsed.started_at == "2026-06-22T16:28:26"
    assert "huber5" in parsed.tags
    assert "fp32" in parsed.tags
    assert "shuffleall" in parsed.tags
    assert "lossw_gepc01" in parsed.tags


def test_parse_run_record_summary_and_artifacts(tmp_path: Path) -> None:
    run = make_run_dir(
        tmp_path,
        "training_output_500m_data_1_2_3_stable_from_scratch_20260515_195223",
        run_record={"created_at_utc": "2026-05-15T11:52:23Z", "argv": ["train.py"], "git": {"commit": "abc"}},
        summary="# Failure summary\n\nThe primary head failed while skip behavior repeated.\n",
        metrics_rows=[{"update_step": 1, "loss_total": 10.0}],
    )
    (run / "loss_detail.png").write_bytes(b"png")

    record = parse_run_record(run)
    summary = parse_summary(run)
    artifacts = collect_artifacts(run)

    assert record["created_at_utc"] == "2026-05-15T11:52:23Z"
    assert summary.title == "Failure summary"
    assert "primary head" in summary.one_liner
    assert {item.kind for item in artifacts} >= {"summary_md", "run_record_json", "metrics_jsonl", "training_curve_png"}
