from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from tests.training_timeline_fixtures import make_run_dir
from training_timeline.api import create_app


def test_api_lists_runs_and_serves_run_details(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    make_run_dir(
        root,
        "training_output_100m_data_1_2_3_stable_from_scratch_20260514_011839",
        summary="# Stable\n\nLoss improved.\n",
        metrics_rows=[{"update_step": 1, "loss_total": 100.0}, {"update_step": 2, "loss_total": 50.0}],
    )
    app = create_app(tmp_path / "timeline.sqlite", [root])
    client = TestClient(app)

    rebuild = client.post("/api/index/rebuild")
    runs = client.get("/api/runs").json()
    run_id = runs["runs"][0]["id"]
    detail = client.get(f"/api/runs/{run_id}").json()
    metrics = client.get(f"/api/runs/{run_id}/metrics?series=loss_total").json()

    assert rebuild.status_code == 200
    assert runs["runs"][0]["model_size"] == "100m"
    assert detail["run"]["summary_title"] == "Stable"
    assert metrics["series"]["loss_total"][0]["step"] == 1


def test_api_analysis_crud_and_evidence(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    make_run_dir(root, "training_output_100m_data_1_2_3_stable_from_scratch_20260514_011839", summary="# Stable\n\nLoss improved.\n")
    app = create_app(tmp_path / "timeline.sqlite", [root])
    client = TestClient(app)
    client.post("/api/index/rebuild")
    run_id = client.get("/api/runs").json()["runs"][0]["id"]

    created = client.post(
        f"/api/runs/{run_id}/analysis",
        json={
            "note_type": "deep_review",
            "title": "Manual read",
            "body": "This run is a healthy control.",
            "confidence": "high",
            "supersedes_diagnostic_ids": [],
            "evidence_refs": ["summary.md"],
            "author": "local",
        },
    ).json()
    patched = client.patch(f"/api/runs/{run_id}/analysis/{created['id']}", json={"confidence": "medium"}).json()
    evidence = client.get(f"/api/runs/{run_id}/evidence").json()

    assert patched["confidence"] == "medium"
    assert evidence["run_id"] == run_id


def test_report_stages_keep_lr_and_multi_epoch_separate_from_stability(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    for name in [
        "training_output_500m_stab_A_clip0p5_lr1e-6_13k_from_scratch_20260518_213646",
        "training_output_100m_data_1_2_3_stable_lr5em4_from_scratch_20260523_130351",
        "training_output_100m_data_1_2_3_stable_5epoch_lrdecay1_from_scratch_20260526_172620",
    ]:
        make_run_dir(root, name, metrics_rows=[{"update_step": 1, "loss_total": 10.0}])
    app = create_app(tmp_path / "timeline.sqlite", [root])
    client = TestClient(app)
    client.post("/api/index/rebuild")

    stage_names = [stage["name"] for stage in client.get("/api/report/stages").json()["stages"]]

    assert "500M stability incident and fixes" in stage_names
    assert "100M learning-rate sweep" in stage_names
    assert "Multi-epoch and resume experiments" in stage_names


def test_report_timeline_returns_relationship_edges(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    make_run_dir(root, "training_output_100m_data_1_from_scratch_20260501_000000", metrics_rows=[{"update_step": 1, "loss_total": 10.0}])
    make_run_dir(root, "training_output_100m_data_1_3_stable_lr5em4_from_scratch_20260502_000000", metrics_rows=[{"update_step": 1, "loss_total": 9.0}])
    app = create_app(tmp_path / "timeline.sqlite", [root])
    client = TestClient(app)
    client.post("/api/index/rebuild")

    timeline = client.get("/api/report/timeline").json()

    assert len(timeline["runs"]) == 2
    assert len(timeline["relationships"]) == 1
    assert timeline["relationships"][0]["parent_run_id"] == timeline["runs"][0]["id"]
    assert timeline["relationships"][0]["child_run_id"] == timeline["runs"][1]["id"]
    assert "training data recipe: data_1 -> data_1_3" in timeline["relationships"][0]["change_summary"]
