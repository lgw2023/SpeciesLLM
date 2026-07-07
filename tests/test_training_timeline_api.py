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
