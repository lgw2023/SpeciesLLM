import "@testing-library/jest-dom/vitest";
import { render, screen } from "@testing-library/react";
import { AnalysisPanel } from "../components/AnalysisPanel";
import { ComparePage } from "../pages/ComparePage";
import { ReportPage } from "../pages/ReportPage";

const runs = [
  {
    id: "run-1",
    name: "training_output_100m_data_1_2_3_stable_from_scratch_20260514_011839",
    model_size: "100m",
    data_recipe: "data_1_2_3",
    experiment_name: "stable",
    status: "needs_review",
    status_reason: "indexed",
    summary_one_liner: "Stable control.",
    tags: ["stable"],
  },
  {
    id: "run-2",
    name: "training_output_100m_data_1_2_3_stable_lr5em4_from_scratch_20260523_130351",
    model_size: "100m",
    data_recipe: "data_1_2_3",
    experiment_name: "lr sweep",
    status: "needs_review",
    status_reason: "indexed",
    summary_one_liner: "Bad plateau candidate.",
    tags: ["lr5em4"],
  },
];

test("compare page shows config diffs", () => {
  render(
    <ComparePage
      runs={runs}
      selectedRunIds={["run-1", "run-2"]}
      onSelectionChange={() => undefined}
      comparison={{
        config_diffs: [{ key: "run_record:lr", values: { "run-1": "1e-6", "run-2": "5e-4" } }],
        metric_summaries: [{ run_id: "run-1", best_loss: 10 }, { run_id: "run-2", best_loss: 20 }],
        diagnostics: [{ run_id: "run-2", event_type: "bad_plateau", title: "Bad plateau" }],
      }}
    />,
  );

  expect(screen.getByText("run_record:lr")).toBeInTheDocument();
  expect(screen.getByText("5e-4")).toBeInTheDocument();
  expect(screen.getByText("Bad plateau")).toBeInTheDocument();
});

test("report page shows phase names and evidence links", () => {
  render(
    <ReportPage
      stages={[{ name: "100M learning-rate sweep", runs }]}
      onOpenRun={() => undefined}
    />,
  );

  expect(screen.getByText("100M learning-rate sweep")).toBeInTheDocument();
  expect(screen.getAllByText("Open evidence")[0]).toBeInTheDocument();
});

test("analysis panel shows confidence and evidence refs", () => {
  render(
    <AnalysisPanel
      runId="run-1"
      diagnostics={[{ id: "evt-1", title: "Clip storm", event_type: "clip_storm" }]}
      notes={[{ id: "note-1", note_type: "deep_review", title: "Manual read", body: "Needs per-run reading.", confidence: "medium", evidence_refs: ["summary.md"] }]}
      onCreate={() => undefined}
      onUpdate={() => undefined}
    />,
  );

  expect(screen.getByText("medium confidence")).toBeInTheDocument();
  expect(screen.getByText("summary.md")).toBeInTheDocument();
});
