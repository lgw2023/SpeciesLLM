import "@testing-library/jest-dom/vitest";
import { render, screen } from "@testing-library/react";
import { RunDetailPage } from "../pages/RunDetailPage";

test("run detail separates automatic diagnostics and deep review notes", () => {
  render(
    <RunDetailPage
      run={{
        id: "run-1",
        name: "training_output_demo",
        model_size: "100m",
        data_recipe: "data_1_2_3",
        experiment_name: "demo",
        status: "needs_review",
        status_reason: "indexed",
        summary_title: "Clip storm candidate",
        summary_one_liner: "GEP rebounded after clipping increased.",
        tags: [],
      }}
      metrics={{ loss_total: [{ step: 1, value: 10 }, { step: 2, value: 12 }] }}
      diagnostics={[{ id: "evt-1", event_type: "clip_storm", severity: "warning", title: "Clip storm", description: "Clip fraction exceeded threshold." }]}
      artifacts={[{ kind: "summary_md", path: "/repo/run/summary.md" }]}
      notes={[{ id: "note-1", note_type: "deep_review", title: "Manual read", body: "Needs per-run interpretation.", confidence: "medium" }]}
    />,
  );

  expect(screen.getByText("Automatic diagnostics")).toBeInTheDocument();
  expect(screen.getByText("Deep review")).toBeInTheDocument();
  expect(screen.getByText("Manual read")).toBeInTheDocument();
});
