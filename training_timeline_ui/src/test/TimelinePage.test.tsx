import "@testing-library/jest-dom/vitest";
import { render, screen } from "@testing-library/react";
import { TimelinePage } from "../pages/TimelinePage";

test("timeline shows run metadata and preliminary labels", () => {
  render(
    <TimelinePage
      runs={[
        {
          id: "run-1",
          name: "training_output_100m_data_1_2_3_stable_from_scratch_20260514_011839",
          started_at: "2026-05-14T01:18:39",
          model_size: "100m",
          data_recipe: "data_1_2_3",
          experiment_name: "stable_from_scratch",
          status: "success",
          status_reason: "converged",
          summary_one_liner: "> **Loss** improved across the `full epoch` ([summary](summary.md)).",
          tags: ["stable"],
        },
        {
          id: "run-2",
          name: "training_output_100m_data_1_2_3_stable_lr1em5_from_scratch_20260519_204028",
          started_at: "2026-05-19T20:40:28",
          model_size: "100m",
          data_recipe: "data_1_2_3",
          experiment_name: "stable_lr1em5_from_scratch",
          status: "needs_review",
          status_reason: "metrics_indexed",
          summary_one_liner: "Learning-rate sweep.",
          tags: ["stable", "lr1em5", "from_scratch"],
        },
      ]}
      loading={false}
      onOpenRun={() => undefined}
    />,
  );

  expect(screen.getByText(/2 experiment nodes/)).toBeInTheDocument();
  expect(screen.getByText(/1 inferred relations/)).toBeInTheDocument();
  expect(screen.getByText(/Older experiments start at the top/)).toBeInTheDocument();
  expect(screen.getAllByText("100m").length).toBeGreaterThan(0);
  expect(screen.getAllByText("data_1_2_3").length).toBeGreaterThan(0);
  expect(screen.getByText("learning rate: default -> 1e-5")).toBeInTheDocument();
  expect(screen.getByText("Loss improved across the full epoch (summary).")).toBeInTheDocument();
  expect(screen.queryByText(/summary.md/)).not.toBeInTheDocument();
});
