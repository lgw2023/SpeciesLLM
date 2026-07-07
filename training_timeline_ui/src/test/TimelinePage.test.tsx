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
          summary_one_liner: "Loss improved across the full epoch.",
          tags: ["stable"],
        },
      ]}
      loading={false}
      onOpenRun={() => undefined}
    />,
  );

  expect(screen.getByText("100m")).toBeInTheDocument();
  expect(screen.getByText("data_1_2_3")).toBeInTheDocument();
  expect(screen.getByText("Preliminary")).toBeInTheDocument();
  expect(screen.getByText("Loss improved across the full epoch.")).toBeInTheDocument();
});
