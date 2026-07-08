import { buildTimelineGraph, describeVariableChange } from "../timelineGraph";
import type { RunSummary } from "../types";

function run(overrides: Partial<RunSummary>): RunSummary {
  return {
    id: "run",
    name: "training_output_100m_data_1_from_scratch_20260501_000000",
    started_at: "2026-05-01T00:00:00",
    model_size: "100m",
    data_recipe: "data_1",
    experiment_name: "from_scratch",
    status: "needs_review",
    status_reason: "metrics_indexed",
    summary_one_liner: "",
    tags: ["from_scratch"],
    ...overrides,
  };
}

test("buildTimelineGraph orders experiments top to bottom and labels relation edges", () => {
  const graph = buildTimelineGraph([
    run({ id: "late", started_at: "2026-05-03T00:00:00", data_recipe: "data_1_3", experiment_name: "from_scratch" }),
    run({ id: "early", started_at: "2026-05-01T00:00:00", data_recipe: "data_1", experiment_name: "from_scratch" }),
  ]);

  expect(graph.nodes.map((node) => node.id)).toEqual(["early", "late"]);
  expect(graph.nodes[0].y).toBeLessThan(graph.nodes[1].y);
  expect(graph.edges[0]).toMatchObject({ fromId: "early", toId: "late", label: "training data recipe: data_1 -> data_1_3" });
});

test("buildTimelineGraph prefers indexed relationship evidence over browser-only inference", () => {
  const graph = buildTimelineGraph(
    [
      run({ id: "early", started_at: "2026-05-01T00:00:00", data_recipe: "data_1", experiment_name: "from_scratch" }),
      run({ id: "late", started_at: "2026-05-02T00:00:00", data_recipe: "data_1_3", experiment_name: "stable_lr5em4_from_scratch" }),
    ],
    [
      {
        id: "edge-1",
        parent_run_id: "early",
        child_run_id: "late",
        relationship_type: "inferred_parent",
        confidence: "medium",
        change_summary: "Git/work_record evidence: data recipe data_1 -> data_1_3; lr default -> 5e-4",
        evidence_refs: [{ kind: "script", ref: "work_record/step1_data_1_3.sh" }],
      },
    ],
  );

  expect(graph.edges).toHaveLength(1);
  expect(graph.edges[0]).toMatchObject({
    fromId: "early",
    toId: "late",
    label: "Git/work_record evidence: data recipe data_1 -> data_1_3; lr default -> 5e-4",
  });
});

test("describeVariableChange extracts common training variable changes", () => {
  expect(
    describeVariableChange(
      run({ id: "base", experiment_name: "stable_from_scratch", tags: ["stable", "from_scratch"] }),
      run({
        id: "lr",
        name: "training_output_100m_data_1_stable_lr5em4_from_scratch_20260502_000000",
        experiment_name: "stable_lr5em4_from_scratch",
        tags: ["stable", "lr5em4", "from_scratch"],
      }),
    ),
  ).toBe("learning rate: default -> 5e-4");
});

test("describeVariableChange uses concrete recipe labels instead of vague stability labels", () => {
  expect(
    describeVariableChange(
      run({ id: "base", experiment_name: "from_scratch", tags: ["from_scratch"] }),
      run({
        id: "stable",
        name: "training_output_100m_data_1_stable_from_scratch_20260502_000000",
        experiment_name: "stable_from_scratch",
        tags: ["stable", "from_scratch"],
      }),
    ),
  ).toBe("training recipe: baseline from-scratch -> stable pretrain recipe");
});
