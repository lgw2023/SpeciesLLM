import type { RunSummary } from "./types";

export const TIMELINE_NODE_WIDTH = 260;
export const TIMELINE_NODE_HEIGHT = 150;
export const TIMELINE_COLUMN_GAP = 86;
export const TIMELINE_ROW_GAP = 96;
export const TIMELINE_LEFT_PAD = 24;
export const TIMELINE_TOP_PAD = 78;

export type TimelineNode = {
  id: string;
  run: RunSummary;
  x: number;
  y: number;
  lane: string;
  index: number;
};

export type TimelineEdge = {
  id: string;
  fromId: string;
  toId: string;
  label: string;
};

export type TimelineGraph = {
  nodes: TimelineNode[];
  edges: TimelineEdge[];
  lanes: string[];
  width: number;
  height: number;
};

const LANE_ORDER = ["Scale smoke", "100M data baselines", "500M stability", "LR / schedule", "E2 ablations"];

export function buildTimelineGraph(runs: RunSummary[]): TimelineGraph {
  const sortedRuns = [...runs].sort((a, b) => (a.started_at ?? "").localeCompare(b.started_at ?? ""));
  const lanes = LANE_ORDER.filter((lane) => sortedRuns.some((run) => laneForRun(run) === lane));
  const activeLanes = lanes.length ? lanes : ["Experiments"];
  const laneIndexes = new Map(activeLanes.map((lane, index) => [lane, index]));
  const nodes = sortedRuns.map<TimelineNode>((run, index) => {
    const lane = laneForRun(run);
    const laneIndex = laneIndexes.get(lane) ?? 0;
    return {
      id: run.id,
      run,
      x: TIMELINE_LEFT_PAD + laneIndex * (TIMELINE_NODE_WIDTH + TIMELINE_COLUMN_GAP),
      y: TIMELINE_TOP_PAD + index * (TIMELINE_NODE_HEIGHT + TIMELINE_ROW_GAP),
      lane,
      index,
    };
  });
  const edges = sortedRuns.flatMap<TimelineEdge>((run, index) => {
    if (index === 0) {
      return [];
    }
    const parent = findParentRun(run, sortedRuns.slice(0, index));
    return [
      {
        id: `${parent.id}-${run.id}`,
        fromId: parent.id,
        toId: run.id,
        label: describeVariableChange(parent, run),
      },
    ];
  });
  return {
    nodes,
    edges,
    lanes: activeLanes,
    width: Math.max(880, TIMELINE_LEFT_PAD * 2 + activeLanes.length * TIMELINE_NODE_WIDTH + Math.max(0, activeLanes.length - 1) * TIMELINE_COLUMN_GAP),
    height: TIMELINE_TOP_PAD + sortedRuns.length * TIMELINE_NODE_HEIGHT + Math.max(0, sortedRuns.length - 1) * TIMELINE_ROW_GAP + 54,
  };
}

export function describeVariableChange(parent: RunSummary, child: RunSummary): string {
  const before = extractVariables(parent);
  const after = extractVariables(child);
  const changes = [
    changed("model size", before.model, after.model, "not recorded"),
    changed("training data recipe", before.data, after.data, "not recorded"),
    changed("training recipe", before.recipe, after.recipe, "baseline from-scratch"),
    changed("learning rate", before.lr, after.lr, "default"),
    changed("grad clip norm", before.clip, after.clip, "default"),
    changed("epoch budget", before.epochs, after.epochs, "default"),
    changed("LR decay schedule", before.lrDecay, after.lrDecay, "default"),
    changed("numeric precision", before.precision, after.precision, "default"),
    changed("loss function", before.loss, after.loss, "default"),
    changed("input modalities", before.modalities, after.modalities, "baseline tokens"),
    changed("data order", before.shuffle, after.shuffle, "default shard order"),
  ].filter((item): item is string => Boolean(item));

  if (changes.length === 0) {
    return "repeat run / validation only";
  }
  const visibleChanges = changes.slice(0, 4);
  const hiddenCount = changes.length - visibleChanges.length;
  return hiddenCount > 0 ? `${visibleChanges.join(" | ")} | +${hiddenCount} more` : visibleChanges.join(" | ");
}

function findParentRun(run: RunSummary, candidates: RunSummary[]): RunSummary {
  return candidates.reduce(
    (best, candidate, index) => {
      const score = parentScore(candidate, run, index);
      if (score > best.score) {
        return { run: candidate, score };
      }
      return best;
    },
    { run: candidates[candidates.length - 1], score: -Infinity },
  ).run;
}

function parentScore(candidate: RunSummary, run: RunSummary, index: number): number {
  const candidateVars = extractVariables(candidate);
  const runVars = extractVariables(run);
  let score = index * 0.01;
  if (laneForRun(candidate) === laneForRun(run)) score += 5;
  if (candidateVars.model === runVars.model) score += 4;
  if (candidateVars.data !== "unknown" && candidateVars.data === runVars.data) score += 4;
  if (candidateVars.recipe === runVars.recipe) score += 2;
  if (candidateVars.lr && candidateVars.lr === runVars.lr) score += 2;
  if (candidateVars.loss && candidateVars.loss === runVars.loss) score += 2;
  if (candidateVars.modalities && candidateVars.modalities === runVars.modalities) score += 2;
  score += sharedTags(candidate, run).length * 1.2;
  return score;
}

function laneForRun(run: RunSummary): string {
  const name = normalizedName(run);
  if (run.model_size === "500m" || name.includes("stab_smoke") || name.includes("clip0p5")) {
    return "500M stability";
  }
  if (name.includes("huber") || name.includes("fp32") || name.includes("esm2") || name.includes("dnaseq") || name.includes("lossw") || name.includes("shuffle")) {
    return "E2 ablations";
  }
  if (name.includes("lr") || name.includes("epoch") || name.includes("lrdecay")) {
    return "LR / schedule";
  }
  if (run.model_size === "1b" || (name.includes("test_from_scratch") && !run.data_recipe)) {
    return "Scale smoke";
  }
  return "100M data baselines";
}

function extractVariables(run: RunSummary) {
  const name = normalizedName(run);
  const tags = run.tags.map((tag) => tag.toLowerCase());
  return {
    model: run.model_size || "unknown",
    data: run.data_recipe || "unknown",
    recipe: trainingRecipeName(name, tags),
    lr: formatLearningRate(firstMatch([...tags, name], /lr(\d+)em(\d+)/) ?? firstMatch([...tags, name], /lr(\d+)e-(\d+)/)),
    clip: formatClip(firstMatch([...tags, name], /clip(\d+)p(\d+)/)),
    epochs: firstMatch([...tags, name], /(?:epoch|epochs)(\d+)/)?.[1] ?? firstMatch([...tags, name], /(\d+)epoch/)?.[1] ?? "",
    lrDecay: formatLrDecay(firstMatch([...tags, name], /lrdecay(\d+)/)),
    precision: name.includes("fp32") || tags.includes("fp32") ? "fp32" : "",
    loss: lossName(name, tags),
    modalities: modalityName(name, tags),
    shuffle: name.includes("shuffleall") || tags.includes("shuffleall") ? "all rows" : "",
  };
}

function changed(label: string, before: string, after: string, defaultBefore: string): string | null {
  if (!after || after === "unknown" || before === after) {
    return null;
  }
  const beforeLabel = !before || before === "unknown" ? defaultBefore : before;
  return `${label}: ${beforeLabel} -> ${after}`;
}

function firstMatch(values: string[], pattern: RegExp): RegExpMatchArray | null {
  for (const value of values) {
    const match = value.match(pattern);
    if (match) {
      return match;
    }
  }
  return null;
}

function formatLearningRate(match: RegExpMatchArray | null): string {
  if (!match) {
    return "";
  }
  return `${match[1]}e-${match[2]}`;
}

function formatClip(match: RegExpMatchArray | null): string {
  if (!match) {
    return "";
  }
  return `${match[1]}.${match[2]}`;
}

function formatLrDecay(match: RegExpMatchArray | null): string {
  if (!match) {
    return "";
  }
  return `lrdecay${match[1]}`;
}

function lossName(name: string, tags: string[]): string {
  if (name.includes("huber5") || tags.includes("huber5")) {
    return "Huber5";
  }
  if (name.includes("huber") || tags.includes("huber")) {
    return "Huber";
  }
  if (name.includes("lossw_gepc01") || tags.includes("lossw_gepc01")) {
    return "GEPC loss weight 0.1";
  }
  return "";
}

function trainingRecipeName(name: string, tags: string[]): string {
  if (name.includes("stab_smoke")) {
    return "stability smoke diagnostic";
  }
  if (name.includes("stable") || tags.includes("stable")) {
    return "stable pretrain recipe";
  }
  return "baseline from-scratch";
}

function modalityName(name: string, tags: string[]): string {
  const modalities = [];
  if (name.includes("esm2") || tags.includes("esm2")) {
    modalities.push("ESM2");
  }
  if (name.includes("dnaseq") || tags.includes("dnaseq")) {
    modalities.push("DNAseq");
  }
  return modalities.join(" + ");
}

function sharedTags(a: RunSummary, b: RunSummary): string[] {
  const ignored = new Set(["from_scratch"]);
  const aTags = new Set(a.tags.map((tag) => tag.toLowerCase()).filter((tag) => !ignored.has(tag)));
  return b.tags.map((tag) => tag.toLowerCase()).filter((tag) => aTags.has(tag));
}

function normalizedName(run: RunSummary): string {
  return `${run.name} ${run.experiment_name}`.toLowerCase();
}
