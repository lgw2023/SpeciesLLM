export type RunSummary = {
  id: string;
  name: string;
  started_at?: string | null;
  model_size: string;
  data_recipe: string;
  experiment_name: string;
  status: string;
  status_reason: string;
  summary_title?: string;
  summary_one_liner: string;
  tags: string[];
};

export type RunsResponse = {
  runs: RunSummary[];
};

export type SourceInfo = {
  path: string;
};

export type SourcesResponse = {
  sources: SourceInfo[];
  run_count: number;
};

export type IndexRebuildResponse = {
  result: {
    discovered: number;
    indexed: number;
    skipped: number;
    warnings: number;
  };
};
