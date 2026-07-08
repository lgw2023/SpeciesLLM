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

export type EvidenceRef = {
  kind: string;
  ref: string;
  detail?: string;
};

export type RunRelationship = {
  id: string;
  parent_run_id: string;
  child_run_id: string;
  relationship_type: string;
  confidence: string;
  change_summary: string;
  evidence_refs: EvidenceRef[];
};

export type MetricDatum = {
  step: number;
  epoch?: number | null;
  value: number;
};

export type MetricSeries = Record<string, MetricDatum[]>;

export type DiagnosticEvent = {
  id?: string;
  run_id?: string;
  event_type: string;
  severity?: string;
  title: string;
  description?: string;
  evidence?: Record<string, unknown>;
  start_step?: number | null;
  end_step?: number | null;
};

export type ArtifactRef = {
  kind: string;
  path: string;
  size_bytes?: number;
  mtime?: number;
};

export type AnalysisNote = {
  id: string;
  run_id?: string;
  note_type: "curated_note" | "deep_review" | string;
  title: string;
  body: string;
  confidence: "low" | "medium" | "high" | string;
  supersedes_diagnostic_ids?: string[];
  evidence_refs?: string[];
  author?: string;
  created_at?: string;
  updated_at?: string;
};

export type RunDetailResponse = {
  run: RunSummary;
};

export type MetricsResponse = {
  run_id: string;
  series: MetricSeries;
};

export type DiagnosticsResponse = {
  run_id: string;
  diagnostics: DiagnosticEvent[];
};

export type ArtifactsResponse = {
  run_id: string;
  artifacts: ArtifactRef[];
};

export type AnalysisResponse = {
  run_id: string;
  notes: AnalysisNote[];
};

export type ComparisonResponse = {
  runs?: RunSummary[];
  config_diffs: Array<{ key: string; values: Record<string, string> }>;
  metric_summaries: Array<Record<string, unknown>>;
  diagnostics: DiagnosticEvent[];
};

export type ReportStage = {
  name: string;
  runs: RunSummary[];
};

export type ReportStagesResponse = {
  stages: ReportStage[];
};

export type ReportTimelineResponse = {
  runs: RunSummary[];
  relationships: RunRelationship[];
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
