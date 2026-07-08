import type {
  AnalysisNote,
  AnalysisResponse,
  ArtifactsResponse,
  ComparisonResponse,
  DiagnosticsResponse,
  IndexRebuildResponse,
  MetricsResponse,
  ReportStagesResponse,
  ReportTimelineResponse,
  RunDetailResponse,
  RunsResponse,
  SourcesResponse,
} from "./types";

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return (await response.json()) as T;
}

export function fetchRuns(): Promise<RunsResponse> {
  return requestJson<RunsResponse>("/api/runs");
}

export function fetchSources(): Promise<SourcesResponse> {
  return requestJson<SourcesResponse>("/api/sources");
}

export function rebuildIndex(): Promise<IndexRebuildResponse> {
  return requestJson<IndexRebuildResponse>("/api/index/rebuild", { method: "POST" });
}

export function fetchRun(runId: string): Promise<RunDetailResponse> {
  return requestJson<RunDetailResponse>(`/api/runs/${runId}`);
}

export function fetchMetrics(runId: string): Promise<MetricsResponse> {
  return requestJson<MetricsResponse>(`/api/runs/${runId}/metrics?series=loss_total,loss_gep,loss_zero_prob,loss_gepc,lr,grad_norm_raw,clip_fraction,skip_fraction`);
}

export function fetchDiagnostics(runId: string): Promise<DiagnosticsResponse> {
  return requestJson<DiagnosticsResponse>(`/api/runs/${runId}/diagnostics`);
}

export function fetchArtifacts(runId: string): Promise<ArtifactsResponse> {
  return requestJson<ArtifactsResponse>(`/api/runs/${runId}/artifacts`);
}

export function fetchAnalysis(runId: string): Promise<AnalysisResponse> {
  return requestJson<AnalysisResponse>(`/api/runs/${runId}/analysis`);
}

export function createAnalysisNote(runId: string, payload: Partial<AnalysisNote>): Promise<AnalysisNote> {
  return requestJson<AnalysisNote>(`/api/runs/${runId}/analysis`, { method: "POST", body: JSON.stringify(payload) });
}

export function updateAnalysisNote(runId: string, noteId: string, payload: Partial<AnalysisNote>): Promise<AnalysisNote> {
  return requestJson<AnalysisNote>(`/api/runs/${runId}/analysis/${noteId}`, { method: "PATCH", body: JSON.stringify(payload) });
}

export function compareRuns(runIds: string[]): Promise<ComparisonResponse> {
  return requestJson<ComparisonResponse>("/api/compare", { method: "POST", body: JSON.stringify({ run_ids: runIds }) });
}

export function fetchReportStages(): Promise<ReportStagesResponse> {
  return requestJson<ReportStagesResponse>("/api/report/stages");
}

export function fetchReportTimeline(): Promise<ReportTimelineResponse> {
  return requestJson<ReportTimelineResponse>("/api/report/timeline");
}
