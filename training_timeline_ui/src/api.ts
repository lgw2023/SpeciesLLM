import type { IndexRebuildResponse, RunsResponse, SourcesResponse } from "./types";

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
