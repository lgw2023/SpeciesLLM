import { BarChart3, Database, FileText, GitCompare, ListTree } from "lucide-react";
import { useEffect, useState } from "react";
import {
  compareRuns,
  createAnalysisNote,
  fetchAnalysis,
  fetchArtifacts,
  fetchDiagnostics,
  fetchMetrics,
  fetchReportStages,
  fetchRun,
  fetchRuns,
  fetchSources,
  rebuildIndex,
  updateAnalysisNote,
} from "./api";
import { ComparePage } from "./pages/ComparePage";
import { ReportPage } from "./pages/ReportPage";
import { RunDetailPage } from "./pages/RunDetailPage";
import { SourcesPage } from "./pages/SourcesPage";
import { TimelinePage } from "./pages/TimelinePage";
import type { AnalysisNote, ArtifactRef, ComparisonResponse, DiagnosticEvent, MetricSeries, ReportStage, RunSummary, SourceInfo } from "./types";

type View = "timeline" | "sources" | "compare" | "report" | "run-detail";

export default function App() {
  const [view, setView] = useState<View>("timeline");
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [sources, setSources] = useState<SourceInfo[]>([]);
  const [runCount, setRunCount] = useState(0);
  const [loadingRuns, setLoadingRuns] = useState(true);
  const [loadingSources, setLoadingSources] = useState(false);
  const [rebuilding, setRebuilding] = useState(false);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [detailRun, setDetailRun] = useState<RunSummary | null>(null);
  const [metrics, setMetrics] = useState<MetricSeries>({});
  const [diagnostics, setDiagnostics] = useState<DiagnosticEvent[]>([]);
  const [artifacts, setArtifacts] = useState<ArtifactRef[]>([]);
  const [notes, setNotes] = useState<AnalysisNote[]>([]);
  const [selectedRunIds, setSelectedRunIds] = useState<string[]>([]);
  const [comparison, setComparison] = useState<ComparisonResponse | null>(null);
  const [reportStages, setReportStages] = useState<ReportStage[]>([]);
  const [appError, setAppError] = useState<string | null>(null);

  useEffect(() => {
    void loadRuns();
  }, []);

  async function loadRuns() {
    setLoadingRuns(true);
    try {
      const response = await fetchRuns();
      setRuns(response.runs);
      setRunCount(response.runs.length);
      setAppError(null);
    } catch {
      setRuns([]);
      setRunCount(0);
      setAppError("Could not load runs. Check that the backend is running and the frontend proxy points to the same port.");
    } finally {
      setLoadingRuns(false);
    }
  }

  async function loadSources() {
    setLoadingSources(true);
    try {
      const response = await fetchSources();
      setSources(response.sources);
      setRunCount(response.run_count);
      setAppError(null);
    } catch {
      setSources([]);
      setAppError("Could not load sources. Check the backend process and rebuild the index after it is available.");
    } finally {
      setLoadingSources(false);
    }
  }

  async function handleRebuild() {
    setRebuilding(true);
    try {
      await rebuildIndex();
      await Promise.all([loadRuns(), loadSources()]);
    } catch {
      setAppError("Index rebuild failed. Check the backend logs for the source path that could not be scanned.");
    } finally {
      setRebuilding(false);
    }
  }

  function openSources() {
    setView("sources");
    void loadSources();
  }

  function openRun(runId: string) {
    setSelectedRunId(runId);
    setView("run-detail");
    void loadRunDetail(runId);
  }

  async function loadRunDetail(runId: string) {
    try {
      const [runResponse, metricsResponse, diagnosticsResponse, artifactsResponse, analysisResponse] = await Promise.all([
        fetchRun(runId),
        fetchMetrics(runId),
        fetchDiagnostics(runId),
        fetchArtifacts(runId),
        fetchAnalysis(runId),
      ]);
      setDetailRun(runResponse.run);
      setMetrics(metricsResponse.series);
      setDiagnostics(diagnosticsResponse.diagnostics);
      setArtifacts(artifactsResponse.artifacts);
      setNotes(analysisResponse.notes);
      setAppError(null);
    } catch {
      setDetailRun(null);
      setAppError("Could not load run detail. The index may be stale; rebuild and try again.");
    }
  }

  async function handleCreateNote(payload: Partial<AnalysisNote>) {
    if (!selectedRunId) {
      return;
    }
    await createAnalysisNote(selectedRunId, payload);
    await loadRunDetail(selectedRunId);
  }

  async function handleUpdateNote(noteId: string, payload: Partial<AnalysisNote>) {
    if (!selectedRunId) {
      return;
    }
    await updateAnalysisNote(selectedRunId, noteId, payload);
    await loadRunDetail(selectedRunId);
  }

  async function handleSelectionChange(runIds: string[]) {
    setSelectedRunIds(runIds);
    if (runIds.length >= 2) {
      try {
        setComparison(await compareRuns(runIds));
        setAppError(null);
      } catch {
        setComparison(null);
        setAppError("Could not compare selected runs. Check that the backend is reachable.");
      }
    } else {
      setComparison(null);
    }
  }

  async function openReport() {
    setView("report");
    try {
      const response = await fetchReportStages();
      setReportStages(response.stages);
      setAppError(null);
    } catch {
      setReportStages([]);
      setAppError("Could not load the report. Check that the backend is reachable.");
    }
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <BarChart3 aria-hidden="true" size={22} />
          <span>SpeciesLLM</span>
        </div>
        <nav>
          <button className={view === "timeline" ? "active" : ""} type="button" onClick={() => setView("timeline")}>
            <ListTree aria-hidden="true" size={18} />
            Timeline
          </button>
          <button className={view === "sources" ? "active" : ""} type="button" onClick={openSources}>
            <Database aria-hidden="true" size={18} />
            Sources
          </button>
          <button className={view === "compare" ? "active" : ""} type="button" onClick={() => setView("compare")}>
            <GitCompare aria-hidden="true" size={18} />
            Compare
          </button>
          <button className={view === "report" ? "active" : ""} type="button" onClick={() => void openReport()}>
            <FileText aria-hidden="true" size={18} />
            Report
          </button>
        </nav>
      </aside>
      <main className="main-surface">
        {appError ? (
          <div className="error-banner" role="alert">
            {appError}
          </div>
        ) : null}
        {view === "timeline" ? <TimelinePage runs={runs} loading={loadingRuns} onOpenRun={openRun} /> : null}
        {view === "sources" ? (
          <SourcesPage sources={sources} runCount={runCount} loading={loadingSources} rebuilding={rebuilding} onRefresh={handleRebuild} />
        ) : null}
        {view === "compare" ? (
          <ComparePage runs={runs} selectedRunIds={selectedRunIds} onSelectionChange={(ids) => void handleSelectionChange(ids)} comparison={comparison} />
        ) : null}
        {view === "report" ? <ReportPage stages={reportStages} onOpenRun={openRun} /> : null}
        {view === "run-detail" && detailRun ? (
          <RunDetailPage
            run={detailRun}
            metrics={metrics}
            diagnostics={diagnostics}
            artifacts={artifacts}
            notes={notes}
            onCreateNote={handleCreateNote}
            onUpdateNote={handleUpdateNote}
          />
        ) : null}
        {view === "run-detail" && !detailRun ? <Placeholder title="Run Detail" /> : null}
      </main>
    </div>
  );
}

function Placeholder({ title }: { title: string }) {
  return (
    <section className="panel">
      <h1>{title}</h1>
    </section>
  );
}
