import { BarChart3, Database, FileText, GitCompare, ListTree } from "lucide-react";
import { useEffect, useState } from "react";
import { fetchRuns, fetchSources, rebuildIndex } from "./api";
import { SourcesPage } from "./pages/SourcesPage";
import { TimelinePage } from "./pages/TimelinePage";
import type { RunSummary, SourceInfo } from "./types";

type View = "timeline" | "sources" | "compare" | "report" | "run-detail";

export default function App() {
  const [view, setView] = useState<View>("timeline");
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [sources, setSources] = useState<SourceInfo[]>([]);
  const [runCount, setRunCount] = useState(0);
  const [loadingRuns, setLoadingRuns] = useState(true);
  const [loadingSources, setLoadingSources] = useState(false);
  const [rebuilding, setRebuilding] = useState(false);
  const [, setSelectedRunId] = useState<string | null>(null);

  useEffect(() => {
    void loadRuns();
  }, []);

  async function loadRuns() {
    setLoadingRuns(true);
    try {
      const response = await fetchRuns();
      setRuns(response.runs);
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
    } finally {
      setLoadingSources(false);
    }
  }

  async function handleRebuild() {
    setRebuilding(true);
    try {
      await rebuildIndex();
      await Promise.all([loadRuns(), loadSources()]);
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
          <button className={view === "report" ? "active" : ""} type="button" onClick={() => setView("report")}>
            <FileText aria-hidden="true" size={18} />
            Report
          </button>
        </nav>
      </aside>
      <main className="main-surface">
        {view === "timeline" ? <TimelinePage runs={runs} loading={loadingRuns} onOpenRun={openRun} /> : null}
        {view === "sources" ? (
          <SourcesPage sources={sources} runCount={runCount} loading={loadingSources} rebuilding={rebuilding} onRefresh={handleRebuild} />
        ) : null}
        {view === "compare" ? <Placeholder title="Compare" /> : null}
        {view === "report" ? <Placeholder title="Report" /> : null}
        {view === "run-detail" ? <Placeholder title="Run Detail" /> : null}
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
