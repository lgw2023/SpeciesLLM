import type { ComparisonResponse, RunSummary } from "../types";

type ComparePageProps = {
  runs: RunSummary[];
  selectedRunIds: string[];
  onSelectionChange: (runIds: string[]) => void;
  comparison: ComparisonResponse | null;
};

export function ComparePage({ runs, selectedRunIds, onSelectionChange, comparison }: ComparePageProps) {
  function toggle(runId: string) {
    const next = selectedRunIds.includes(runId) ? selectedRunIds.filter((id) => id !== runId) : [...selectedRunIds, runId];
    onSelectionChange(next);
  }

  return (
    <section className="compare-page">
      <div className="section-heading">
        <div>
          <h1>Compare</h1>
          <p>{selectedRunIds.length} selected runs</p>
        </div>
      </div>
      <div className="compare-layout">
        <div className="selector-panel">
          {runs.map((run) => (
            <label className="check-row" key={run.id}>
              <input type="checkbox" checked={selectedRunIds.includes(run.id)} onChange={() => toggle(run.id)} />
              <span>{run.experiment_name || run.name}</span>
            </label>
          ))}
        </div>
        <div className="comparison-panel">
          <h2>Configuration diff</h2>
          {(comparison?.config_diffs ?? []).map((diff) => (
            <div className="diff-row" key={diff.key}>
              <strong>{diff.key}</strong>
              {Object.entries(diff.values).map(([runId, value]) => (
                <span key={runId}>{value}</span>
              ))}
            </div>
          ))}
          <h2>Diagnostics</h2>
          <div className="tag-row">
            {(comparison?.diagnostics ?? []).map((event) => (
              <span key={`${event.run_id}-${event.event_type}-${event.title}`}>{event.title}</span>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
