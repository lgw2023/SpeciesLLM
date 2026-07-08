import type { AnalysisNote, ArtifactRef, DiagnosticEvent, MetricSeries, RunSummary } from "../types";
import { AnalysisPanel } from "../components/AnalysisPanel";
import { MetricChart } from "../components/MetricChart";
import { RunStatusBadge } from "../components/RunStatusBadge";
import { bestRunSummary } from "../display";

type RunDetailPageProps = {
  run: RunSummary;
  metrics: MetricSeries;
  diagnostics: DiagnosticEvent[];
  artifacts: ArtifactRef[];
  notes: AnalysisNote[];
  onCreateNote?: (payload: Partial<AnalysisNote>) => void | Promise<void>;
  onUpdateNote?: (noteId: string, payload: Partial<AnalysisNote>) => void | Promise<void>;
};

export function RunDetailPage({ run, metrics, diagnostics, artifacts, notes, onCreateNote, onUpdateNote }: RunDetailPageProps) {
  return (
    <section className="detail-page">
      <div className="detail-header">
        <div>
          <h1>{run.summary_title || run.experiment_name || run.name}</h1>
          <p className="summary-text">{bestRunSummary(run.summary_one_liner, run.status_reason)}</p>
        </div>
        <RunStatusBadge status={run.status} />
      </div>

      <div className="metric-grid">
        <MetricChart title="Loss curves" series={pickSeries(metrics, ["loss_total", "loss_gep", "loss_zero_prob", "loss_gepc"])} />
        <MetricChart title="Optimization behavior" series={pickSeries(metrics, ["lr", "grad_norm_raw", "clip_fraction", "skip_fraction"])} />
      </div>

      <section className="panel-section">
        <h2>Automatic diagnostics</h2>
        <div className="diagnostic-list">
          {diagnostics.length === 0 ? <div className="empty-state compact">No preliminary diagnostics</div> : null}
          {diagnostics.map((event) => (
            <article className="diagnostic-card" key={event.id ?? `${event.event_type}-${event.title}`}>
              <span className="preliminary-label">Preliminary</span>
              <h3>{event.title}</h3>
              <p>{event.description}</p>
            </article>
          ))}
        </div>
      </section>

      <AnalysisPanel
        runId={run.id}
        diagnostics={diagnostics}
        notes={notes}
        onCreate={onCreateNote ?? (() => undefined)}
        onUpdate={onUpdateNote ?? (() => undefined)}
      />

      <section className="panel-section">
        <h2>Artifacts</h2>
        <div className="artifact-list">
          {artifacts.map((artifact) => (
            <div className="artifact-row" key={`${artifact.kind}-${artifact.path}`}>
              <span>{artifact.kind}</span>
              <code>{artifact.path}</code>
            </div>
          ))}
        </div>
      </section>
    </section>
  );
}

function pickSeries(metrics: MetricSeries, names: string[]): MetricSeries {
  return names.reduce<MetricSeries>((selected, name) => {
    if (metrics[name]) {
      selected[name] = metrics[name];
    }
    return selected;
  }, {});
}
