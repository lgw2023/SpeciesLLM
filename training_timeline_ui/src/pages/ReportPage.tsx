import type { ReportStage } from "../types";
import { bestRunSummary } from "../display";

type ReportPageProps = {
  stages: ReportStage[];
  onOpenRun: (runId: string) => void;
};

export function ReportPage({ stages, onOpenRun }: ReportPageProps) {
  return (
    <section className="report-page">
      <div className="section-heading">
        <div>
          <h1>Report</h1>
          <p>{stages.length} phases</p>
        </div>
      </div>
      <div className="stage-list">
        {stages.map((stage) => (
          <section className="stage-section" key={stage.name}>
            <h2>{stage.name}</h2>
            {stage.runs.map((run) => (
              <article className="stage-run" key={run.id}>
                <div>
                  <h3>{run.experiment_name || run.name}</h3>
                  <p className="summary-text">{bestRunSummary(run.summary_one_liner, run.status_reason)}</p>
                </div>
                <button className="icon-button text-button" type="button" onClick={() => onOpenRun(run.id)}>
                  Open evidence
                </button>
              </article>
            ))}
          </section>
        ))}
      </div>
    </section>
  );
}
