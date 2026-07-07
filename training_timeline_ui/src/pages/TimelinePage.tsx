import { CalendarClock, ChevronRight } from "lucide-react";
import type { RunSummary } from "../types";
import { RunStatusBadge } from "../components/RunStatusBadge";

type TimelinePageProps = {
  runs: RunSummary[];
  loading: boolean;
  onOpenRun: (runId: string) => void;
};

export function TimelinePage({ runs, loading, onOpenRun }: TimelinePageProps) {
  const sortedRuns = [...runs].sort((a, b) => (a.started_at ?? "").localeCompare(b.started_at ?? ""));

  if (loading) {
    return <section className="panel">Loading timeline...</section>;
  }

  return (
    <section className="timeline">
      <div className="section-heading">
        <div>
          <h1>Training Timeline</h1>
          <p>{sortedRuns.length} indexed runs</p>
        </div>
      </div>

      {sortedRuns.length === 0 ? (
        <div className="empty-state">No indexed runs yet.</div>
      ) : (
        <div className="run-list">
          {sortedRuns.map((run) => (
            <button className="run-card" key={run.id} type="button" onClick={() => onOpenRun(run.id)}>
              <div className="run-card-main">
                <div className="run-time">
                  <CalendarClock aria-hidden="true" size={16} />
                  {formatTime(run.started_at)}
                </div>
                <h2>{run.experiment_name || run.name}</h2>
                <p>{run.summary_one_liner || run.status_reason || "Needs review"}</p>
                <div className="tag-row">
                  <span>{run.model_size}</span>
                  <span>{run.data_recipe || "unknown data"}</span>
                  {run.tags.slice(0, 4).map((tag) => (
                    <span key={tag}>{tag}</span>
                  ))}
                </div>
              </div>
              <div className="run-card-side">
                <span className="preliminary-label">Preliminary</span>
                <RunStatusBadge status={run.status} />
                <ChevronRight aria-hidden="true" size={18} />
              </div>
            </button>
          ))}
        </div>
      )}
    </section>
  );
}

function formatTime(value?: string | null) {
  if (!value) {
    return "Unknown start";
  }
  return value.replace("T", " ").slice(0, 16);
}
