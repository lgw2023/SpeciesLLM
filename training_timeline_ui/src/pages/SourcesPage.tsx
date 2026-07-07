import { Database, RefreshCw } from "lucide-react";
import type { SourceInfo } from "../types";

type SourcesPageProps = {
  sources: SourceInfo[];
  runCount: number;
  loading: boolean;
  rebuilding: boolean;
  onRefresh: () => void;
};

export function SourcesPage({ sources, runCount, loading, rebuilding, onRefresh }: SourcesPageProps) {
  return (
    <section className="sources">
      <div className="section-heading">
        <div>
          <h1>Sources</h1>
          <p>{runCount} indexed runs</p>
        </div>
        <button className="icon-button text-button" type="button" onClick={onRefresh} disabled={rebuilding}>
          <RefreshCw aria-hidden="true" size={16} />
          {rebuilding ? "Rebuilding" : "Rebuild"}
        </button>
      </div>

      {loading ? (
        <div className="panel">Loading sources...</div>
      ) : (
        <div className="source-list">
          {sources.map((source) => (
            <div className="source-row" key={source.path}>
              <Database aria-hidden="true" size={18} />
              <span>{source.path}</span>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
