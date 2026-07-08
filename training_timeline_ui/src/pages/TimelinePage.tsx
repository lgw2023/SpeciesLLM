import { CalendarClock, GitBranch } from "lucide-react";
import { RunStatusBadge } from "../components/RunStatusBadge";
import { bestRunSummary } from "../display";
import { TIMELINE_COLUMN_GAP, TIMELINE_LEFT_PAD, TIMELINE_NODE_HEIGHT, TIMELINE_NODE_WIDTH, TIMELINE_TOP_PAD, buildTimelineGraph } from "../timelineGraph";
import type { RunRelationship, RunSummary } from "../types";

type TimelinePageProps = {
  runs: RunSummary[];
  relationships?: RunRelationship[];
  loading: boolean;
  onOpenRun: (runId: string) => void;
};

export function TimelinePage({ runs, relationships = [], loading, onOpenRun }: TimelinePageProps) {
  const sortedRuns = [...runs].sort((a, b) => (a.started_at ?? "").localeCompare(b.started_at ?? ""));
  const graph = buildTimelineGraph(sortedRuns, relationships);
  const nodesById = new Map(graph.nodes.map((node) => [node.id, node]));

  if (loading) {
    return <section className="panel">Loading timeline...</section>;
  }

  return (
    <section className="timeline">
      <div className="section-heading">
        <div>
          <h1>Training Timeline</h1>
          <p>
            {sortedRuns.length} experiment nodes | {graph.edges.length} inferred relations
          </p>
        </div>
      </div>

      {sortedRuns.length === 0 ? (
        <div className="empty-state">No indexed runs yet.</div>
      ) : (
        <div className="timeline-graph-shell">
          <div className="timeline-graph-hint">
            <GitBranch aria-hidden="true" size={16} />
            Older experiments start at the top; arrows flow downward, and labels spell out concrete changes in data, model, recipe, and training parameters.
          </div>
          <div className="timeline-graph-scroll" aria-label="Training experiment relationship graph">
            <div className="timeline-graph-canvas" style={{ width: graph.width, height: graph.height }}>
              <svg className="timeline-edges" width={graph.width} height={graph.height} aria-hidden="true">
                <defs>
                  <marker id="timeline-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
                    <path d="M 0 0 L 10 5 L 0 10 z" />
                  </marker>
                </defs>
                {graph.lanes.map((lane, laneIndex) => (
                  <line
                    className="timeline-lane-line"
                    key={lane}
                    x1={laneCenterX(laneIndex)}
                    y1={TIMELINE_TOP_PAD - 34}
                    x2={laneCenterX(laneIndex)}
                    y2={graph.height - 26}
                  />
                ))}
                {graph.edges.map((edge) => {
                  const from = nodesById.get(edge.fromId);
                  const to = nodesById.get(edge.toId);
                  if (!from || !to) {
                    return null;
                  }
                  return <path className="timeline-edge" key={edge.id} d={edgePath(from.x, from.y, to.x, to.y)} markerEnd="url(#timeline-arrow)" />;
                })}
              </svg>
              {graph.lanes.map((lane, laneIndex) => (
                <div className="timeline-lane-label" key={lane} style={{ left: laneX(laneIndex), top: 18, width: TIMELINE_NODE_WIDTH }}>
                  {lane}
                </div>
              ))}
              {graph.edges.map((edge) => {
                const from = nodesById.get(edge.fromId);
                const to = nodesById.get(edge.toId);
                if (!from || !to) {
                  return null;
                }
                const position = edgeLabelPosition(from.x, from.y, to.x, to.y, graph.width);
                return (
                  <div className="timeline-edge-label" key={edge.id} style={{ left: position.x, top: position.y }}>
                    {edge.label}
                    {edge.confidence && edge.evidenceCount ? (
                      <span className="timeline-edge-meta">
                        {edge.confidence} confidence | {edge.evidenceCount} evidence refs
                      </span>
                    ) : null}
                  </div>
                );
              })}
              {graph.nodes.map((node) => (
                <button
                  aria-label={`Open experiment ${node.run.experiment_name || node.run.name}`}
                  className="timeline-node"
                  key={node.id}
                  style={{ left: node.x, top: node.y, width: TIMELINE_NODE_WIDTH, height: TIMELINE_NODE_HEIGHT }}
                  type="button"
                  onClick={() => onOpenRun(node.id)}
                >
                  <div className="run-time">
                    <CalendarClock aria-hidden="true" size={15} />
                    {formatTime(node.run.started_at)}
                  </div>
                  <h2>{node.run.experiment_name || node.run.name}</h2>
                  <p className="summary-text">{bestRunSummary(node.run.summary_one_liner, node.run.status_reason)}</p>
                  <div className="timeline-node-footer">
                    <div className="tag-row">
                      <span>{node.run.model_size}</span>
                      <span>{node.run.data_recipe || "unknown data"}</span>
                    </div>
                    <RunStatusBadge status={node.run.status} />
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

function edgePath(fromX: number, fromY: number, toX: number, toY: number) {
  const startX = fromX + TIMELINE_NODE_WIDTH / 2;
  const startY = fromY + TIMELINE_NODE_HEIGHT;
  const endX = toX + TIMELINE_NODE_WIDTH / 2;
  const endY = toY;
  const curve = Math.max(70, Math.min(180, (endY - startY) / 2));
  return `M ${startX} ${startY} C ${startX} ${startY + curve}, ${endX} ${endY - curve}, ${endX} ${endY}`;
}

function edgeLabelPosition(fromX: number, fromY: number, toX: number, toY: number, graphWidth: number) {
  const labelWidth = 320;
  const startX = fromX + TIMELINE_NODE_WIDTH / 2;
  const endX = toX + TIMELINE_NODE_WIDTH / 2;
  const idealX = (startX + endX) / 2 - labelWidth / 2;
  const minX = 12;
  const maxX = Math.max(minX, graphWidth - labelWidth - 12);
  return {
    x: Math.min(Math.max(idealX, minX), maxX),
    y: Math.max(fromY + TIMELINE_NODE_HEIGHT + 12, toY - 112),
  };
}

function laneX(laneIndex: number) {
  return TIMELINE_LEFT_PAD + laneIndex * (TIMELINE_NODE_WIDTH + TIMELINE_COLUMN_GAP);
}

function laneCenterX(laneIndex: number) {
  return laneX(laneIndex) + TIMELINE_NODE_WIDTH / 2;
}

function formatTime(value?: string | null) {
  if (!value) {
    return "Unknown start";
  }
  return value.replace("T", " ").slice(0, 16);
}
