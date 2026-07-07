import { CheckCircle2, CircleAlert, CircleHelp, PauseCircle } from "lucide-react";

type RunStatusBadgeProps = {
  status: string;
};

export function RunStatusBadge({ status }: RunStatusBadgeProps) {
  const normalized = status || "needs_review";
  const Icon = normalized === "success" ? CheckCircle2 : normalized === "failure" ? CircleAlert : normalized === "partial" ? PauseCircle : CircleHelp;
  return (
    <span className={`status-badge status-${normalized}`}>
      <Icon aria-hidden="true" size={14} />
      {normalized.replace("_", " ")}
    </span>
  );
}
