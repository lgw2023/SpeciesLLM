export function cleanDisplayText(value?: string | null): string {
  const text = value ?? "";
  return text
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]*$/g, "$1")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/^\s*>\s?/gm, "")
    .replace(/\s+/g, " ")
    .trim();
}

export function bestRunSummary(summary?: string | null, fallback?: string | null): string {
  return cleanDisplayText(summary) || cleanDisplayText(fallback) || "Needs review";
}
