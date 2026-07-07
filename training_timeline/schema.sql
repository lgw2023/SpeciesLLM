CREATE TABLE IF NOT EXISTS runs (
  id TEXT PRIMARY KEY,
  path TEXT NOT NULL,
  real_path TEXT NOT NULL UNIQUE,
  name TEXT NOT NULL,
  source_root TEXT NOT NULL,
  started_at TEXT,
  created_at_utc TEXT,
  mtime REAL NOT NULL,
  model_size TEXT NOT NULL DEFAULT 'unknown',
  experiment_name TEXT NOT NULL DEFAULT '',
  data_recipe TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT 'needs_review',
  status_reason TEXT NOT NULL DEFAULT '',
  summary_title TEXT NOT NULL DEFAULT '',
  summary_one_liner TEXT NOT NULL DEFAULT '',
  git_commit TEXT,
  git_subject TEXT,
  git_dirty INTEGER,
  indexed_at TEXT NOT NULL,
  index_version INTEGER NOT NULL,
  parse_warnings_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS run_configs (
  run_id TEXT NOT NULL,
  source TEXT NOT NULL,
  key TEXT NOT NULL,
  value TEXT NOT NULL,
  value_type TEXT NOT NULL,
  PRIMARY KEY (run_id, source, key),
  FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS metric_series (
  run_id TEXT NOT NULL,
  series_name TEXT NOT NULL,
  step INTEGER NOT NULL,
  epoch REAL,
  value REAL NOT NULL,
  sample_count INTEGER NOT NULL DEFAULT 1,
  aggregation TEXT NOT NULL DEFAULT 'raw',
  FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS metric_summaries (
  run_id TEXT PRIMARY KEY,
  best_loss REAL,
  best_loss_step INTEGER,
  final_loss REAL,
  final_step INTEGER,
  early_loss_mean REAL,
  tail_loss_mean REAL,
  grad_norm_p50 REAL,
  grad_norm_p95 REAL,
  grad_norm_p99 REAL,
  grad_norm_max REAL,
  clip_count INTEGER NOT NULL DEFAULT 0,
  clip_fraction REAL,
  skip_count INTEGER NOT NULL DEFAULT 0,
  skip_fraction REAL,
  row_count INTEGER NOT NULL DEFAULT 0,
  FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS diagnostic_events (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  event_type TEXT NOT NULL,
  severity TEXT NOT NULL,
  title TEXT NOT NULL,
  description TEXT NOT NULL,
  start_step INTEGER,
  end_step INTEGER,
  evidence_json TEXT NOT NULL,
  source_file TEXT,
  created_by TEXT NOT NULL DEFAULT 'auto',
  FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS analysis_notes (
  id TEXT PRIMARY KEY,
  run_id TEXT NOT NULL,
  note_type TEXT NOT NULL,
  title TEXT NOT NULL,
  body TEXT NOT NULL,
  confidence TEXT NOT NULL,
  supersedes_diagnostic_ids TEXT NOT NULL DEFAULT '[]',
  evidence_refs TEXT NOT NULL DEFAULT '[]',
  author TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS artifacts (
  run_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  path TEXT NOT NULL,
  size_bytes INTEGER NOT NULL,
  mtime REAL NOT NULL,
  PRIMARY KEY (run_id, path),
  FOREIGN KEY (run_id) REFERENCES runs(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS index_state (
  run_id TEXT PRIMARY KEY,
  fingerprint TEXT NOT NULL,
  indexed_at TEXT NOT NULL,
  index_version INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_started_at ON runs(started_at);
CREATE INDEX IF NOT EXISTS idx_run_configs_run_key ON run_configs(run_id, key);
CREATE INDEX IF NOT EXISTS idx_metric_series_run_series_step ON metric_series(run_id, series_name, step);
CREATE INDEX IF NOT EXISTS idx_diagnostic_events_run_type ON diagnostic_events(run_id, event_type);
CREATE INDEX IF NOT EXISTS idx_analysis_notes_run_type ON analysis_notes(run_id, note_type);
