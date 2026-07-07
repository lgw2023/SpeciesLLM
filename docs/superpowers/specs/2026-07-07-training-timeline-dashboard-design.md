# SpeciesLLM Training Timeline Dashboard Design

Date: 2026-07-07

## Purpose

Build a local web application that reconstructs the SpeciesLLM training history from existing training output directories. The first version serves two users:

- The project owner, who needs fast experiment review, configuration comparison, curve inspection, and evidence drill-down.
- Internal reviewers, who need a clear narrative of the training process: what was tried, what failed, what was fixed, and why the next experiment followed.

The application is an offline retrospective tool. It does not run training, monitor live server jobs, download datasets, or treat missing large artifacts on this workstation as errors.

## Chosen Approach

Use a Python FastAPI backend, a React/Vite frontend, and a local SQLite index.

The backend scans training output directories, parses available evidence, computes summaries and preliminary diagnostics, and writes a rebuildable SQLite index. The frontend reads only API responses, not raw multi-megabyte JSONL logs.

The source of truth remains the original training directories. SQLite stores derived metadata, summaries, downsampled curves, diagnostics, artifact references, and curated analysis notes.

## Scope

First-version data sources:

- Default source: the SpeciesLLM repository root.
- Additional source roots: user-configurable local directories containing training outputs copied from other machines.
- Candidate run directories: `training_output*`.
- Explicitly out of first-version core input: `*_text_split` archive mirrors. They can be supported later.

First-version capabilities:

- Offline scanning and incremental indexing.
- Timeline overview of existing training runs.
- Single-run detail pages.
- Multi-run comparison.
- Report-oriented training-story view.
- Preliminary automatic diagnostics with evidence.
- A reserved curated/deep-analysis layer for manually reviewed conclusions.

Non-goals:

- Realtime training monitor.
- Remote server file synchronization.
- Full training, evaluation, checkpoint loading, or large dataset inspection.
- Automatic final scientific judgment for every run.

## Architecture

Data flow:

```text
training_output directories
  -> scanner/parsers
  -> metrics summarizer and diagnostics
  -> SQLite index
  -> FastAPI
  -> React/Vite UI
```

Backend responsibilities:

- Discover candidate run directories under the repo root and configured extra roots.
- Resolve symlinks and avoid duplicate indexing of the same real path.
- Parse directory names, `run_record.json`, `summary.md`, `metrics.*.jsonl`, logs, plots, and checkpoint filenames.
- Prefer rank-0 metrics when available, with fallback to another rank when needed.
- Deduplicate repeated `update_step` values introduced by resume runs.
- Downsample metric curves before serving them to the frontend.
- Run conservative diagnostic rules and store their evidence.
- Store curated and deep-review notes separately from automatic diagnostics.

Frontend responsibilities:

- Render the timeline, run detail, comparison, report, and data-source views.
- Make evidence traceable from every generated conclusion.
- Clearly label automatic diagnostics as preliminary and requiring review.
- Provide a place for curated or deep-review analysis to override or supplement automatic diagnostics.

## Page Structure

### Timeline

The timeline is the default page and sorts runs by inferred start time, not directory name.

Each run card shows:

- Experiment name from directory name and `run_record` when available.
- Start time from directory timestamp, `run_record.created_at_utc`, and first metric/log timestamp.
- Model size: `100m`, `500m`, `1b`, or `unknown`.
- Data recipe: examples include `data_1`, `data_1_3`, `data_1_2_3`, and `shuffleall`.
- Tags such as LR sweep, stability incident, resume, Huber, fp32, modality ablation, loss weight, and row shuffle.
- Preliminary status: success, failure, stopped, partial, or needs review.
- A one-line conclusion from `summary.md` when available, otherwise an automatic draft.

The timeline groups runs into training phases:

- Initial smoke and model-scale tests.
- Data 1 / 1+3 / 1+2+3 comparisons.
- 500M stability incident and fixes.
- 100M learning-rate sweep.
- Multi-epoch and resume experiments.
- fp32 + Huber, modality, loss-weight, and shuffle experiments.

### Run Detail

The run detail page follows a conclusion-first structure:

- Top: status, best loss, final loss, major turning point, primary diagnosis, and curated/deep-review notes if present.
- Middle: loss curves, component losses, learning rate, gradient norm, clip/skip behavior, throughput, MFU, and timing.
- Bottom: `run_record.json`, `summary.md`, selected log snippets, artifact paths, git commit, and source file references.

### Compare

The compare view supports selecting multiple runs and generates:

- Configuration diff for learning rate, min LR, epoch count, batch size, precision, loss weights, active gene embedding modalities, shuffle settings, and gradient-clip settings.
- Curve overlays for total loss, GEP, zero-prob, GEPC, learning rate, raw gradient norm, clip fraction, and skip fraction where data exists.
- Result matrix with best loss, final loss, completion status, plateau/failure markers, and major diagnostics.
- Sweep-level summaries such as "1e-6 is the only complete healthy control" when evidence supports them.

### Report

The report view presents the training history as a narrative:

- What question this phase tried to answer.
- Which experiments were run.
- What was observed.
- What code or configuration changes followed.
- Why the next experiment was chosen.

Report conclusions link back to run detail evidence. The report view hides most raw file detail by default.

### Sources

The sources page shows:

- Configured source roots.
- Last indexing time.
- Run count by source.
- Parse warnings and skipped directories.
- Per-run refresh controls.

## Data Model

### `runs`

One row per training directory.

Key fields:

- `id`: stable hash from real path and run name.
- `path`, `real_path`, `name`, `source_root`.
- `started_at`, `created_at_utc`, `mtime`.
- `model_size`, `experiment_name`, `data_recipe`.
- `status`, `status_reason`.
- `summary_title`, `summary_one_liner`.
- `git_commit`, `git_subject`, `git_dirty`.
- `indexed_at`, `index_version`, `parse_warnings_count`.

### `run_configs`

Key-value records extracted from `run_record.json`, log args, and directory-name parsing.

Fields:

- `run_id`
- `source`
- `key`
- `value`
- `value_type`

### `metric_series`

Downsampled curve data.

Fields:

- `run_id`
- `series_name`
- `step`
- `epoch`
- `value`
- `sample_count`
- `aggregation`

Examples of `series_name`: `loss_total`, `loss_gep`, `loss_zero_prob`, `loss_gepc`, `lr`, `grad_norm_raw`, `clip_fraction`, `skip_fraction`, `samples_per_s`, `mfu`.

### `metric_summaries`

Per-run aggregate metrics.

Fields include:

- Best loss and best-loss step.
- Final loss and final step.
- Early-window, best-window, and tail-window means.
- First threshold-crossing steps.
- Gradient norm p50, p95, p99, and max.
- Clip and skip counts and fractions.
- Resume and completion markers.

### `diagnostic_events`

Preliminary automatic diagnostic events.

Fields:

- `run_id`
- `event_type`
- `severity`
- `title`
- `description`
- `start_step`, `end_step`
- `evidence_json`
- `source_file`
- `created_by = auto`

### `analysis_notes`

Curated or deep-review conclusions that supplement or override automatic diagnostics.

Fields:

- `run_id`
- `note_type`: `curated_note` or `deep_review`.
- `title`
- `body`
- `confidence`: `low`, `medium`, or `high`.
- `supersedes_diagnostic_ids`
- `evidence_refs`
- `author`
- `created_at`, `updated_at`

This table is required because each training task can be complex. Automatic rules are useful triage, but final interpretation may require reading logs, summaries, configurations, and curves run by run.

### `artifacts`

References to files under a run directory.

Fields:

- `run_id`
- `kind`
- `path`
- `size_bytes`
- `mtime`

Artifact kinds include `summary_md`, `run_record_json`, `metrics_jsonl`, `loss_log`, `log`, `training_curve_png`, `loss_detail_png`, `grad_clip_png`, and `checkpoint`.

## Indexing Flow

1. Load the configured source roots.
2. Discover candidate directories matching `training_output*`.
3. Skip empty directories, `*_text_split` mirrors, and directories without shareable training evidence.
4. Resolve real paths and deduplicate symlinks.
5. Generate stable run IDs.
6. Extract metadata from the directory name, `run_record.json`, logs, and `summary.md`.
7. Read metrics from rank 0 when available, with fallback to another rank.
8. Deduplicate repeated resume steps using `update_step`, keeping the last occurrence.
9. Compute metric summaries and downsampled metric series.
10. Run preliminary diagnostic rules.
11. Write changes to SQLite in one transaction per run.
12. Record parse warnings without failing the entire index build.

Incremental indexing uses file size and mtime as the default change detector. If a run changes or the index schema version changes, that run is rebuilt. SHA-256 verification is optional and can be added for stricter modes.

## Preliminary Diagnostics

Automatic diagnostics are conservative and evidence-bound. They are never presented as final scientific conclusions.

First-version event types:

- `converged`: loss falls substantially from early to late windows and tail windows are stable.
- `bad_plateau`: loss reaches a lower point, rebounds, and remains on a high plateau.
- `clip_storm`: rolling clip fraction exceeds a threshold while loss worsens.
- `skip_loop`: repeated or continuous skip behavior indicates stalled optimizer updates.
- `primary_head_failure`: GEPC-style heads improve while GEP or zero-prob does not.
- `lr_floor_freeze`: LR reaches the floor while loss stops improving.
- `resume_boundary`: resume markers exist and pre/post-resume windows are checked for continuity.
- `sweep_winner`: cross-run comparison identifies the healthiest run in a controlled sweep.

Default thresholds:

- Rolling window: 100 logged loss rows or about 1000 update steps.
- Clip storm: clip fraction above 50 percent with simultaneous or following loss increase.
- Skip loop: more than 50 consecutive skips or skip fraction above 30 percent in a window.
- Bad plateau: tail loss more than 2 times the best-window loss for multiple windows.
- Convergence: tail loss below 30 percent of early loss and low tail volatility.

Each diagnostic stores the threshold values it used in `evidence_json`.

## Backend Modules

- `scanner`: source-root loading, directory discovery, symlink handling, skip rules.
- `parsers`: directory-name parser, run-record parser, summary parser, metrics parser, artifact parser.
- `indexer`: incremental rebuilds, transactions, schema management, metric summaries, downsampling.
- `diagnostics`: automatic diagnostic rules and cross-run sweep summaries.
- `analysis`: curated and deep-review notes.
- `api`: FastAPI routes.

## API Shape

Core endpoints:

```text
GET  /api/health
GET  /api/sources
POST /api/index/rebuild
POST /api/index/runs/{run_id}/refresh

GET  /api/runs
GET  /api/runs/{run_id}
GET  /api/runs/{run_id}/metrics?series=loss_total,loss_gep,lr
GET  /api/runs/{run_id}/configs
GET  /api/runs/{run_id}/diagnostics
GET  /api/runs/{run_id}/artifacts
GET  /api/runs/{run_id}/evidence

POST /api/compare
GET  /api/report/timeline
GET  /api/report/stages

GET   /api/runs/{run_id}/analysis
POST  /api/runs/{run_id}/analysis
PATCH /api/runs/{run_id}/analysis/{analysis_id}
```

Evidence responses return bounded text snippets or structured file metadata, not unbounded raw file contents.

## Frontend Modules

- `TimelinePage`
- `RunDetailPage`
- `ComparePage`
- `ReportPage`
- `SourcesPage`
- `AnalysisPanel`
- Shared chart components for line charts, event bands, threshold overlays, and run comparison tables.

The UI should clearly distinguish:

- Raw evidence.
- Automatic diagnostics.
- Curated notes.
- Deep-review conclusions.

## Testing Strategy

Backend tests:

- Directory scanning detects repo-root and extra-root training outputs.
- Scanner skips empty directories, `*_text_split` mirrors, and unrelated directories.
- Parsers tolerate missing `run_record.json`, missing `summary.md`, missing plots, and partial rank files.
- Metrics parser handles 35-65 MB rank-0 JSONL files without loading all source roots into memory at once.
- Resume deduplication keeps the last row for duplicate `update_step` values.
- Incremental indexing skips unchanged runs and rebuilds changed runs.
- Diagnostic regression fixtures use representative existing runs:
  - `training_output_100m_data_1_2_3_stable_from_scratch_20260514_011839` for stable convergence.
  - `training_output_500m_data_1_2_3_stable_from_scratch_20260515_195223` for skip loop and primary-head failure candidates.
  - `training_output_100m_data_1_2_3_stable_lr5em4_from_scratch_20260523_130351` for bad plateau.
  - `training_output_100m_data_1_2_3_stable_5epoch_lrdecay1_from_scratch_20260526_172620` for LR floor freeze and resume boundary health.
  - `training_output_100m_data_1_3_E2_huber5_fp32_esm2_dnaseq_lossw_gepc01_shuffleall_from_scratch_20260622_162826` for clip storm and GEP rebound candidates.

Frontend checks:

- Timeline sorts by inferred start time.
- Run cards show time, model size, data recipe, tags, preliminary status, and summary.
- Run detail loads curves, configs, diagnostics, artifacts, and evidence references.
- Compare view supports multiple selected runs and shows config diffs plus curve overlays.
- Report view groups runs into training phases and links conclusions back to evidence.
- Automatic diagnostics are visibly labeled as preliminary.
- Analysis panel can display curated and deep-review notes.

## Acceptance Criteria

The first version is complete when a local user can start the app, rebuild the index, and inspect the existing May-June 2026 training history without opening raw log files manually.

The app must make the main experiment arc visible:

- Initial 100M/500M/1B smoke and scale tests.
- Data-combination experiments.
- 500M stability incident and gradient-control follow-up.
- 100M learning-rate sweep.
- Multi-epoch and resume behavior.
- fp32 + Huber experiments.
- Gene embedding modality and loss-weight changes.
- Row-shuffle run and its clip/loss issue.

Every automatic conclusion must link to supporting evidence. Complex run interpretation must be editable or supplementable through curated/deep-review analysis notes.

## Operational Constraints

- Do not run full training or evaluation locally.
- Do not expect large datasets, generated parquet shards, or checkpoints to exist.
- Do not modify raw training output directories during indexing.
- Keep frontend data requests bounded and use downsampled curves by default.
- Keep all file reads inside configured source roots.
