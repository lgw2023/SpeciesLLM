# Agent Notes

This machine is a code development workstation only.

Large project datasets are not present locally and should be assumed unavailable.
Missing real dataset files, checkpoints, indexes, generated parquet shards, or large
artifacts are expected on this machine and should not be treated as evidence that
the codebase is broken.

Target training / data-processing nodes are much larger than this workstation.
Assume each real node has:

- 1.5 TB RAM.
- Local storage directories `/data/disk1`, `/data/disk2`, and `/data/disk3`.
- About 7 TB capacity per local storage directory.
- 8 Huawei Ascend NPU devices.
- 64 GB device memory per NPU.

When working in this repository:

- Prefer static code inspection, lightweight unit tests, mocks, fixtures, and small
  synthetic samples.
- Do not download, copy, generate, or expect large datasets unless explicitly asked.
- Do not run full training, full evaluation, full indexing, or other data-heavy
  pipelines locally.
- If a task needs real data, ask for the expected schema, file layout, path
  contract, or a small representative sample.
- Treat dataset paths, checkpoint paths, generated artifacts, and environment
  variables such as `DATA_ROOT` as deployment-specific.
- Code intended for real training or preprocessing may assume the target node
  hardware above, but local validation on this workstation must remain lightweight.
- Keep local validation commands safe for a development workstation. Use tiny test
  data or repository-provided fixture generators when available.
- When adding tests for data-dependent code, make them use synthetic data or small
  fixtures that can run without the real datasets.
