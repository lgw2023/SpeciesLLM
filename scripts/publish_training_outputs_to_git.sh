#!/usr/bin/env bash
set -euo pipefail

# Build git-friendly mirrors of local training_output* directories.
#
# Local training_output/ and training_output_*/ stay in .gitignore. This script
# writes companion directories named training_output_<run>_text_split/ that contain:
#   - losslessly compressed log/metrics text (no checkpoints/weights)
#   - optional plot snapshots under plots/
#
# Collaborators can restore the original text files with:
#   python scripts/archive_training_output_text.py split-unpack \
#     training_output_<run>_text_split --output-dir .

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -P)"
ARCHIVE_SCRIPT="${PROJECT_ROOT}/scripts/archive_training_output_text.py"

PYTHON_BIN="${PYTHON_BIN:-python3}"
JOBS="${JOBS:-8}"
CHUNK_SIZE="${CHUNK_SIZE:-80KB}"
OVERWRITE="${OVERWRITE:-0}"
GIT_ADD="${GIT_ADD:-0}"
DRY_RUN="${DRY_RUN:-0}"
INCLUDE_PLOTS="${INCLUDE_PLOTS:-1}"

PLOT_GLOBS=(
  "training_curves.png"
  "training_curves.pdf"
  "loss_detail.png"
  "loss_detail.pdf"
  "step_timing.png"
  "step_timing.pdf"
)

usage() {
  cat <<'EOF'
Usage: bash scripts/publish_training_outputs_to_git.sh [options] [training_output_dir ...]

Options:
  --jobs N          Parallel compression workers (default: 8)
  --chunk-size SZ   Max split chunk size, e.g. 80KB (default: 80KB)
  --overwrite       Rebuild existing *_text_split directories
  --no-plots        Skip copying plot png/pdf snapshots
  --git-add         Stage generated *_text_split directories for commit
  --dry-run         Print actions without writing files
  -h, --help        Show this help

With no directories given, all training_output* directories in the repo root
are processed, except existing *_text_split mirrors.
EOF
}

log() {
  printf '[INFO] %s\n' "$*"
}

warn() {
  printf '[WARN] %s\n' "$*" >&2
}

run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[DRY-RUN]'
    printf ' %q' "$@"
    printf '\n'
    return 0
  fi
  "$@"
}

has_shareable_text() {
  local input_dir="$1"
  compgen -G "${input_dir}/log"*.txt >/dev/null \
    || compgen -G "${input_dir}/loss_to_log"*.txt >/dev/null \
    || compgen -G "${input_dir}/metrics"*.jsonl >/dev/null
}

copy_plots() {
  local input_dir="$1"
  local output_dir="$2"
  local plots_dir="${output_dir}/plots"
  local copied=0

  for pattern in "${PLOT_GLOBS[@]}"; do
    local matches=()
    shopt -s nullglob
    matches=("${input_dir}/${pattern}")
    shopt -u nullglob
    for src in "${matches[@]}"; do
      [[ -f "$src" ]] || continue
      if [[ "$DRY_RUN" == "1" ]]; then
        log "would copy plot ${src} -> ${plots_dir}/$(basename "$src")"
      else
        mkdir -p "$plots_dir"
        cp -f "$src" "${plots_dir}/$(basename "$src")"
      fi
      copied=$((copied + 1))
    done
  done

  if [[ "$copied" -eq 0 ]]; then
    warn "no plot files found under ${input_dir}"
  else
    log "copied ${copied} plot file(s) to ${plots_dir}"
  fi
}

publish_one() {
  local input_dir="$1"
  local base_name
  base_name="$(basename "${input_dir%/}")"
  local output_dir="${PROJECT_ROOT}/${base_name}_text_split"

  if [[ "$base_name" == *_text_split ]]; then
    warn "skip mirror directory: ${input_dir}"
    return 0
  fi

  if ! has_shareable_text "$input_dir"; then
    warn "skip ${input_dir}: no log/loss_to_log/metrics text files"
    return 0
  fi

  if [[ -e "$output_dir" && "$OVERWRITE" != "1" ]]; then
    log "skip existing mirror (use --overwrite): ${output_dir}"
    return 0
  fi

  log "split-pack ${input_dir} -> ${output_dir}"
  local pack_cmd=(
    "$PYTHON_BIN" "$ARCHIVE_SCRIPT" split-pack "$input_dir"
    --output-dir "$output_dir"
    --chunk-size "$CHUNK_SIZE"
    --jobs "$JOBS"
  )
  if [[ "$OVERWRITE" == "1" ]]; then
    pack_cmd+=(--overwrite)
  fi
  run_cmd "${pack_cmd[@]}"

  if [[ "$INCLUDE_PLOTS" == "1" ]]; then
    copy_plots "$input_dir" "$output_dir"
  fi

  if [[ "$DRY_RUN" != "1" ]]; then
    run_cmd "$PYTHON_BIN" "$ARCHIVE_SCRIPT" split-verify "$output_dir"
    du -sh "$output_dir"
  fi
}

discover_dirs() {
  local dir
  shopt -s nullglob
  for dir in "${PROJECT_ROOT}/training_output" "${PROJECT_ROOT}"/training_output_*; do
    [[ -d "$dir" ]] || continue
    basename "${dir%/}"
  done | sort -u
}

main() {
  local positional=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --jobs)
        JOBS="$2"
        shift 2
        ;;
      --chunk-size)
        CHUNK_SIZE="$2"
        shift 2
        ;;
      --overwrite)
        OVERWRITE=1
        shift
        ;;
      --no-plots)
        INCLUDE_PLOTS=0
        shift
        ;;
      --git-add)
        GIT_ADD=1
        shift
        ;;
      --dry-run)
        DRY_RUN=1
        shift
        ;;
      -h | --help)
        usage
        exit 0
        ;;
      --)
        shift
        positional+=("$@")
        break
        ;;
      -*)
        echo "unknown option: $1" >&2
        usage >&2
        exit 2
        ;;
      *)
        positional+=("$1")
        shift
        ;;
    esac
  done

  if [[ ${#positional[@]} -eq 0 ]]; then
    local discovered
    while IFS= read -r discovered; do
      [[ -n "$discovered" ]] || continue
      positional+=("$discovered")
    done < <(discover_dirs)
  fi

  if [[ ${#positional[@]} -eq 0 ]]; then
    warn "no training_output directories found under ${PROJECT_ROOT}"
    exit 0
  fi

  local name input_dir
  for name in "${positional[@]}"; do
    if [[ "$name" = /* ]]; then
      input_dir="$name"
    else
      input_dir="${PROJECT_ROOT}/${name}"
    fi
    if [[ ! -d "$input_dir" ]]; then
      warn "skip missing directory: ${input_dir}"
      continue
    fi
    publish_one "$input_dir"
  done

  if [[ "$GIT_ADD" == "1" ]]; then
    local split_dir
    shopt -s nullglob
    for split_dir in "${PROJECT_ROOT}"/training_output_*_text_split; do
      [[ -d "$split_dir" ]] || continue
      log "git add ${split_dir}"
      run_cmd git -C "$PROJECT_ROOT" add "$split_dir"
    done
    shopt -u nullglob
  fi
}

main "$@"
