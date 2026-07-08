from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VariableChange:
    label: str
    before: str
    after: str

    @property
    def summary(self) -> str:
        return f"{self.label}: {self.before} -> {self.after}"


@dataclass(frozen=True)
class RunRelationship:
    id: str
    parent_run_id: str
    child_run_id: str
    relationship_type: str
    confidence: str
    change_summary: str
    evidence: list[dict[str, Any]]
    created_by: str = "auto"


LANE_ORDER = ["Scale smoke", "100M data baselines", "500M stability", "LR / schedule", "E2 ablations"]


def refresh_run_relationships(conn: sqlite3.Connection, source_roots: list[Path]) -> int:
    roots = [str(root.resolve()) for root in source_roots]
    if not roots:
        return 0
    runs = [
        dict(row)
        for row in conn.execute(
            f"""
            SELECT *
            FROM runs
            WHERE source_root IN ({','.join('?' for _ in roots)})
            ORDER BY COALESCE(started_at, '9999-12-31T23:59:59'), name
            """,
            tuple(roots),
        ).fetchall()
    ]
    run_ids = [run["id"] for run in runs]
    relationships = infer_run_relationships(conn, runs, source_roots)
    with conn:
        if run_ids:
            conn.execute(
                f"""
                DELETE FROM run_relationships
                WHERE created_by = 'auto'
                  AND child_run_id IN ({','.join('?' for _ in run_ids)})
                """,
                tuple(run_ids),
            )
        conn.executemany(
            """
            INSERT OR REPLACE INTO run_relationships (
              id, parent_run_id, child_run_id, relationship_type, confidence,
              change_summary, evidence_json, created_by, indexed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    edge.id,
                    edge.parent_run_id,
                    edge.child_run_id,
                    edge.relationship_type,
                    edge.confidence,
                    edge.change_summary,
                    json.dumps(edge.evidence, ensure_ascii=False, sort_keys=True),
                    edge.created_by,
                    _utc_now(),
                )
                for edge in relationships
            ],
        )
    return len(relationships)


def infer_run_relationships(conn: sqlite3.Connection, runs: list[dict[str, Any]], source_roots: list[Path]) -> list[RunRelationship]:
    if len(runs) < 2:
        return []
    configs = _configs_by_run(conn, [run["id"] for run in runs])
    context = _evidence_context(source_roots)
    relationships: list[RunRelationship] = []
    for index, child in enumerate(runs[1:], start=1):
        earlier = runs[:index]
        parent = _find_parent(child, earlier, configs)
        changes = _variable_changes(parent, child, configs)
        evidence = _relationship_evidence(conn, parent, child, changes, context)
        relationships.append(
            RunRelationship(
                id=f"{parent['id']}-{child['id']}",
                parent_run_id=parent["id"],
                child_run_id=child["id"],
                relationship_type="inferred_parent",
                confidence=_confidence(changes, evidence),
                change_summary=_change_summary(changes),
                evidence=evidence,
            )
        )
    return relationships


def list_run_relationships(conn: sqlite3.Connection, run_ids: list[str] | None = None) -> list[dict[str, Any]]:
    if run_ids:
        rows = conn.execute(
            f"""
            SELECT rr.*
            FROM run_relationships rr
            WHERE rr.child_run_id IN ({','.join('?' for _ in run_ids)})
            ORDER BY rr.child_run_id, rr.parent_run_id
            """,
            tuple(run_ids),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM run_relationships ORDER BY child_run_id, parent_run_id").fetchall()
    relationships: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["evidence_refs"] = json.loads(item.pop("evidence_json"))
        relationships.append(item)
    return relationships


def _configs_by_run(conn: sqlite3.Connection, run_ids: list[str]) -> dict[str, dict[str, str]]:
    if not run_ids:
        return {}
    rows = conn.execute(
        f"""
        SELECT run_id, source, key, value
        FROM run_configs
        WHERE run_id IN ({','.join('?' for _ in run_ids)})
        """,
        tuple(run_ids),
    ).fetchall()
    configs: dict[str, dict[str, str]] = {run_id: {} for run_id in run_ids}
    for row in rows:
        configs.setdefault(row["run_id"], {})[f"{row['source']}:{row['key']}"] = row["value"]
    return configs


def _find_parent(child: dict[str, Any], candidates: list[dict[str, Any]], configs: dict[str, dict[str, str]]) -> dict[str, Any]:
    best = candidates[-1]
    best_score = float("-inf")
    for index, candidate in enumerate(candidates):
        score = _parent_score(candidate, child, index, configs)
        if score > best_score:
            best = candidate
            best_score = score
    return best


def _parent_score(candidate: dict[str, Any], child: dict[str, Any], index: int, configs: dict[str, dict[str, str]]) -> float:
    candidate_vars = _extract_variables(candidate, configs)
    child_vars = _extract_variables(child, configs)
    score = index * 0.01
    if _lane_for_run(candidate, configs) == _lane_for_run(child, configs):
        score += 5
    if candidate_vars["model"] == child_vars["model"]:
        score += 4
    if candidate_vars["data"] != "unknown" and candidate_vars["data"] == child_vars["data"]:
        score += 4
    if candidate_vars["recipe"] == child_vars["recipe"]:
        score += 2
    for key in ("lr", "loss", "modalities", "shuffle"):
        if candidate_vars[key] and candidate_vars[key] == child_vars[key]:
            score += 2
    score += len(_shared_tags(candidate, child, configs)) * 1.2
    return score


def _variable_changes(
    parent: dict[str, Any],
    child: dict[str, Any],
    configs: dict[str, dict[str, str]],
) -> list[VariableChange]:
    before = _extract_variables(parent, configs)
    after = _extract_variables(child, configs)
    candidates = [
        _changed("model size", before["model"], after["model"], "not recorded"),
        _changed("training data recipe", before["data"], after["data"], "not recorded"),
        _changed("training recipe", before["recipe"], after["recipe"], "baseline from-scratch"),
        _changed("learning rate", before["lr"], after["lr"], "default"),
        _changed("grad clip norm", before["clip"], after["clip"], "default"),
        _changed("grad skip ratio", before["skip_ratio"], after["skip_ratio"], "default"),
        _changed("grad skip max", before["skip_max"], after["skip_max"], "default"),
        _changed("epoch budget", before["epochs"], after["epochs"], "default"),
        _changed("LR decay schedule", before["lr_decay"], after["lr_decay"], "default"),
        _changed("numeric precision", before["precision"], after["precision"], "default"),
        _changed("loss function", before["loss"], after["loss"], "default"),
        _changed("input modalities", before["modalities"], after["modalities"], "baseline tokens"),
        _changed("data order", before["shuffle"], after["shuffle"], "default shard order"),
        _changed("resume source", before["resume"], after["resume"], "from scratch"),
        _changed("resume update step", before["resume_step"], after["resume_step"], "none"),
    ]
    return [change for change in candidates if change is not None]


def _relationship_evidence(
    conn: sqlite3.Connection,
    parent: dict[str, Any],
    child: dict[str, Any],
    changes: list[VariableChange],
    context: dict[str, list[dict[str, str]]],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for change in changes[:6]:
        evidence.append(
            {
                "kind": "run_config",
                "ref": f"{parent['name']} -> {child['name']}",
                "detail": change.summary,
            }
        )
    evidence.extend(_script_evidence(child, changes, context.get("scripts", [])))
    evidence.extend(_git_evidence(child, changes, context.get("git_commits", [])))
    evidence.extend(_context_evidence(child, changes, context.get("memory_records", []), "memory"))
    evidence.extend(_context_evidence(child, changes, context.get("conversation_records", []), "conversation"))
    evidence.extend(_analysis_note_evidence(conn, child["id"]))
    return evidence[:12]


def _evidence_context(source_roots: list[Path]) -> dict[str, list[dict[str, str]]]:
    return {
        "scripts": _collect_script_records(source_roots),
        "git_commits": _collect_git_records(source_roots),
        "memory_records": _collect_memory_records(),
        "conversation_records": _collect_conversation_records(),
    }


def _collect_script_records(source_roots: list[Path]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    relative_dirs = ["work_record", "scripts", "docs", "docs/superpowers"]
    suffixes = {".md", ".sh", ".py", ".json", ".yaml", ".yml", ".txt"}
    for root in source_roots:
        resolved_root = root.resolve()
        for relative_dir in relative_dirs:
            base = resolved_root / relative_dir
            if not base.exists() or not base.is_dir():
                continue
            for path in sorted(base.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in suffixes:
                    continue
                try:
                    if path.stat().st_size > 1_000_000:
                        continue
                    text = path.read_text(encoding="utf-8", errors="replace")
                    records.append({"ref": str(path.relative_to(resolved_root)), "text": f"{path.name}\n{text}".lower()})
                except OSError:
                    continue
    return records


def _collect_git_records(source_roots: list[Path]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for root in source_roots:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(root.resolve()),
                "log",
                "--format=%h%x09%s",
                "--max-count=80",
                "--",
                "work_record",
                "scripts",
                "docs",
                "training_timeline",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            continue
        for line in result.stdout.splitlines():
            if "\t" not in line:
                continue
            commit, subject = line.split("\t", 1)
            records.append({"ref": commit, "text": subject.lower(), "detail": subject})
    return records


def _collect_memory_records() -> list[dict[str, str]]:
    root = Path(os.environ.get("TRAINING_TIMELINE_MEMORY_ROOT", Path.home() / ".codex" / "memories"))
    if not root.exists():
        return []
    records: list[dict[str, str]] = []
    for path in [root / "MEMORY.md", root / "memory_summary.md"]:
        record = _small_text_record(path, root)
        if record is not None:
            records.append(record)
    summaries = root / "rollout_summaries"
    if summaries.exists():
        for path in sorted(summaries.glob("*.md")):
            record = _small_text_record(path, root)
            if record is None:
                continue
            if any(token in record["text"] for token in ["speciesllm", "training timeline", "stage2_speciesllmdata"]):
                records.append(record)
            if len(records) >= 12:
                break
    return records


def _collect_conversation_records() -> list[dict[str, str]]:
    root = Path(os.environ.get("TRAINING_TIMELINE_SESSIONS_ROOT", Path.home() / ".codex" / "sessions"))
    if not root.exists():
        return []
    pattern = "Training Timeline|training_timeline|训练时间线|训练实验|训练目录|半自动诊断|data_1_2_3|data_1_3|stable_lr|E2_huber"
    try:
        result = subprocess.run(
            ["rg", "-n", pattern, str(root)],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return _collect_conversation_records_without_rg(root)
    if result.returncode not in (0, 1):
        return _collect_conversation_records_without_rg(root)
    records: list[dict[str, str]] = []
    for line in result.stdout.splitlines()[:160]:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        raw_path, line_no, text = parts
        path = Path(raw_path)
        try:
            ref = f"{path.relative_to(root)}:{line_no}"
        except ValueError:
            ref = f"{path.name}:{line_no}"
        records.append({"ref": ref, "text": text.lower(), "detail": _shorten(text)})
    return records


def _collect_conversation_records_without_rg(root: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    needles = ["training timeline", "training_timeline", "训练时间线", "训练实验", "data_1_2_3", "data_1_3"]
    for path in sorted(root.rglob("*.jsonl"), reverse=True):
        record = _small_text_record(path, root, max_size=500_000)
        if record is None:
            continue
        if any(needle in record["text"] for needle in needles):
            records.append(record)
        if len(records) >= 20:
            break
    return records


def _small_text_record(path: Path, root: Path, *, max_size: int = 1_000_000) -> dict[str, str] | None:
    try:
        if not path.exists() or path.stat().st_size > max_size:
            return None
        text = path.read_text(encoding="utf-8", errors="replace")
        return {"ref": str(path.relative_to(root)), "text": text.lower(), "detail": _shorten(text)}
    except OSError:
        return None


def _script_evidence(child: dict[str, Any], changes: list[VariableChange], records: list[dict[str, str]]) -> list[dict[str, Any]]:
    keywords = _evidence_keywords(child, changes)
    if not keywords:
        return []
    evidence: list[dict[str, Any]] = []
    for record in records:
        hits = sorted({keyword for keyword in keywords if keyword in record["text"]})
        if len(hits) < 2:
            continue
        evidence.append({"kind": "script", "ref": record["ref"], "detail": f"mentions {', '.join(hits[:5])}"})
        if len(evidence) >= 4:
            break
    return evidence


def _git_evidence(child: dict[str, Any], changes: list[VariableChange], records: list[dict[str, str]]) -> list[dict[str, Any]]:
    keywords = _evidence_keywords(child, changes)
    evidence: list[dict[str, Any]] = []
    for record in records:
        hits = sorted({keyword for keyword in keywords if keyword in record["text"]})
        if not hits:
            continue
        evidence.append({"kind": "git_commit", "ref": record["ref"], "detail": record.get("detail", "")})
        if len(evidence) >= 3:
            break
    return evidence


def _context_evidence(child: dict[str, Any], changes: list[VariableChange], records: list[dict[str, str]], kind: str) -> list[dict[str, Any]]:
    keywords = _evidence_keywords(child, changes)
    evidence: list[dict[str, Any]] = []
    for record in records:
        hits = sorted({keyword for keyword in keywords if keyword in record["text"]})
        if len(hits) < 2:
            continue
        evidence.append({"kind": kind, "ref": record["ref"], "detail": f"mentions {', '.join(hits[:5])}"})
        if len(evidence) >= 2:
            break
    return evidence


def _analysis_note_evidence(conn: sqlite3.Connection, child_run_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT id, note_type, title, confidence
        FROM analysis_notes
        WHERE run_id = ?
        ORDER BY updated_at DESC, id
        LIMIT 3
        """,
        (child_run_id,),
    ).fetchall()
    return [
        {
            "kind": "analysis_note",
            "ref": row["id"],
            "detail": f"{row['note_type']} ({row['confidence']}): {row['title']}",
        }
        for row in rows
    ]


def _evidence_keywords(child: dict[str, Any], changes: list[VariableChange]) -> set[str]:
    keywords = {_normalize_token(str(child.get("model_size") or "")), _normalize_token(str(child.get("data_recipe") or ""))}
    keywords.update(_tokens(str(child.get("experiment_name") or "")))
    keywords.update(_tokens(str(child.get("name") or "")))
    for change in changes:
        keywords.update(_tokens(change.label))
        keywords.update(_tokens(change.after))
        if change.label == "data order":
            keywords.add("shuffle_rows")
        if change.label == "learning rate":
            keywords.add("lr")
    return {keyword for keyword in keywords if len(keyword) >= 2 and keyword not in {"training", "output", "from", "scratch", "stable"}}


def _extract_variables(run: dict[str, Any], configs: dict[str, dict[str, str]]) -> dict[str, str]:
    args = _run_args(run, configs)
    text = _run_text(run, configs, args)
    tags = _run_tags(run, configs)
    return {
        "model": str(run.get("model_size") or "unknown"),
        "data": str(run.get("data_recipe") or "unknown"),
        "recipe": _training_recipe_name(text, tags),
        "lr": _format_arg_value(args.get("LEARNING_RATE"))
        or _format_learning_rate(_first_match([*tags, text], r"lr(\d+)em(\d+)"))
        or _format_learning_rate(_first_match([*tags, text], r"lr(\d+)e(\d+)"))
        or _format_raw_learning_rate(_first_match([text], r"lr[^0-9]{0,20}([0-9]+(?:\.[0-9]+)?e-?\d+)")),
        "clip": _format_arg_value(args.get("GRAD_CLIP")) or _format_clip(_first_match([*tags, text], r"clip(\d+)p(\d+)")),
        "skip_ratio": _format_arg_value(args.get("GRAD_SKIP_RATIO")) or (_first_match([*tags, text], r"skip(\d+)") or ["", ""])[1],
        "skip_max": _format_arg_value(args.get("GRAD_SKIP_MAX")) or (_first_match([*tags, text], r"skip\d+_(\d+)") or ["", ""])[1],
        "epochs": (_first_match([*tags, text], r"(?:epoch|epochs)(\d+)") or ["", ""])[1]
        or (_first_match([*tags, text], r"(\d+)epoch") or ["", ""])[1]
        or _format_arg_value(args.get("EPOCH")),
        "lr_decay": _format_arg_value(args.get("LR_DECAY_EPOCHS")) or _format_lr_decay(_first_match([*tags, text], r"lrdecay(\d+)")),
        "precision": _format_arg_value(args.get("AMP_DTYPE")) or ("fp32" if "fp32" in text or "fp32" in tags else ""),
        "loss": _loss_name(text, tags),
        "modalities": _format_modalities(args.get("GENE_EMBEDDING_MODALITIES")) or _modality_name(text, tags),
        "shuffle": "all rows" if args.get("TRAIN_SHUFFLE_ROWS", "").lower() == "true" or "shuffleall" in text or "shuffle_rows" in text else "",
        "resume": "checkpoint resume" if _is_resume(args, text) else "",
        "resume_step": _nonzero_arg(args.get("RESUME_UPDATE_STEP")),
    }


def _run_text(run: dict[str, Any], configs: dict[str, dict[str, str]], args: dict[str, str]) -> str:
    pieces = [str(run.get("name") or ""), str(run.get("experiment_name") or ""), str(run.get("data_recipe") or "")]
    pieces.extend(_run_tags(run, configs))
    pieces.extend(f"{key}={value}" for key, value in args.items() if key not in {"INIT_MODEL_PATH", "INIT_OPTIMIZER_PATH"})
    return " ".join(pieces).lower()


def _run_args(run: dict[str, Any], configs: dict[str, dict[str, str]]) -> dict[str, str]:
    raw = configs.get(run["id"], {}).get("run_record:argv", "[]")
    try:
        argv = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(argv, list):
        return {}
    values: dict[str, str] = {}
    for item in argv:
        if not isinstance(item, str) or "=" not in item:
            continue
        key, value = item.split("=", 1)
        if re.fullmatch(r"[A-Z_][A-Z0-9_]*", key):
            values[key] = value
    return values


def _run_tags(run: dict[str, Any], configs: dict[str, dict[str, str]]) -> list[str]:
    raw = configs.get(run["id"], {}).get("directory_name:tags", "[]")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if not isinstance(value, list):
        return []
    return [str(item).lower() for item in value]


def _changed(label: str, before: str, after: str, default_before: str) -> VariableChange | None:
    if not after or after == "unknown" or before == after:
        return None
    before_label = default_before if not before or before == "unknown" else before
    return VariableChange(label, before_label, after)


def _change_summary(changes: list[VariableChange]) -> str:
    if not changes:
        return "repeat run / validation only"
    visible = [change.summary for change in changes[:4]]
    hidden_count = len(changes) - len(visible)
    if hidden_count > 0:
        visible.append(f"+{hidden_count} more")
    return " | ".join(visible)


def _confidence(changes: list[VariableChange], evidence: list[dict[str, Any]]) -> str:
    evidence_kinds = {item["kind"] for item in evidence}
    if len(changes) >= 2 and ({"script", "git_commit", "analysis_note"} & evidence_kinds):
        return "high"
    if changes:
        return "medium"
    return "low"


def _lane_for_run(run: dict[str, Any], configs: dict[str, dict[str, str]]) -> str:
    text = _run_text(run, configs, _run_args(run, configs))
    if run.get("model_size") == "500m" or "stab_smoke" in text or "clip0p5" in text:
        return "500M stability"
    if any(token in text for token in ["huber", "fp32", "esm2", "dnaseq", "lossw", "shuffle"]):
        return "E2 ablations"
    if any(token in text for token in ["lr", "epoch", "lrdecay"]):
        return "LR / schedule"
    if run.get("model_size") == "1b" or ("test_from_scratch" in text and not run.get("data_recipe")):
        return "Scale smoke"
    return "100M data baselines"


def _shared_tags(a: dict[str, Any], b: dict[str, Any], configs: dict[str, dict[str, str]]) -> list[str]:
    ignored = {"from_scratch"}
    a_tags = {tag for tag in _run_tags(a, configs) if tag not in ignored}
    return [tag for tag in _run_tags(b, configs) if tag in a_tags]


def _training_recipe_name(text: str, tags: list[str]) -> str:
    if "stab_smoke" in text:
        return "stability smoke diagnostic"
    if "stable" in text or "stable" in tags:
        return "stable pretrain recipe"
    return "baseline from-scratch"


def _loss_name(text: str, tags: list[str]) -> str:
    if "huber5" in text or "huber5" in tags:
        return "Huber5"
    if "huber" in text or "huber" in tags:
        return "Huber"
    if "lossw_gepc01" in text or "lossw_gepc01" in tags:
        return "GEPC loss weight 0.1"
    return ""


def _modality_name(text: str, tags: list[str]) -> str:
    modalities = []
    if "esm2" in text or "esm2" in tags:
        modalities.append("ESM2")
    if "dnaseq" in text or "dnaseq" in tags:
        modalities.append("DNAseq")
    return " + ".join(modalities)


def _first_match(values: list[str], pattern: str) -> re.Match[str] | None:
    compiled = re.compile(pattern)
    for value in values:
        match = compiled.search(value)
        if match:
            return match
    return None


def _format_learning_rate(match: re.Match[str] | None) -> str:
    if match is None:
        return ""
    return f"{match.group(1)}e-{match.group(2)}"


def _format_arg_value(value: str | None) -> str:
    if value is None:
        return ""
    stripped = str(value).strip()
    return "" if stripped in {"", "0", "0.0", "false", "False"} else stripped


def _format_raw_learning_rate(match: re.Match[str] | None) -> str:
    if match is None:
        return ""
    return match.group(1).replace("e--", "e-")


def _format_clip(match: re.Match[str] | None) -> str:
    if match is None:
        return ""
    return f"{match.group(1)}.{match.group(2)}"


def _format_lr_decay(match: re.Match[str] | None) -> str:
    if match is None:
        return ""
    return f"lrdecay{match.group(1)}"


def _format_modalities(value: str | None) -> str:
    if not value:
        return ""
    names = []
    for item in value.split(","):
        token = item.strip().lower()
        if token == "esm2":
            names.append("ESM2")
        elif token == "dnaseq":
            names.append("DNAseq")
        elif token:
            names.append(token)
    return " + ".join(names)


def _is_resume(args: dict[str, str], text: str) -> bool:
    return bool(args.get("INIT_MODEL_PATH") or args.get("INIT_OPTIMIZER_PATH") or _nonzero_arg(args.get("RESUME_UPDATE_STEP")) or "recovery" in text)


def _nonzero_arg(value: str | None) -> str:
    if value is None:
        return ""
    stripped = str(value).strip()
    return "" if stripped in {"", "0", "0.0"} else stripped


def _tokens(value: str) -> set[str]:
    return {_normalize_token(token) for token in re.split(r"[^A-Za-z0-9_.-]+", value.lower()) if token}


def _normalize_token(value: str) -> str:
    return value.strip().lower()


def _shorten(text: str, limit: int = 220) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    return compact if len(compact) <= limit else f"{compact[: limit - 1]}…"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
