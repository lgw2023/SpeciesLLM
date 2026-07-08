from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from training_timeline.relationships import (
    _change_summary,
    _configs_by_run,
    _evidence_context,
    _evidence_keywords,
    _extract_variables,
    _tokens,
    _variable_changes,
    list_run_relationships,
)


AUTO_CONTEXT_AUTHOR = "auto-context"
AUTO_CONTEXT_TITLE = "Auto context inference"


def refresh_auto_context_notes(conn: sqlite3.Connection, source_roots: list[Path]) -> int:
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
    if not runs:
        return 0
    configs = _configs_by_run(conn, [run["id"] for run in runs])
    run_by_id = {run["id"]: run for run in runs}
    incoming = {item["child_run_id"]: item for item in list_run_relationships(conn, [run["id"] for run in runs])}
    context = _evidence_context(source_roots)
    notes = [_build_note(conn, run, run_by_id, incoming, configs, context) for run in runs]
    now = _utc_now()
    with conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO analysis_notes (
              id, run_id, note_type, title, body, confidence,
              supersedes_diagnostic_ids, evidence_refs, author, created_at, updated_at
            ) VALUES (?, ?, 'curated_note', ?, ?, ?, '[]', ?, ?, ?, ?)
            """,
            [
                (
                    _note_id(note["run_id"]),
                    note["run_id"],
                    AUTO_CONTEXT_TITLE,
                    note["body"],
                    note["confidence"],
                    json.dumps(note["evidence_refs"], ensure_ascii=False),
                    AUTO_CONTEXT_AUTHOR,
                    now,
                    now,
                )
                for note in notes
            ],
        )
    return len(notes)


def _build_note(
    conn: sqlite3.Connection,
    run: dict[str, Any],
    run_by_id: dict[str, dict[str, Any]],
    incoming: dict[str, dict[str, Any]],
    configs: dict[str, dict[str, str]],
    context: dict[str, list[dict[str, str]]],
) -> dict[str, Any]:
    variables = _extract_variables(run, configs)
    relationship = incoming.get(run["id"])
    relationship_summary = "时间线起点；暂无可确认父实验。"
    changes = []
    if relationship is not None:
        parent = run_by_id.get(relationship["parent_run_id"])
        relationship_summary = f"由 {relationship['parent_run_id']} 推进而来；{relationship['change_summary']}。"
        if parent is not None:
            changes = _variable_changes(parent, run, configs)
            relationship_summary = f"由 {parent['name']} 推进而来；{_change_summary(changes)}。"
    metric_summary = _metric_summary(conn, run["id"])
    evidence_refs = _evidence_refs(run, relationship, changes, context)
    body = "\n".join(
        [
            f"实验定位：{_intent(run, variables)}",
            f"训练设置：{_settings_summary(variables)}",
            f"结果线索：{_metric_line(metric_summary)}",
            f"关系推断：{relationship_summary}",
            "证据边界：该笔记由目录名、run_record、summary/metrics、脚本、Git、记忆和对话命中项自动生成；结论仍需人工 deep review 确认。",
        ]
    )
    return {
        "run_id": run["id"],
        "body": body,
        "confidence": _confidence(evidence_refs, relationship),
        "evidence_refs": evidence_refs,
    }


def _intent(run: dict[str, Any], variables: dict[str, str]) -> str:
    text = f"{run.get('name', '')} {run.get('experiment_name', '')}".lower()
    if "test_from_scratch" in text:
        return "规模启动 smoke，用于确认不同参数量的基础训练链路和日志/指标可用。"
    if run.get("model_size") == "500m" and "stab" in text:
        return "500M 稳定性诊断或修复验证，重点观察梯度裁剪、skip fuse 和主任务 loss。"
    if "lr" in text or variables["lr"]:
        return "学习率或训练计划 sweep，用于比较稳定配方下的优化速度与稳定性。"
    if "e2" in text or variables["loss"] or variables["modalities"]:
        return "E2 loss / 输入模态消融，用于评估 loss 形式、fp32、ESM2/DNAseq 和 GEPC 权重影响。"
    if variables["recipe"] == "stable pretrain recipe":
        return "稳定预训练配方基线，用于承接早期数据配方并作为后续 sweep 的对照。"
    if variables["data"] != "unknown":
        return "数据配方基线，用于比较 Stage2 不同数据组合对训练曲线的影响。"
    return "基础训练实验，用于补齐时间线中的配置或启动状态。"


def _settings_summary(variables: dict[str, str]) -> str:
    parts = [
        f"model={variables['model']}",
        f"data={variables['data']}",
        f"recipe={variables['recipe']}",
    ]
    for key, label in [
        ("lr", "lr"),
        ("clip", "clip"),
        ("skip_ratio", "skip_ratio"),
        ("skip_max", "skip_max"),
        ("epochs", "epochs"),
        ("lr_decay", "lr_decay"),
        ("precision", "precision"),
        ("loss", "loss"),
        ("modalities", "modalities"),
        ("shuffle", "shuffle"),
        ("resume", "resume"),
        ("resume_step", "resume_step"),
    ]:
        if variables[key]:
            parts.append(f"{label}={variables[key]}")
    return ", ".join(parts)


def _metric_summary(conn: sqlite3.Connection, run_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM metric_summaries WHERE run_id = ?", (run_id,)).fetchone()
    return dict(row) if row is not None else None


def _metric_line(summary: dict[str, Any] | None) -> str:
    if summary is None or summary.get("row_count", 0) == 0:
        return "未索引到可用 metrics，当前只能依据配置和外部记录推断。"
    parts = [f"metric rows={summary['row_count']}"]
    if summary.get("best_loss") is not None:
        parts.append(f"best_loss={summary['best_loss']:.4g}@step{summary.get('best_loss_step')}")
    if summary.get("final_loss") is not None:
        parts.append(f"final_loss={summary['final_loss']:.4g}@step{summary.get('final_step')}")
    if summary.get("clip_fraction") is not None:
        parts.append(f"clip_fraction={summary['clip_fraction']:.3g}")
    if summary.get("skip_fraction") is not None:
        parts.append(f"skip_fraction={summary['skip_fraction']:.3g}")
    return ", ".join(parts)


def _evidence_refs(
    run: dict[str, Any],
    relationship: dict[str, Any] | None,
    changes: list[Any],
    context: dict[str, list[dict[str, str]]],
) -> list[str]:
    refs: list[str] = [f"run:{run['name']}"]
    if relationship is not None:
        refs.extend(f"{item['kind']}:{item['ref']}" for item in relationship.get("evidence_refs", [])[:8])
    keywords = _note_keywords(run, changes)
    for context_key, prefix in [
        ("scripts", "script"),
        ("git_commits", "git_commit"),
        ("memory_records", "memory"),
        ("conversation_records", "conversation"),
    ]:
        for record in context.get(context_key, []):
            hits = {keyword for keyword in keywords if keyword in record["text"]}
            if len(hits) < 2:
                continue
            refs.append(f"{prefix}:{record['ref']}")
            break
    return _dedupe(refs)[:12]


def _note_keywords(run: dict[str, Any], changes: list[Any]) -> set[str]:
    keywords = _evidence_keywords(run, changes)
    keywords.update(_tokens(str(run.get("name") or "")))
    keywords.update(_tokens(str(run.get("experiment_name") or "")))
    keywords.update(_tokens(str(run.get("data_recipe") or "")))
    return {keyword for keyword in keywords if len(keyword) >= 2}


def _confidence(evidence_refs: list[str], relationship: dict[str, Any] | None) -> str:
    ref_text = " ".join(evidence_refs)
    if relationship is not None and all(prefix in ref_text for prefix in ["script:", "git_commit:"]):
        return "high"
    if relationship is not None or len(evidence_refs) >= 3:
        return "medium"
    return "low"


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _note_id(run_id: str) -> str:
    return f"auto-context-{run_id}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
