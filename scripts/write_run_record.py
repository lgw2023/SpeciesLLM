#!/usr/bin/env python3
"""Write a minimal run_record.json into a training output directory."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shlex
import socket
import subprocess
from pathlib import Path


SENSITIVE_KEYWORDS = ("PASSWORD", "TOKEN", "SECRET", "KEY")


def run_git(repo: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.rstrip("\n")


def redact_arg(arg: str) -> str:
    if "=" not in arg:
        return arg
    key, value = arg.split("=", 1)
    if any(keyword in key.upper() for keyword in SENSITIVE_KEYWORDS):
        return f"{key}=<redacted>"
    return f"{key}={value}"


def resolve_out_path(cwd: Path, out_path: str) -> Path:
    path = Path(out_path)
    if not path.is_absolute():
        path = cwd / path
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, help="Git repository/workdir path")
    parser.add_argument("--out-path", required=True, help="Training output directory")
    parser.add_argument("--script", required=True, help="Launcher script path")
    parser.add_argument("--shell", default="bash", help="Shell used to invoke the launcher")
    parser.add_argument("argv", nargs=argparse.REMAINDER, help="Original launcher arguments after --")
    args = parser.parse_args()

    argv = args.argv
    if argv and argv[0] == "--":
        argv = argv[1:]

    repo = Path(args.repo).resolve()
    out_path = resolve_out_path(repo, args.out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    safe_argv = [redact_arg(arg) for arg in argv]
    command_parts = [args.shell, args.script, *safe_argv]
    command = " ".join(shlex.quote(part) for part in command_parts)

    git_root = run_git(repo, "rev-parse", "--show-toplevel")
    git_repo = Path(git_root).resolve() if git_root else repo
    commit = run_git(git_repo, "rev-parse", "HEAD")
    status_short = run_git(git_repo, "status", "--short")
    diff = run_git(git_repo, "diff", "HEAD", "--no-ext-diff")
    diff_stat = run_git(git_repo, "diff", "HEAD", "--stat")
    commit_subject = run_git(git_repo, "log", "-1", "--pretty=format:%s")
    describe = run_git(git_repo, "describe", "--tags", "--always", "--dirty")
    recent_log = run_git(
        git_repo, "log", "-n", "20", "--date=short", "--pretty=format:%h %ad %s"
    )
    recent_commits = recent_log.split("\n") if recent_log else []

    record = {
        "schema_version": 2,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "cwd": str(repo),
        "out_path": str(out_path),
        "command": command,
        "script": args.script,
        "argv": safe_argv,
        "git": {
            "repo_root": str(git_repo),
            "branch": run_git(git_repo, "branch", "--show-current"),
            "commit": commit,
            "commit_short": commit[:12] if commit else "",
            "commit_subject": commit_subject,
            "describe": describe,
            "recent_commits": recent_commits,
            "is_dirty": bool(status_short),
            "status_short": status_short,
            "diff_stat": diff_stat,
            "diff_sha256": hashlib.sha256(diff.encode("utf-8")).hexdigest() if diff else "",
        },
    }

    output_file = out_path / "run_record.json"
    output_file.write_text(
        json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[INFO] wrote run record: {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
