#!/usr/bin/env python3
"""Pack and unpack large training-output text logs losslessly.

The archive is a single .tar.xz file containing only the selected text outputs
plus a manifest with SHA-256 checksums. File contents are handled as raw bytes,
so no encoding assumptions are made.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import io
import json
import lzma
import os
from pathlib import Path
import stat
import sys
import tarfile
from typing import BinaryIO, Iterable


DEFAULT_PATTERNS = ("log*txt", "loss_to_log*txt", "metrics*jsonl")
MANIFEST_NAME = ".speciesllm_training_output_text_manifest.json"
ARCHIVE_FORMAT_VERSION = 1
COPY_BUFFER_SIZE = 1024 * 1024


class ArchiveError(RuntimeError):
    """Raised for expected archive/CLI failures."""


class HashingReader:
    """File-like wrapper that updates a sha256 digest as tarfile reads."""

    def __init__(self, fileobj: BinaryIO) -> None:
        self._fileobj = fileobj
        self._sha256 = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        data = self._fileobj.read(size)
        if data:
            self._sha256.update(data)
        return data

    @property
    def hexdigest(self) -> str:
        return self._sha256.hexdigest()


def parse_patterns(value: str | None) -> tuple[str, ...]:
    if not value:
        return DEFAULT_PATTERNS
    patterns = tuple(item.strip() for item in value.split(",") if item.strip())
    if not patterns:
        raise ArchiveError("--patterns must contain at least one glob")
    return patterns


def iter_matching_files(
    root: Path, patterns: tuple[str, ...], recursive: bool
) -> Iterable[Path]:
    candidates = root.rglob("*") if recursive else root.iterdir()
    for path in candidates:
        if path.is_file() and any(fnmatch.fnmatch(path.name, pat) for pat in patterns):
            yield path


def portable_path(path: Path) -> str:
    return path.as_posix()


def safe_output_path(output_dir: Path, member_name: str) -> Path:
    member_path = Path(member_name)
    if member_path.is_absolute() or ".." in member_path.parts:
        raise ArchiveError(f"unsafe archive path: {member_name!r}")
    output_path = output_dir / member_path
    try:
        output_path.resolve().relative_to(output_dir.resolve())
    except ValueError as exc:
        raise ArchiveError(f"unsafe archive path: {member_name!r}") from exc
    return output_path


def build_tarinfo(path: Path, arcname: str) -> tarfile.TarInfo:
    st = path.stat()
    tarinfo = tarfile.TarInfo(arcname)
    tarinfo.size = st.st_size
    tarinfo.mode = stat.S_IMODE(st.st_mode)
    tarinfo.mtime = int(st.st_mtime)
    tarinfo.uid = 0
    tarinfo.gid = 0
    tarinfo.uname = ""
    tarinfo.gname = ""
    return tarinfo


def build_manifest(
    *,
    source_root: Path,
    root_dir_name: str | None,
    patterns: tuple[str, ...],
    files: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "format": "speciesllm-training-output-text-archive",
        "version": ARCHIVE_FORMAT_VERSION,
        "source_root": str(source_root),
        "root_dir_name": root_dir_name,
        "patterns": list(patterns),
        "file_count": len(files),
        "files": files,
    }


def pack(args: argparse.Namespace) -> int:
    root = args.input_dir.resolve()
    if not root.is_dir():
        raise ArchiveError(f"input directory does not exist: {root}")

    patterns = parse_patterns(args.patterns)
    files = sorted(iter_matching_files(root, patterns, args.recursive))
    if not files and not args.allow_empty:
        raise ArchiveError(
            f"no files matched {', '.join(patterns)} under {root}; "
            "pass --allow-empty if this is expected"
        )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.overwrite:
        raise ArchiveError(f"output already exists: {output} (pass --overwrite)")

    root_dir_name = None if args.no_root_dir else root.name
    manifest_files: list[dict[str, object]] = []
    total_input_bytes = 0
    preset = args.preset | (lzma.PRESET_EXTREME if args.extreme else 0)

    with output.open("wb") as raw_out:
        with lzma.LZMAFile(raw_out, "wb", preset=preset) as xz_out:
            with tarfile.open(fileobj=xz_out, mode="w|", format=tarfile.PAX_FORMAT) as tar:
                for path in files:
                    rel = path.relative_to(root)
                    arc_path = Path(root.name) / rel if root_dir_name else rel
                    arcname = portable_path(arc_path)
                    tarinfo = build_tarinfo(path, arcname)
                    with path.open("rb") as raw_in:
                        hashing_reader = HashingReader(raw_in)
                        tar.addfile(tarinfo, hashing_reader)
                    manifest_files.append(
                        {
                            "path": arcname,
                            "source_relative_path": portable_path(rel),
                            "size": tarinfo.size,
                            "mode": tarinfo.mode,
                            "mtime": tarinfo.mtime,
                            "sha256": hashing_reader.hexdigest,
                        }
                    )
                    total_input_bytes += tarinfo.size

                manifest = build_manifest(
                    source_root=root,
                    root_dir_name=root_dir_name,
                    patterns=patterns,
                    files=manifest_files,
                )
                manifest_bytes = (
                    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
                    + "\n"
                ).encode("utf-8")
                manifest_info = tarfile.TarInfo(MANIFEST_NAME)
                manifest_info.size = len(manifest_bytes)
                manifest_info.mode = 0o644
                manifest_info.mtime = 0
                manifest_info.uid = 0
                manifest_info.gid = 0
                tar.addfile(manifest_info, io.BytesIO(manifest_bytes))

    compressed_bytes = output.stat().st_size
    ratio = compressed_bytes / total_input_bytes if total_input_bytes else 0.0
    print(
        f"packed {len(files)} files, "
        f"{total_input_bytes:,} bytes -> {compressed_bytes:,} bytes "
        f"({ratio:.2%})"
    )
    print(output)
    return 0


def read_member_bytes(tar: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    fileobj = tar.extractfile(member)
    if fileobj is None:
        raise ArchiveError(f"cannot read archive member: {member.name}")
    return fileobj.read()


def write_member_file(
    tar: tarfile.TarFile,
    member: tarfile.TarInfo,
    output_path: Path,
    overwrite: bool,
) -> str:
    if member.isdir():
        output_path.mkdir(parents=True, exist_ok=True)
        return hashlib.sha256(b"").hexdigest()
    if not member.isfile():
        raise ArchiveError(f"unsupported archive member type: {member.name}")
    if output_path.exists() and not overwrite:
        raise ArchiveError(f"refusing to overwrite existing file: {output_path}")

    fileobj = tar.extractfile(member)
    if fileobj is None:
        raise ArchiveError(f"cannot read archive member: {member.name}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with output_path.open("wb") as out:
        while True:
            chunk = fileobj.read(COPY_BUFFER_SIZE)
            if not chunk:
                break
            digest.update(chunk)
            out.write(chunk)
    os.chmod(output_path, member.mode & 0o777)
    os.utime(output_path, (member.mtime, member.mtime))
    return digest.hexdigest()


def load_archive(
    archive_path: Path,
    *,
    output_dir: Path | None,
    overwrite: bool = False,
    list_only: bool = False,
    verify_only: bool = False,
) -> dict[str, object]:
    if not archive_path.is_file():
        raise ArchiveError(f"archive does not exist: {archive_path}")

    output_root = output_dir.resolve() if output_dir else None
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] | None = None
    observed: dict[str, dict[str, object]] = {}

    with archive_path.open("rb") as raw_in:
        with lzma.LZMAFile(raw_in, "rb") as xz_in:
            with tarfile.open(fileobj=xz_in, mode="r|") as tar:
                for member in tar:
                    if member.name == MANIFEST_NAME:
                        manifest = json.loads(read_member_bytes(tar, member).decode("utf-8"))
                        continue

                    if list_only:
                        if member.isfile():
                            observed[member.name] = {"size": member.size}
                        continue

                    if output_root is None and not verify_only:
                        raise ArchiveError("output_dir is required unless list_only is set")

                    if verify_only:
                        digest = hashlib.sha256()
                        fileobj = tar.extractfile(member)
                        if fileobj is None:
                            raise ArchiveError(f"cannot read archive member: {member.name}")
                        while True:
                            chunk = fileobj.read(COPY_BUFFER_SIZE)
                            if not chunk:
                                break
                            digest.update(chunk)
                        observed[member.name] = {
                            "size": member.size,
                            "sha256": digest.hexdigest(),
                        }
                    else:
                        assert output_root is not None
                        output_path = safe_output_path(output_root, member.name)
                        digest_hex = write_member_file(tar, member, output_path, overwrite)
                        observed[member.name] = {
                            "size": member.size,
                            "sha256": digest_hex,
                            "output_path": str(output_path),
                        }

    if manifest is None:
        raise ArchiveError(f"archive is missing {MANIFEST_NAME}")

    expected_files = manifest.get("files")
    if not isinstance(expected_files, list):
        raise ArchiveError("archive manifest is malformed: files must be a list")
    if int(manifest.get("version", -1)) != ARCHIVE_FORMAT_VERSION:
        raise ArchiveError(f"unsupported archive version: {manifest.get('version')}")

    expected_by_path = {}
    for item in expected_files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ArchiveError("archive manifest is malformed: invalid file entry")
        expected_by_path[item["path"]] = item

    if set(observed) != set(expected_by_path):
        missing = sorted(set(expected_by_path) - set(observed))
        extra = sorted(set(observed) - set(expected_by_path))
        raise ArchiveError(f"archive file set mismatch; missing={missing}, extra={extra}")

    for path, expected in expected_by_path.items():
        observed_item = observed[path]
        if observed_item["size"] != expected["size"]:
            raise ArchiveError(f"size mismatch for {path}")
        expected_sha = expected.get("sha256")
        observed_sha = observed_item.get("sha256")
        if expected_sha and observed_sha and observed_sha != expected_sha:
            raise ArchiveError(f"sha256 mismatch for {path}")

    return manifest


def unpack(args: argparse.Namespace) -> int:
    manifest = load_archive(
        args.archive.resolve(),
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(f"unpacked and verified {manifest['file_count']} files")
    print(args.output_dir.resolve())
    return 0


def verify(args: argparse.Namespace) -> int:
    manifest = load_archive(
        args.archive.resolve(),
        output_dir=None,
        overwrite=False,
        verify_only=True,
    )
    print(f"verified {manifest['file_count']} files")
    return 0


def list_archive(args: argparse.Namespace) -> int:
    manifest = load_archive(args.archive.resolve(), output_dir=None, list_only=True)
    files = manifest["files"]
    assert isinstance(files, list)
    total_size = sum(int(item["size"]) for item in files if isinstance(item, dict))
    print(f"{manifest['file_count']} files, {total_size:,} uncompressed bytes")
    for item in files:
        assert isinstance(item, dict)
        print(f"{int(item['size']):>12}  {item['path']}")
    return 0


def add_common_archive_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("archive", type=Path, help="Path to the .tar.xz archive.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Losslessly compress/decompress SpeciesLLM training_output text files "
            "(log*txt, loss_to_log*txt, metrics*jsonl)."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    pack_parser = subparsers.add_parser("pack", help="Create a compressed archive.")
    pack_parser.add_argument("input_dir", type=Path, help="training_output directory.")
    pack_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output archive path, for example /tmp/training_output_text.tar.xz.",
    )
    pack_parser.add_argument(
        "--patterns",
        help=(
            "Comma-separated basename globs. Defaults to "
            f"{','.join(DEFAULT_PATTERNS)}."
        ),
    )
    pack_parser.add_argument(
        "--preset",
        type=int,
        choices=range(0, 10),
        default=9,
        metavar="0-9",
        help="LZMA compression preset. Default: 9.",
    )
    pack_parser.add_argument(
        "--no-extreme",
        dest="extreme",
        action="store_false",
        help="Disable LZMA extreme mode.",
    )
    pack_parser.add_argument(
        "--non-recursive",
        dest="recursive",
        action="store_false",
        help="Only scan files directly under input_dir.",
    )
    pack_parser.add_argument(
        "--no-root-dir",
        action="store_true",
        help="Do not store the input directory basename in archive paths.",
    )
    pack_parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow creating an archive when no files match.",
    )
    pack_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing output archive.",
    )
    pack_parser.set_defaults(func=pack, recursive=True, extreme=True)

    unpack_parser = subparsers.add_parser("unpack", help="Extract and verify an archive.")
    add_common_archive_arg(unpack_parser)
    unpack_parser.add_argument(
        "-C",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where files should be restored.",
    )
    unpack_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing restored files.",
    )
    unpack_parser.set_defaults(func=unpack)

    verify_parser = subparsers.add_parser("verify", help="Verify archive contents.")
    add_common_archive_arg(verify_parser)
    verify_parser.add_argument(
        "-C",
        "--output-dir",
        type=Path,
        required=False,
        help="Ignored; accepted for command symmetry.",
    )
    verify_parser.set_defaults(func=verify)

    list_parser = subparsers.add_parser("list", help="List archived files.")
    add_common_archive_arg(list_parser)
    list_parser.set_defaults(func=list_archive)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ArchiveError, OSError, lzma.LZMAError, tarfile.TarError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
