#!/usr/bin/env python3
"""Pack and unpack large training-output text logs losslessly.

The default archive stores independently compressed per-file xz blocks in a
single tar container. This allows many log files to be compressed in parallel.
File contents are handled as raw bytes, so no encoding assumptions are made.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import fnmatch
import hashlib
import io
import json
import lzma
import os
from pathlib import Path
import re
import stat
import sys
import tarfile
import tempfile
from typing import BinaryIO, Iterable


DEFAULT_PATTERNS = ("log*txt", "loss_to_log*txt", "metrics*jsonl", "run_record.json")
MANIFEST_NAME = ".speciesllm_training_output_text_manifest.json"
LEGACY_ARCHIVE_FORMAT_VERSION = 1
PARALLEL_ARCHIVE_FORMAT_VERSION = 2
SPLIT_ARCHIVE_FORMAT_VERSION = 3
PARALLEL_MEMBER_DIR = ".speciesllm_training_output_text_members"
SPLIT_INDEX_NAME = "speciesllm_training_text_split_index.json"
SPLIT_MANIFEST_PART_PREFIX = "speciesllm_training_text_manifest.part"
SPLIT_DATA_SUFFIX = ".xzpart"
COPY_BUFFER_SIZE = 1024 * 1024
DEFAULT_SPLIT_CHUNK_SIZE = 80_000
XZ_MAGIC = b"\xfd7zXZ\x00"
LOG_LINE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) "
    r"([A-Z]+) node=(\d+) rank=(\d+) local_rank=(\d+) pid=(\d+) (.*)$"
)


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


class HashingWriter:
    """File-like wrapper that updates a sha256 digest as bytes are written."""

    def __init__(self, fileobj: BinaryIO) -> None:
        self._fileobj = fileobj
        self._sha256 = hashlib.sha256()

    def write(self, data: bytes) -> int:
        if data:
            self._sha256.update(data)
        return self._fileobj.write(data)

    def flush(self) -> None:
        self._fileobj.flush()

    @property
    def hexdigest(self) -> str:
        return self._sha256.hexdigest()


class SplitChunkWriter:
    """Write a byte stream as fixed-size chunk files while hashing it."""

    def __init__(
        self,
        output_dir: Path,
        *,
        file_index: int,
        chunk_size: int,
        overwrite: bool,
    ) -> None:
        self._output_dir = output_dir
        self._file_index = file_index
        self._chunk_size = chunk_size
        self._overwrite = overwrite
        self._sha256 = hashlib.sha256()
        self._chunks: list[dict[str, object]] = []
        self._current_file: BinaryIO | None = None
        self._current_size = 0
        self._closed = False

    def _open_next_chunk(self) -> None:
        self.close_current_chunk()
        chunk_index = len(self._chunks)
        name = (
            f"data_{self._file_index:06d}_part_{chunk_index:06d}"
            f"{SPLIT_DATA_SUFFIX}"
        )
        path = self._output_dir / name
        if path.exists() and not self._overwrite:
            raise ArchiveError(f"refusing to overwrite existing file: {path}")
        self._current_file = path.open("wb")
        self._current_size = 0
        self._chunks.append({"name": name, "size": 0})

    def write(self, data: bytes) -> int:
        if self._closed:
            raise ValueError("write to closed SplitChunkWriter")
        if not data:
            return 0

        self._sha256.update(data)
        view = memoryview(data)
        offset = 0
        while offset < len(view):
            if self._current_file is None or self._current_size >= self._chunk_size:
                self._open_next_chunk()
            writable = min(len(view) - offset, self._chunk_size - self._current_size)
            assert self._current_file is not None
            self._current_file.write(view[offset : offset + writable])
            self._current_size += writable
            self._chunks[-1]["size"] = self._current_size
            offset += writable
        return len(data)

    def flush(self) -> None:
        if self._current_file is not None:
            self._current_file.flush()

    def close_current_chunk(self) -> None:
        if self._current_file is not None:
            self._current_file.close()
            self._current_file = None

    def close(self) -> None:
        self.close_current_chunk()
        self._closed = True

    @property
    def chunks(self) -> list[dict[str, object]]:
        return list(self._chunks)

    @property
    def compressed_size(self) -> int:
        return sum(int(chunk["size"]) for chunk in self._chunks)

    @property
    def hexdigest(self) -> str:
        return self._sha256.hexdigest()


class ChunkSequenceReader:
    """Read a sequence of chunk files as one continuous byte stream."""

    def __init__(self, base_dir: Path, chunks: list[dict[str, object]]) -> None:
        self._base_dir = base_dir
        self._chunks = chunks
        self._index = 0
        self._current_file: BinaryIO | None = None

    def _open_next(self) -> bool:
        self.close_current()
        if self._index >= len(self._chunks):
            return False
        name = self._chunks[self._index].get("name")
        if not isinstance(name, str):
            raise ArchiveError("split manifest is malformed: invalid chunk name")
        path = self._base_dir / name
        if not path.is_file():
            raise ArchiveError(f"missing split chunk: {path}")
        expected_size = int(self._chunks[self._index]["size"])
        actual_size = path.stat().st_size
        if actual_size != expected_size:
            raise ArchiveError(f"split chunk size mismatch: {path}")
        self._current_file = path.open("rb")
        self._index += 1
        return True

    def read(self, size: int = -1) -> bytes:
        if size == 0:
            return b""
        parts: list[bytes] = []
        remaining = size
        while size < 0 or remaining > 0:
            if self._current_file is None and not self._open_next():
                break
            assert self._current_file is not None
            chunk = self._current_file.read(-1 if size < 0 else remaining)
            if chunk:
                parts.append(chunk)
                if size > 0:
                    remaining -= len(chunk)
            else:
                self.close_current()
        return b"".join(parts)

    def close_current(self) -> None:
        if self._current_file is not None:
            self._current_file.close()
            self._current_file = None

    def close(self) -> None:
        self.close_current()


def parse_patterns(value: str | None) -> tuple[str, ...]:
    if not value:
        return DEFAULT_PATTERNS
    patterns = tuple(item.strip() for item in value.split(",") if item.strip())
    if not patterns:
        raise ArchiveError("--patterns must contain at least one glob")
    return patterns


def parse_size(value: str | int) -> int:
    if isinstance(value, int):
        size = value
    else:
        text = value.strip().lower()
        multipliers = {
            "kib": 1024,
            "kb": 1000,
            "k": 1000,
            "mib": 1024 * 1024,
            "mb": 1000 * 1000,
            "m": 1000 * 1000,
        }
        multiplier = 1
        for suffix, candidate in sorted(multipliers.items(), key=lambda item: -len(item[0])):
            if text.endswith(suffix):
                text = text[: -len(suffix)]
                multiplier = candidate
                break
        try:
            size = int(float(text) * multiplier)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid size: {value!r}") from exc
    if size < 1:
        raise argparse.ArgumentTypeError("size must be >= 1 byte")
    return size


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
    compression: str,
    version: int,
) -> dict[str, object]:
    return {
        "format": "speciesllm-training-output-text-archive",
        "version": version,
        "compression": compression,
        "source_root": str(source_root),
        "root_dir_name": root_dir_name,
        "patterns": list(patterns),
        "file_count": len(files),
        "files": files,
    }


def pack(args: argparse.Namespace) -> int:
    if args.single_stream:
        return pack_single_stream(args)
    return pack_parallel(args)


def resolve_jobs(requested_jobs: int | None, file_count: int) -> int:
    if file_count <= 1:
        return 1
    if requested_jobs is not None:
        if requested_jobs < 1:
            raise ArchiveError("--jobs must be >= 1")
        return min(requested_jobs, file_count)
    cpu_count = os.cpu_count() or 1
    return min(24, cpu_count, file_count)


def make_archive_name(root: Path, rel: Path, include_root_dir: bool) -> str:
    arc_path = Path(root.name) / rel if include_root_dir else rel
    return portable_path(arc_path)


def compression_preset(args: argparse.Namespace) -> int:
    return args.preset | (lzma.PRESET_EXTREME if args.extreme else 0)


def compress_file_to_xz(task: tuple[int, str, str, str, str, int]) -> dict[str, object]:
    index, source_path_text, arcname, source_relative_path, temp_dir_text, preset = task
    source_path = Path(source_path_text)
    compressed_path = Path(temp_dir_text) / f"{index:06d}.xz"
    st = source_path.stat()
    source_sha256 = hashlib.sha256()
    source_size = 0

    with source_path.open("rb") as raw_in, compressed_path.open("wb") as raw_out:
        hashing_writer = HashingWriter(raw_out)
        with lzma.LZMAFile(hashing_writer, "wb", preset=preset) as xz_out:
            while True:
                chunk = raw_in.read(COPY_BUFFER_SIZE)
                if not chunk:
                    break
                source_sha256.update(chunk)
                source_size += len(chunk)
                xz_out.write(chunk)
        hashing_writer.flush()

    compressed_size = compressed_path.stat().st_size
    return {
        "index": index,
        "path": arcname,
        "source_relative_path": source_relative_path,
        "compressed_path": f"{PARALLEL_MEMBER_DIR}/{index:06d}.xz",
        "tmp_compressed_path": str(compressed_path),
        "size": source_size,
        "compressed_size": compressed_size,
        "mode": stat.S_IMODE(st.st_mode),
        "mtime": int(st.st_mtime),
        "sha256": source_sha256.hexdigest(),
        "compressed_sha256": hashing_writer.hexdigest,
    }


def compress_file_to_split_chunks(
    task: tuple[int, str, str, str, str, int, int, bool, bool]
) -> dict[str, object]:
    (
        index,
        source_path_text,
        arcname,
        source_relative_path,
        output_dir_text,
        preset,
        chunk_size,
        overwrite,
        use_structured,
    ) = task
    source_path = Path(source_path_text)
    output_dir = Path(output_dir_text)
    st = source_path.stat()
    source_data = source_path.read_bytes()
    source_size = len(source_data)
    source_sha256 = hashlib.sha256(source_data).hexdigest()
    codec, encoded_payload = encode_structured_payload(
        source_path, source_data, use_structured
    )
    encoded_sha256 = hashlib.sha256(encoded_payload).hexdigest()
    split_writer = SplitChunkWriter(
        output_dir,
        file_index=index,
        chunk_size=chunk_size,
        overwrite=overwrite,
    )

    try:
        with lzma.LZMAFile(split_writer, "wb", preset=preset) as xz_out:
            xz_out.write(encoded_payload)
    finally:
        split_writer.close()

    return {
        "index": index,
        "path": arcname,
        "source_relative_path": source_relative_path,
        "codec": codec,
        "chunks": split_writer.chunks,
        "size": source_size,
        "encoded_size": len(encoded_payload),
        "encoded_sha256": encoded_sha256,
        "compressed_size": split_writer.compressed_size,
        "mode": stat.S_IMODE(st.st_mode),
        "mtime": int(st.st_mtime),
        "sha256": source_sha256,
        "compressed_sha256": split_writer.hexdigest,
    }


def ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite and any(output_dir.iterdir()):
        raise ArchiveError(f"output directory is not empty: {output_dir}")


def write_split_parts(
    *,
    output_dir: Path,
    prefix: str,
    suffix: str,
    data: bytes,
    chunk_size: int,
    overwrite: bool,
) -> list[dict[str, object]]:
    parts: list[dict[str, object]] = []
    if not data:
        data = b""
    for index, offset in enumerate(range(0, max(len(data), 1), chunk_size)):
        chunk = data[offset : offset + chunk_size]
        name = f"{prefix}{index:06d}{suffix}"
        path = output_dir / name
        if path.exists() and not overwrite:
            raise ArchiveError(f"refusing to overwrite existing file: {path}")
        path.write_bytes(chunk)
        parts.append({"name": name, "size": len(chunk)})
    return parts


def compact_json_bytes(payload: dict[str, object]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def detect_newline(data: bytes) -> str:
    crlf = data.find(b"\r\n")
    lf = data.find(b"\n")
    if crlf != -1 and (lf == -1 or crlf <= lf):
        return "\r\n"
    if lf != -1:
        return "\n"
    return ""


def split_line_bytes(data: bytes) -> list[bytes]:
    if not data:
        return []
    return data.splitlines(keepends=True)


def strip_line_ending(text: str) -> tuple[str, str]:
    if text.endswith("\r\n"):
        return text[:-2], "\r\n"
    if text.endswith("\n"):
        return text[:-1], "\n"
    if text.endswith("\r"):
        return text[:-1], "\r"
    return text, ""


def decode_utf8(data: bytes) -> str:
    return data.decode("utf-8")


def encode_metric_jsonl_payload(data: bytes) -> bytes:
    text = decode_utf8(data)
    newline = detect_newline(data)
    rows: list[list[tuple[str, object]]] = []
    fields: list[str] = []
    field_seen: set[str] = set()
    compact_style: bool | None = None
    trailing_empty_line = False

    lines = text.splitlines()
    if text.endswith("\n") or text.endswith("\r"):
        trailing_empty_line = False
    elif not text:
        lines = []

    for raw_line in lines:
        if raw_line == "":
            trailing_empty_line = True
            continue
        pairs = json.loads(raw_line, object_pairs_hook=list)
        if not isinstance(pairs, list):
            raise ArchiveError("metrics jsonl row is not a JSON object")
        typed_pairs: list[tuple[str, object]] = []
        for key, value in pairs:
            if not isinstance(key, str):
                raise ArchiveError("metrics jsonl key is not a string")
            typed_pairs.append((key, value))
            if key not in field_seen:
                fields.append(key)
                field_seen.add(key)
        if compact_style is None:
            compact_style = '":' in raw_line and '": ' not in raw_line
        rows.append(typed_pairs)

    field_to_index = {field: index for index, field in enumerate(fields)}
    encoded_rows: list[list[object]] = []
    for row in rows:
        indexes = [field_to_index[key] for key, _value in row]
        if indexes != sorted(indexes):
            raise ArchiveError("metrics jsonl field order is not stable")
        mask = 0
        values = []
        for index, _key_value in zip(indexes, row):
            mask |= 1 << index
            values.append(_key_value[1])
        encoded_rows.append([format(mask, "x"), values])

    payload = {
        "kind": "metrics_jsonl_v1",
        "newline": newline,
        "json_style": "compact" if compact_style else "spaced",
        "fields": fields,
        "rows": encoded_rows,
        "trailing_empty_line": trailing_empty_line,
    }
    return compact_json_bytes(payload)


def restore_metric_jsonl_payload(encoded: bytes) -> bytes:
    payload = json.loads(encoded.decode("utf-8"))
    fields = payload["fields"]
    newline = payload["newline"]
    separators = (",", ":") if payload["json_style"] == "compact" else None
    output_lines = []
    for mask_hex, values in payload["rows"]:
        mask = int(mask_hex, 16)
        value_iter = iter(values)
        row = {}
        for index, field in enumerate(fields):
            if mask & (1 << index):
                row[field] = next(value_iter)
        output_lines.append(
            json.dumps(row, ensure_ascii=False, separators=separators)
        )
    text = newline.join(output_lines)
    if output_lines and newline:
        text += newline
    if payload.get("trailing_empty_line"):
        text += newline
    return text.encode("utf-8")


def encode_loss_csv_payload(data: bytes) -> bytes:
    text = decode_utf8(data)
    newline = detect_newline(data) or "\n"
    reader = csv.reader(io.StringIO(text), lineterminator=newline)
    rows = list(reader)
    if not rows:
        raise ArchiveError("loss csv is empty")
    header = rows[0]
    data_rows = rows[1:]
    width = len(header)
    for row in data_rows:
        if len(row) != width:
            raise ArchiveError("loss csv row width is not stable")
    columns = [[row[index] for row in data_rows] for index in range(width)]
    payload = {
        "kind": "loss_csv_v1",
        "newline": newline,
        "header": header,
        "columns": columns,
    }
    return compact_json_bytes(payload)


def restore_loss_csv_payload(encoded: bytes) -> bytes:
    payload = json.loads(encoded.decode("utf-8"))
    header = payload["header"]
    columns = payload["columns"]
    newline = payload["newline"]
    row_count = len(columns[0]) if columns else 0
    output = io.StringIO()
    writer = csv.writer(output, lineterminator=newline)
    writer.writerow(header)
    for row_index in range(row_count):
        writer.writerow([column[row_index] for column in columns])
    return output.getvalue().encode("utf-8")


def encode_log_payload(data: bytes) -> bytes:
    text = decode_utf8(data)
    records: list[list[object]] = []
    for line_bytes in split_line_bytes(data):
        line_text = line_bytes.decode("utf-8")
        body, newline = strip_line_ending(line_text)
        match = LOG_LINE_RE.match(body)
        if match:
            ts, level, node, rank, local_rank, pid, message = match.groups()
            records.append(
                ["L", ts, level, int(node), int(rank), int(local_rank), int(pid), message, newline]
            )
        else:
            records.append(["R", body, newline])
    if data and not records:
        raise ArchiveError("log payload has no records")
    payload = {
        "kind": "log_template_v1",
        "records": records,
    }
    return compact_json_bytes(payload)


def restore_log_payload(encoded: bytes) -> bytes:
    payload = json.loads(encoded.decode("utf-8"))
    parts = []
    for record in payload["records"]:
        if record[0] == "L":
            _kind, ts, level, node, rank, local_rank, pid, message, newline = record
            parts.append(
                f"{ts} {level} node={node} rank={rank} "
                f"local_rank={local_rank} pid={pid} {message}{newline}"
            )
        elif record[0] == "R":
            _kind, body, newline = record
            parts.append(f"{body}{newline}")
        else:
            raise ArchiveError(f"unknown log record kind: {record[0]}")
    return "".join(parts).encode("utf-8")


def restore_structured_payload(codec: str, encoded: bytes) -> bytes:
    if codec == "raw-bytes":
        return encoded
    if codec == "metrics-jsonl-v1":
        return restore_metric_jsonl_payload(encoded)
    if codec == "loss-csv-v1":
        return restore_loss_csv_payload(encoded)
    if codec == "log-template-v1":
        return restore_log_payload(encoded)
    raise ArchiveError(f"unsupported payload codec: {codec}")


def encode_structured_payload(path: Path, data: bytes, use_structured: bool) -> tuple[str, bytes]:
    if not use_structured:
        return "raw-bytes", data

    name = path.name
    candidates = []
    if fnmatch.fnmatch(name, "metrics*jsonl"):
        candidates.append(("metrics-jsonl-v1", encode_metric_jsonl_payload))
    elif fnmatch.fnmatch(name, "loss_to_log*txt"):
        candidates.append(("loss-csv-v1", encode_loss_csv_payload))
    elif fnmatch.fnmatch(name, "log*txt"):
        candidates.append(("log-template-v1", encode_log_payload))

    for codec, encoder in candidates:
        try:
            encoded = encoder(data)
            if restore_structured_payload(codec, encoded) == data:
                return codec, encoded
        except Exception:
            continue
    return "raw-bytes", data


def validate_generated_file_sizes(
    output_dir: Path, generated_names: Iterable[str], max_size: int
) -> None:
    for name in generated_names:
        path = output_dir / name
        size = path.stat().st_size
        if size > max_size:
            raise ArchiveError(
                f"generated file exceeds chunk limit: {path} is {size} bytes "
                f"(limit {max_size})"
            )


def split_pack(args: argparse.Namespace) -> int:
    root = args.input_dir.resolve()
    if not root.is_dir():
        raise ArchiveError(f"input directory does not exist: {root}")

    output_dir = args.output_dir.resolve()
    chunk_size = args.chunk_size
    ensure_output_dir(output_dir, args.overwrite)

    patterns = parse_patterns(args.patterns)
    files = sorted(iter_matching_files(root, patterns, args.recursive))
    if not files and not args.allow_empty:
        raise ArchiveError(
            f"no files matched {', '.join(patterns)} under {root}; "
            "pass --allow-empty if this is expected"
        )

    include_root_dir = not args.no_root_dir
    root_dir_name = root.name if include_root_dir else None
    preset = compression_preset(args)
    jobs = resolve_jobs(args.jobs, len(files))

    tasks = []
    for index, path in enumerate(files):
        rel = path.relative_to(root)
        tasks.append(
            (
                index,
                str(path),
                make_archive_name(root, rel, include_root_dir),
                portable_path(rel),
                str(output_dir),
                preset,
                chunk_size,
                args.overwrite,
                not args.raw_bytes,
            )
        )

    results: list[dict[str, object]] = []
    if jobs == 1:
        for task in tasks:
            result = compress_file_to_split_chunks(task)
            results.append(result)
            print(
                f"compressed {len(results)}/{len(files)} into "
                f"{len(result['chunks'])} chunk(s): {result['path']}"
            )
    else:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            futures = [executor.submit(compress_file_to_split_chunks, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                print(
                    f"compressed {len(results)}/{len(files)} into "
                    f"{len(result['chunks'])} chunk(s): {result['path']}"
                )

    results.sort(key=lambda item: int(item["index"]))
    total_input_bytes = sum(int(item["size"]) for item in results)
    compressed_payload_bytes = sum(int(item["compressed_size"]) for item in results)
    data_chunk_count = sum(len(item["chunks"]) for item in results)

    manifest = build_manifest(
        source_root=root,
        root_dir_name=root_dir_name,
        patterns=patterns,
        files=results,
        compression="split-per-file-xz",
        version=SPLIT_ARCHIVE_FORMAT_VERSION,
    )
    manifest["chunk_size"] = chunk_size
    manifest_bytes = compact_json_bytes(manifest)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest_parts = write_split_parts(
        output_dir=output_dir,
        prefix=SPLIT_MANIFEST_PART_PREFIX,
        suffix=".jsonpart",
        data=manifest_bytes,
        chunk_size=chunk_size,
        overwrite=args.overwrite,
    )

    index = {
        "format": "speciesllm-training-output-text-split-index",
        "version": SPLIT_ARCHIVE_FORMAT_VERSION,
        "chunk_size": chunk_size,
        "manifest_size": len(manifest_bytes),
        "manifest_sha256": manifest_sha256,
        "manifest_parts": manifest_parts,
    }
    index_bytes = compact_json_bytes(index)
    if len(index_bytes) > chunk_size:
        raise ArchiveError(
            f"split index would be {len(index_bytes)} bytes, above limit {chunk_size}"
        )
    index_path = output_dir / SPLIT_INDEX_NAME
    if index_path.exists() and not args.overwrite:
        raise ArchiveError(f"refusing to overwrite existing file: {index_path}")
    index_path.write_bytes(index_bytes)

    generated_names = [SPLIT_INDEX_NAME]
    generated_names.extend(str(part["name"]) for part in manifest_parts)
    for item in results:
        generated_names.extend(str(chunk["name"]) for chunk in item["chunks"])
    validate_generated_file_sizes(output_dir, generated_names, chunk_size)

    total_output_bytes = sum((output_dir / name).stat().st_size for name in generated_names)
    ratio = total_output_bytes / total_input_bytes if total_input_bytes else 0.0
    payload_ratio = (
        compressed_payload_bytes / total_input_bytes if total_input_bytes else 0.0
    )
    print(
        f"split-packed {len(files)} files with {jobs} worker(s), "
        f"{data_chunk_count} data chunk(s), {len(manifest_parts)} manifest chunk(s), "
        f"max chunk {chunk_size} bytes"
    )
    print(
        f"{total_input_bytes:,} bytes -> {total_output_bytes:,} bytes "
        f"({ratio:.2%}; compressed payload {payload_ratio:.2%})"
    )
    print(output_dir)
    return 0


def pack_parallel(args: argparse.Namespace) -> int:
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

    include_root_dir = not args.no_root_dir
    root_dir_name = root.name if include_root_dir else None
    preset = compression_preset(args)
    jobs = resolve_jobs(args.jobs, len(files))

    temp_parent = args.work_dir.resolve() if args.work_dir else output.parent
    temp_parent.mkdir(parents=True, exist_ok=True)
    tasks = []
    for index, path in enumerate(files):
        rel = path.relative_to(root)
        tasks.append(
            (
                index,
                str(path),
                make_archive_name(root, rel, include_root_dir),
                portable_path(rel),
                "",  # filled after the temporary directory is created
                preset,
            )
        )

    total_input_bytes = 0
    compressed_payload_bytes = 0
    results: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(
        prefix=".speciesllm_training_text_", dir=temp_parent
    ) as temp_dir:
        runnable_tasks = [
            (idx, src, arc, rel, temp_dir, task_preset)
            for idx, src, arc, rel, _, task_preset in tasks
        ]
        if jobs == 1:
            for task in runnable_tasks:
                result = compress_file_to_xz(task)
                results.append(result)
                print(f"compressed {len(results)}/{len(files)}: {result['path']}")
        else:
            with ProcessPoolExecutor(max_workers=jobs) as executor:
                futures = [executor.submit(compress_file_to_xz, task) for task in runnable_tasks]
                for future in as_completed(futures):
                    result = future.result()
                    results.append(result)
                    print(f"compressed {len(results)}/{len(files)}: {result['path']}")

        results.sort(key=lambda item: int(item["index"]))
        manifest_files = []
        for result in results:
            total_input_bytes += int(result["size"])
            compressed_payload_bytes += int(result["compressed_size"])
            manifest_item = dict(result)
            manifest_item.pop("tmp_compressed_path")
            manifest_files.append(manifest_item)

        manifest = build_manifest(
            source_root=root,
            root_dir_name=root_dir_name,
            patterns=patterns,
            files=manifest_files,
            compression="per-file-xz",
            version=PARALLEL_ARCHIVE_FORMAT_VERSION,
        )
        manifest_bytes = (
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")

        with output.open("wb") as raw_out:
            with tarfile.open(fileobj=raw_out, mode="w|", format=tarfile.PAX_FORMAT) as tar:
                manifest_info = tarfile.TarInfo(MANIFEST_NAME)
                manifest_info.size = len(manifest_bytes)
                manifest_info.mode = 0o644
                manifest_info.mtime = 0
                manifest_info.uid = 0
                manifest_info.gid = 0
                tar.addfile(manifest_info, io.BytesIO(manifest_bytes))

                for result in results:
                    tmp_compressed_path = Path(str(result["tmp_compressed_path"]))
                    tarinfo = tarfile.TarInfo(str(result["compressed_path"]))
                    tarinfo.size = tmp_compressed_path.stat().st_size
                    tarinfo.mode = 0o644
                    tarinfo.mtime = 0
                    tarinfo.uid = 0
                    tarinfo.gid = 0
                    with tmp_compressed_path.open("rb") as raw_in:
                        tar.addfile(tarinfo, raw_in)

    archive_bytes = output.stat().st_size
    ratio = archive_bytes / total_input_bytes if total_input_bytes else 0.0
    payload_ratio = (
        compressed_payload_bytes / total_input_bytes if total_input_bytes else 0.0
    )
    print(
        f"packed {len(files)} files with {jobs} worker(s), "
        f"{total_input_bytes:,} bytes -> {archive_bytes:,} bytes "
        f"({ratio:.2%}; compressed payload {payload_ratio:.2%})"
    )
    print(output)
    return 0


def pack_single_stream(args: argparse.Namespace) -> int:
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

    include_root_dir = not args.no_root_dir
    root_dir_name = root.name if include_root_dir else None
    manifest_files: list[dict[str, object]] = []
    total_input_bytes = 0
    preset = compression_preset(args)

    with output.open("wb") as raw_out:
        with lzma.LZMAFile(raw_out, "wb", preset=preset) as xz_out:
            with tarfile.open(fileobj=xz_out, mode="w|", format=tarfile.PAX_FORMAT) as tar:
                for path in files:
                    rel = path.relative_to(root)
                    arcname = make_archive_name(root, rel, include_root_dir)
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
                    compression="outer-tar-xz",
                    version=LEGACY_ARCHIVE_FORMAT_VERSION,
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


def is_xz_archive(path: Path) -> bool:
    with path.open("rb") as raw_in:
        return raw_in.read(len(XZ_MAGIC)) == XZ_MAGIC


def validate_manifest(
    manifest: dict[str, object],
    *,
    expected_version: int,
    expected_compression: str | None,
) -> list[dict[str, object]]:
    if manifest.get("format") != "speciesllm-training-output-text-archive":
        raise ArchiveError("archive manifest is malformed: unexpected format")
    if int(manifest.get("version", -1)) != expected_version:
        raise ArchiveError(f"unsupported archive version: {manifest.get('version')}")
    compression = manifest.get("compression")
    if expected_compression and compression not in (expected_compression, None):
        raise ArchiveError(f"unsupported archive compression: {compression}")

    expected_files = manifest.get("files")
    if not isinstance(expected_files, list):
        raise ArchiveError("archive manifest is malformed: files must be a list")
    if int(manifest.get("file_count", -1)) != len(expected_files):
        raise ArchiveError("archive manifest is malformed: file_count mismatch")

    typed_files: list[dict[str, object]] = []
    for item in expected_files:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ArchiveError("archive manifest is malformed: invalid file entry")
        typed_files.append(item)
    return typed_files


def verify_observed_files(
    manifest: dict[str, object], observed: dict[str, dict[str, object]]
) -> None:
    expected_files = validate_manifest(
        manifest,
        expected_version=int(manifest.get("version", -1)),
        expected_compression=None,
    )
    expected_by_path = {str(item["path"]): item for item in expected_files}

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
    if is_xz_archive(archive_path):
        return load_legacy_archive(
            archive_path,
            output_dir=output_dir,
            overwrite=overwrite,
            list_only=list_only,
            verify_only=verify_only,
        )
    return load_parallel_archive(
        archive_path,
        output_dir=output_dir,
        overwrite=overwrite,
        list_only=list_only,
        verify_only=verify_only,
    )


def load_legacy_archive(
    archive_path: Path,
    *,
    output_dir: Path | None,
    overwrite: bool,
    list_only: bool,
    verify_only: bool,
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

    validate_manifest(
        manifest,
        expected_version=LEGACY_ARCHIVE_FORMAT_VERSION,
        expected_compression="outer-tar-xz",
    )
    verify_observed_files(manifest, observed)
    return manifest


def decompress_parallel_member(
    member_fileobj: BinaryIO,
    *,
    output_path: Path | None,
    overwrite: bool,
    mode: int,
    mtime: int,
) -> tuple[int, str, str]:
    if output_path is not None and output_path.exists() and not overwrite:
        raise ArchiveError(f"refusing to overwrite existing file: {output_path}")
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    compressed_reader = HashingReader(member_fileobj)
    source_digest = hashlib.sha256()
    source_size = 0
    out = output_path.open("wb") if output_path is not None else None
    try:
        with lzma.LZMAFile(compressed_reader, "rb") as xz_in:
            while True:
                chunk = xz_in.read(COPY_BUFFER_SIZE)
                if not chunk:
                    break
                source_digest.update(chunk)
                source_size += len(chunk)
                if out is not None:
                    out.write(chunk)
    finally:
        if out is not None:
            out.close()

    if output_path is not None:
        os.chmod(output_path, mode & 0o777)
        os.utime(output_path, (mtime, mtime))
    return source_size, source_digest.hexdigest(), compressed_reader.hexdigest


def load_parallel_archive(
    archive_path: Path,
    *,
    output_dir: Path | None,
    overwrite: bool,
    list_only: bool,
    verify_only: bool,
) -> dict[str, object]:
    output_root = output_dir.resolve() if output_dir else None
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] | None = None
    expected_by_compressed_path: dict[str, dict[str, object]] = {}
    observed: dict[str, dict[str, object]] = {}

    with archive_path.open("rb") as raw_in:
        with tarfile.open(fileobj=raw_in, mode="r|") as tar:
            for member in tar:
                if member.name == MANIFEST_NAME:
                    manifest = json.loads(read_member_bytes(tar, member).decode("utf-8"))
                    expected_files = validate_manifest(
                        manifest,
                        expected_version=PARALLEL_ARCHIVE_FORMAT_VERSION,
                        expected_compression="per-file-xz",
                    )
                    for item in expected_files:
                        compressed_path = item.get("compressed_path")
                        if not isinstance(compressed_path, str):
                            raise ArchiveError(
                                "archive manifest is malformed: missing compressed_path"
                            )
                        expected_by_compressed_path[compressed_path] = item
                    continue

                if manifest is None:
                    raise ArchiveError(
                        f"parallel archive manifest must be first; saw {member.name}"
                    )
                if not member.isfile():
                    raise ArchiveError(f"unsupported archive member type: {member.name}")
                expected = expected_by_compressed_path.get(member.name)
                if expected is None:
                    raise ArchiveError(f"unexpected archive member: {member.name}")
                if int(member.size) != int(expected["compressed_size"]):
                    raise ArchiveError(f"compressed size mismatch for {expected['path']}")

                path = str(expected["path"])
                if list_only:
                    observed[path] = {
                        "size": int(expected["size"]),
                        "sha256": expected.get("sha256"),
                    }
                    continue

                fileobj = tar.extractfile(member)
                if fileobj is None:
                    raise ArchiveError(f"cannot read archive member: {member.name}")

                output_path = None
                if not verify_only:
                    if output_root is None:
                        raise ArchiveError("output_dir is required unless verify is set")
                    output_path = safe_output_path(output_root, path)

                source_size, source_sha, compressed_sha = decompress_parallel_member(
                    fileobj,
                    output_path=output_path,
                    overwrite=overwrite,
                    mode=int(expected["mode"]),
                    mtime=int(expected["mtime"]),
                )

                if source_size != int(expected["size"]):
                    raise ArchiveError(f"size mismatch for {path}")
                if source_sha != expected.get("sha256"):
                    raise ArchiveError(f"sha256 mismatch for {path}")
                expected_compressed_sha = expected.get("compressed_sha256")
                if expected_compressed_sha and compressed_sha != expected_compressed_sha:
                    raise ArchiveError(f"compressed sha256 mismatch for {path}")

                observed[path] = {
                    "size": source_size,
                    "sha256": source_sha,
                    "output_path": str(output_path) if output_path else None,
                }

    if manifest is None:
        raise ArchiveError(f"archive is missing {MANIFEST_NAME}")

    verify_observed_files(manifest, observed)
    return manifest


def read_split_manifest(split_dir: Path) -> dict[str, object]:
    index_path = split_dir / SPLIT_INDEX_NAME
    if not index_path.is_file():
        raise ArchiveError(f"missing split index: {index_path}")
    index = json.loads(index_path.read_bytes().decode("utf-8"))
    if index.get("format") != "speciesllm-training-output-text-split-index":
        raise ArchiveError("split index is malformed: unexpected format")
    if int(index.get("version", -1)) != SPLIT_ARCHIVE_FORMAT_VERSION:
        raise ArchiveError(f"unsupported split index version: {index.get('version')}")

    parts = index.get("manifest_parts")
    if not isinstance(parts, list):
        raise ArchiveError("split index is malformed: manifest_parts must be a list")
    manifest_bytes_parts: list[bytes] = []
    for part in parts:
        if not isinstance(part, dict) or not isinstance(part.get("name"), str):
            raise ArchiveError("split index is malformed: invalid manifest part")
        path = split_dir / str(part["name"])
        if not path.is_file():
            raise ArchiveError(f"missing manifest part: {path}")
        data = path.read_bytes()
        if len(data) != int(part["size"]):
            raise ArchiveError(f"manifest part size mismatch: {path}")
        manifest_bytes_parts.append(data)

    manifest_bytes = b"".join(manifest_bytes_parts)
    if len(manifest_bytes) != int(index.get("manifest_size", -1)):
        raise ArchiveError("manifest size mismatch")
    if hashlib.sha256(manifest_bytes).hexdigest() != index.get("manifest_sha256"):
        raise ArchiveError("manifest sha256 mismatch")

    manifest = json.loads(manifest_bytes.decode("utf-8"))
    validate_manifest(
        manifest,
        expected_version=SPLIT_ARCHIVE_FORMAT_VERSION,
        expected_compression="split-per-file-xz",
    )
    return manifest


def decompress_chunk_sequence(
    split_dir: Path,
    chunks: list[dict[str, object]],
) -> tuple[bytes, str]:
    chunk_reader = ChunkSequenceReader(split_dir, chunks)
    compressed_reader = HashingReader(chunk_reader)
    output = io.BytesIO()
    try:
        with lzma.LZMAFile(compressed_reader, "rb") as xz_in:
            while True:
                chunk = xz_in.read(COPY_BUFFER_SIZE)
                if not chunk:
                    break
                output.write(chunk)
    finally:
        chunk_reader.close()
    return output.getvalue(), compressed_reader.hexdigest


def load_split_archive(
    split_dir: Path,
    *,
    output_dir: Path | None,
    overwrite: bool = False,
    verify_only: bool = False,
) -> dict[str, object]:
    if not split_dir.is_dir():
        raise ArchiveError(f"split directory does not exist: {split_dir}")

    output_root = output_dir.resolve() if output_dir else None
    if output_root is not None:
        output_root.mkdir(parents=True, exist_ok=True)

    manifest = read_split_manifest(split_dir)
    expected_files = validate_manifest(
        manifest,
        expected_version=SPLIT_ARCHIVE_FORMAT_VERSION,
        expected_compression="split-per-file-xz",
    )
    observed: dict[str, dict[str, object]] = {}

    for expected in expected_files:
        path = str(expected["path"])
        chunks = expected.get("chunks")
        if not isinstance(chunks, list):
            raise ArchiveError(f"split manifest is malformed: missing chunks for {path}")
        typed_chunks: list[dict[str, object]] = []
        for chunk in chunks:
            if not isinstance(chunk, dict) or not isinstance(chunk.get("name"), str):
                raise ArchiveError(f"split manifest is malformed: invalid chunk for {path}")
            typed_chunks.append(chunk)

        encoded_payload, compressed_sha = decompress_chunk_sequence(split_dir, typed_chunks)
        if len(encoded_payload) != int(expected.get("encoded_size", len(encoded_payload))):
            raise ArchiveError(f"encoded size mismatch for {path}")
        expected_encoded_sha = expected.get("encoded_sha256")
        if expected_encoded_sha and hashlib.sha256(encoded_payload).hexdigest() != expected_encoded_sha:
            raise ArchiveError(f"encoded sha256 mismatch for {path}")

        codec = str(expected.get("codec", "raw-bytes"))
        source_payload = restore_structured_payload(codec, encoded_payload)
        source_size = len(source_payload)
        source_sha = hashlib.sha256(source_payload).hexdigest()

        if source_size != int(expected["size"]):
            raise ArchiveError(f"size mismatch for {path}")
        if source_sha != expected.get("sha256"):
            raise ArchiveError(f"sha256 mismatch for {path}")
        if compressed_sha != expected.get("compressed_sha256"):
            raise ArchiveError(f"compressed sha256 mismatch for {path}")

        output_path = None
        if not verify_only:
            if output_root is None:
                raise ArchiveError("output_dir is required unless verify is set")
            output_path = safe_output_path(output_root, path)
            if output_path.exists() and not overwrite:
                raise ArchiveError(f"refusing to overwrite existing file: {output_path}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(source_payload)
            os.chmod(output_path, int(expected["mode"]) & 0o777)
            os.utime(output_path, (int(expected["mtime"]), int(expected["mtime"])))

        observed[path] = {
            "size": source_size,
            "sha256": source_sha,
            "output_path": str(output_path) if output_path else None,
        }

    verify_observed_files(manifest, observed)
    return manifest


def split_unpack(args: argparse.Namespace) -> int:
    manifest = load_split_archive(
        args.split_dir.resolve(),
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )
    print(f"split-unpacked and verified {manifest['file_count']} files")
    print(args.output_dir.resolve())
    return 0


def split_verify(args: argparse.Namespace) -> int:
    manifest = load_split_archive(
        args.split_dir.resolve(),
        output_dir=None,
        overwrite=False,
        verify_only=True,
    )
    print(f"verified split archive with {manifest['file_count']} files")
    return 0


def split_list(args: argparse.Namespace) -> int:
    manifest = read_split_manifest(args.split_dir.resolve())
    files = manifest["files"]
    assert isinstance(files, list)
    total_size = sum(int(item["size"]) for item in files if isinstance(item, dict))
    total_chunks = sum(len(item.get("chunks", [])) for item in files if isinstance(item, dict))
    print(
        f"{manifest['file_count']} files, {total_chunks} data chunks, "
        f"{total_size:,} uncompressed bytes"
    )
    for item in files:
        assert isinstance(item, dict)
        print(
            f"{int(item['size']):>12}  "
            f"{len(item.get('chunks', [])):>5} chunks  {item['path']}"
        )
    return 0


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
    parser.add_argument("archive", type=Path, help="Path to the archive.")


def add_split_dir_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("split_dir", type=Path, help="Path to the split archive directory.")


def add_common_pack_options(pack_parser: argparse.ArgumentParser) -> None:
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
        help="Overwrite existing outputs.",
    )
    pack_parser.add_argument(
        "--jobs",
        type=int,
        help=(
            "Parallel per-file compression worker count. Default: "
            "min(24, CPU count, matched file count)."
        ),
    )
    pack_parser.set_defaults(recursive=True, extreme=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Losslessly compress/decompress SpeciesLLM training_output text files "
            "(log*txt, loss_to_log*txt, metrics*jsonl, run_record.json)."
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
        help="Output archive path, for example /tmp/training_output_text.tar.",
    )
    add_common_pack_options(pack_parser)
    pack_parser.add_argument(
        "--work-dir",
        type=Path,
        help="Directory for temporary per-file compressed chunks. Default: output dir.",
    )
    pack_parser.add_argument(
        "--single-stream",
        action="store_true",
        help=(
            "Use the old single .tar.xz stream. This can be slightly smaller "
            "but is serial and much slower on many large files."
        ),
    )
    pack_parser.set_defaults(func=pack)

    split_pack_parser = subparsers.add_parser(
        "split-pack",
        help="Create a directory of small compressed chunk files.",
    )
    split_pack_parser.add_argument(
        "input_dir", type=Path, help="training_output directory."
    )
    split_pack_parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for small split files.",
    )
    split_pack_parser.add_argument(
        "--chunk-size",
        type=parse_size,
        default=DEFAULT_SPLIT_CHUNK_SIZE,
        help=(
            "Maximum size of each generated file. Default: 80KB "
            f"({DEFAULT_SPLIT_CHUNK_SIZE} bytes)."
        ),
    )
    split_pack_parser.add_argument(
        "--raw-bytes",
        action="store_true",
        help="Disable structure-aware encoding and compress raw file bytes.",
    )
    add_common_pack_options(split_pack_parser)
    split_pack_parser.set_defaults(func=split_pack)

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

    split_unpack_parser = subparsers.add_parser(
        "split-unpack", help="Merge/decompress a split archive directory."
    )
    add_split_dir_arg(split_unpack_parser)
    split_unpack_parser.add_argument(
        "-C",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where files should be restored.",
    )
    split_unpack_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing restored files.",
    )
    split_unpack_parser.set_defaults(func=split_unpack)

    split_verify_parser = subparsers.add_parser(
        "split-verify", help="Verify a split archive directory without extracting."
    )
    add_split_dir_arg(split_verify_parser)
    split_verify_parser.set_defaults(func=split_verify)

    split_list_parser = subparsers.add_parser(
        "split-list", help="List files in a split archive directory."
    )
    add_split_dir_arg(split_list_parser)
    split_list_parser.set_defaults(func=split_list)

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
