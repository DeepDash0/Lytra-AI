#!/usr/bin/env python3
"""Create and verify deterministic, content-hashed dataset file manifests.

Manifest creation and verification are local; raw file bytes are never exported.
This does not certify a dataset, nor make a manifest tamper-proof.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable


def sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def safe_path(root: Path, relative: str) -> tuple[Path, str]:
    if (not isinstance(relative, str) or not relative or Path(relative).is_absolute()
            or ".." in Path(relative).parts):
        raise ValueError("all paths must be nonempty and relative to dataset root")
    base = root.resolve(strict=True)
    p = (base / relative).resolve(strict=True)
    try:
        canonical_relative = p.relative_to(base).as_posix()
    except ValueError as e:
        raise ValueError("data path escapes the dataset root") from e
    if not p.is_file() or p.is_symlink():
        raise ValueError("data path must be a regular file")
    return p, canonical_relative


def entry_list(root: Path, relatives: Iterable[str]) -> list[dict]:
    names: set[str] = set()
    entries: list[dict] = []
    for relative in relatives:
        path, name = safe_path(root, relative)
        if name in names:
            raise ValueError(f"duplicate file path: {name}")
        names.add(name)
        digest, byte_count = sha256_file(path)
        entries.append({"path": name, "size_bytes": byte_count, "sha256": digest})
    if not entries:
        raise ValueError("manifest must contain at least one file")
    return sorted(entries, key=lambda value: value["path"])


def manifest_digest(entries: list[dict]) -> str:
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def create_manifest(root: Path, relatives: Iterable[str]) -> dict:
    entries = entry_list(root, relatives)
    return {
        "schema_version": 1,
        "algorithm": "sha256",
        "file_count": len(entries),
        "files": entries,
        "entries_sha256": manifest_digest(entries),
    }


def verify_manifest(root: Path, manifest: dict) -> dict:
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or manifest.get("algorithm") != "sha256":
        return {"status": "FAIL", "errors": ["unsupported manifest schema"]}
    files = manifest.get("files")
    if not isinstance(files, list) or not files or any(
        not isinstance(row, dict)
        or set(row) != {"path", "sha256", "size_bytes"}
        or not isinstance(row["path"], str)
        or not isinstance(row["sha256"], str)
        or len(row["sha256"]) != 64
        or any(c not in "0123456789abcdef" for c in row["sha256"])
        or type(row["size_bytes"]) is not int or row["size_bytes"] < 0
        for row in files
    ):
        return {"status": "FAIL", "errors": ["malformed manifest file entries"]}
    expected_names = [row["path"] for row in files]
    if expected_names != sorted(set(expected_names)):
        return {"status": "FAIL", "errors": ["manifest paths must be unique and sorted"]}
    if manifest.get("file_count") != len(files) or manifest.get("entries_sha256") != manifest_digest(files):
        return {"status": "FAIL", "errors": ["manifest metadata or entries hash mismatch"]}

    errors: list[str] = []
    for row in files:
        try:
            path, name = safe_path(root, row["path"])
            if name != row["path"]:
                raise ValueError("noncanonical path")
            digest, size = sha256_file(path)
            if digest != row["sha256"] or size != row["size_bytes"]:
                errors.append(f"content mismatch for {name}")
        except (OSError, ValueError) as exc:
            errors.append(f"cannot validate {row['path']}: {type(exc).__name__}")
    return {"status": "PASS" if not errors else "FAIL", "file_count": len(files),
            "entries_sha256": manifest["entries_sha256"], "errors": errors}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    create = sub.add_parser("create")
    create.add_argument("--root", type=Path, required=True)
    create.add_argument("--file", action="append", required=True, help="Relative path; may be repeated")
    create.add_argument("--output", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        if args.action == "create":
            result = create_manifest(args.root, args.file)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(json.dumps({"status": "CREATED", "file_count": result["file_count"],
                              "entries_sha256": result["entries_sha256"]}))
            return 0
        result = verify_manifest(args.root, json.loads(args.manifest.read_text(encoding="utf-8")))
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "PASS" else 2
    except (OSError, ValueError, TypeError) as exc:
        print(f"manifest error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
