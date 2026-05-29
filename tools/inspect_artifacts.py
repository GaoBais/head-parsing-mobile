"""Inspect exported model artifacts before device benchmark runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


KNOWN_MODEL_SUFFIXES = {
    ".onnx": "onnx",
    ".tflite": "tflite",
    ".mlpackage": "coreml",
    ".mlmodel": "coreml",
    ".pt": "pytorch",
    ".pth": "pytorch",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def sha256_directory(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted((p for p in path.rglob("*") if p.is_file()), key=lambda p: p.relative_to(path).as_posix()):
        rel = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(rel)
        digest.update(b"\0")
        with child.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def inspect_artifact(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Artifact does not exist: {path}")

    is_dir = path.is_dir()
    size_bytes = directory_size(path) if is_dir else path.stat().st_size
    suffix = path.suffix.lower()
    artifact_type = KNOWN_MODEL_SUFFIXES.get(suffix, "unknown")
    sha256 = sha256_directory(path) if is_dir else sha256_file(path)

    return {
        "path": str(path),
        "name": path.name,
        "type": artifact_type,
        "is_directory": is_dir,
        "size_bytes": size_bytes,
        "size_mb": round(size_bytes / (1024 * 1024), 3),
        "sha256": sha256,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect exported model artifacts.")
    parser.add_argument("artifacts", nargs="+", type=Path, help="Model files or directories.")
    parser.add_argument("--output", type=Path, default=Path("outputs/export/artifact_manifest.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = {
        "artifacts": [inspect_artifact(path) for path in args.artifacts],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
