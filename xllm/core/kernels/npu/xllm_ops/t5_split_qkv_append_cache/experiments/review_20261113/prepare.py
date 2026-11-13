# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Materialize pinned, isolated candidate source trees; never edits the product."""

import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any

HERE = Path(__file__).resolve().parent


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args])


def _prepare(repo: Path, output: Path) -> dict[str, Any]:
    config = json.loads((HERE / "candidates.json").read_text())
    # Avoid recursive copies, untracked product trees and accidental overwrites.
    if output.is_relative_to(repo) or repo.is_relative_to(output):
        raise ValueError("Choose an output directory outside the source repository")
    commit = config["source_commit"]
    prefix = config["operator_path"] + "/"
    files = _git(repo, "ls-tree", "-rz", "--name-only", commit, "--", prefix).decode().split("\0")[:-1]
    files = [p for p in files if not p.startswith(
        (prefix + "docs/", prefix + "development_checkpoints/"))]
    original = {p.removeprefix(prefix): _git(repo, "show", f"{commit}:{p}") for p in files}
    header = config["header"]
    if _sha(original[header]) != config["header_sha256"]:
        raise ValueError("Pinned header identity differs from the reviewed control")
    needle = "constexpr uint64_t kReadBalanceBudgetBytes = uint64_t{190} * 1024 * 1024;"
    if original[header].decode().count(needle) != 1:
        raise ValueError("Expected exactly one cache-admission budget declaration")
    official_prefix = config["official_path"] + "/"
    official = _git(repo, "ls-tree", "-rz", "--name-only", commit, "--", official_prefix).decode().split("\0")[:-1]
    if len(official) != 12:
        raise ValueError("Pinned original task package must contain 12 files")
    output.mkdir(parents=True, exist_ok=False)
    official_hashes = {}
    for path in official:
        relative = path.removeprefix(official_prefix)
        data = _git(repo, "show", f"{commit}:{path}")
        target = output / "official_task_package" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        official_hashes[relative] = _sha(data)
    entries = []
    for candidate in config["candidates"]:
        name = candidate["name"]
        project = output / name / "project"
        project.mkdir(parents=True)
        changed = dict(original)
        if not candidate.get("patch"):
            replacement = needle.replace("{190}", "{" + str(candidate["budget_mib"]) + "}")
            changed[header] = original[header].decode().replace(needle, replacement).encode()
        for path, data in changed.items():
            target = project / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        if candidate.get("patch"):
            source_patch = (HERE / candidate["patch"]).resolve()
            if not source_patch.is_relative_to(HERE):
                raise ValueError("Candidate patch must stay inside the experiment directory")
            # Apply trusted experiment patches outside any Git worktree, with an
            # explicit isolated output prefix. Never changes the repository index.
            with tempfile.TemporaryDirectory(prefix="t5qkv-apply-") as temporary:
                subprocess.run(["git", "apply", "--unsafe-paths", "--directory=" + str(project),
                                str(source_patch)], cwd=temporary, check=True)
            paths = {str(p.relative_to(project)) for p in project.rglob("*") if p.is_file()}
            if paths != set(original):
                raise ValueError("Patch added or removed files outside the pinned source inventory")
            changed = {p: (project / p).read_bytes() for p in original}
            allowed = set(candidate.get("allowed_patch_files", [header]))
            if {p for p in original if changed[p] != original[p]} - allowed:
                raise ValueError("Patch changed files outside its explicit trusted inventory")
        patch = "".join(line for path in sorted(original) for line in difflib.unified_diff(
            original[path].decode().splitlines(keepends=True),
            changed[path].decode().splitlines(keepends=True),
            fromfile="a/" + path, tofile="b/" + path))
        (project.parent / "candidate.patch").write_text(patch)
        entry = {**candidate, "source_files": {p: _sha(b) for p, b in changed.items()},
                 "patch_sha256": _sha(patch.encode()), "status": "source_prepared_a3_unvalidated"}
        (project.parent / "source_manifest.json").write_text(json.dumps(entry, indent=2) + "\n")
        entries.append(entry)
    result = {"schema_version": 1, "source_commit": commit, "npu_executed": False,
              "official_files": official_hashes, "candidates": entries,
              "scope": "source_preparation_only", "status": "prepared_not_validated"}
    (output / "prepared_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(subprocess.check_output(
        ["git", "-C", str(HERE), "rev-parse", "--show-toplevel"], text=True).strip()))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = _prepare(args.repo.resolve(), args.output_root.resolve())
    print(json.dumps({"output_root": str(args.output_root.resolve()),
                      "status": result["status"], "npu_executed": False}))


if __name__ == "__main__":
    _main()
