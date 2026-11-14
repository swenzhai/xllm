# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Compare retained old/new ELF sections without loading or executing an NPU."""

import argparse
import hashlib
import json
from pathlib import Path
import struct
from typing import Any


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sections(data: bytes) -> dict[str, bytes]:
    if data[:6] != b"\x7fELF\x02\x01":
        raise ValueError("Expected little-endian ELF64")
    offset = struct.unpack_from("<Q", data, 40)[0]
    size, count, names_index = struct.unpack_from("<HHH", data, 58)
    if size != 64 or not 0 < names_index < count or offset + size * count > len(data):
        raise ValueError("Invalid or unsupported section table")
    entries = [struct.unpack_from("<IIQQQQIIQQ", data, offset + size * i) for i in range(count)]
    names = entries[names_index]
    strings = data[names[4]:names[4] + names[5]]
    result = {}
    for item in entries[1:]:
        name = strings[item[0]:].split(b"\0", 1)[0].decode()
        if item[1] != 8:  # SHT_NOBITS has no bytes in the file.
            if item[4] + item[5] > len(data):
                raise ValueError("Section exceeds ELF file")
            result[name] = data[item[4]:item[4] + item[5]]
    return result


def _elf(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    sections = _sections(data)
    result: dict[str, Any] = {"path": str(path), "sha256": _sha(data), "bytes": len(data),
        "sections": {n: {"bytes": len(b), "sha256": _sha(b)} for n, b in sections.items()}}
    nested = {n: _sections(b) for n, b in sections.items()
              if n.startswith(".ascend.kernel.") and b.startswith(b"\x7fELF")}
    result["embedded_elf"] = {n: {k: {"bytes": len(v), "sha256": _sha(v)}
                                    for k, v in values.items()} for n, values in nested.items()}
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve prior evidence; choose a new output")
    config = json.loads(args.config.read_text())
    expected = dict(config["old"]["expected_artifacts"])
    new = config["new"]
    directory = Path(new["project"]).parent
    manifest = json.loads(Path(new["build_manifest"]).read_text())
    expected.update({str(directory / p): h for p, h in manifest["artifacts"].items()})
    observed = {p: _sha(Path(p).read_bytes()) for p in expected}
    if observed != expected:
        raise ValueError("Retained artifact identity differs from pinned evidence")
    source = json.loads(Path(new["source_manifest"]).read_text())["source_files"]
    sources = {name: {p: _sha((Path(config[name]["project"]) / p).read_bytes()) for p in source}
               for name in ["old", "new"]}
    if sources["new"] != source:
        raise ValueError("New source identity changed")
    records = {}
    for name in ["old", "new"]:
        project, prefix = Path(config[name]["project"]), Path(config[name]["prefix"])
        files = {"extension": next(project.glob("_t5qkv*.so")),
                 "direct": project / "build_direct/libt5qkv_direct.so",
                 "opapi": prefix / "op_api/lib/libcust_opapi.so"}
        files.update({"sdk_" + p.name: p for p in prefix.rglob("*.o")})
        records[name] = {k: _elf(v) for k, v in files.items()}
    comparisons = {}
    for key in records["old"]:
        first, second = records["old"][key], records["new"][key]
        comparisons[key] = {"whole_file_equal": first["sha256"] == second["sha256"],
            "text_equal": first["sections"].get(".text") == second["sections"].get(".text"),
            "embedded_text": {n: {"present_both": n in second["embedded_elf"],
                "equal": sec.get(".text") == second["embedded_elf"].get(n, {}).get(".text")}
                for n, sec in first["embedded_elf"].items()}}
    output = {"scope": "elf_identity_not_runtime_or_performance", "npu_executed": False,
        "pinned_artifacts_verified": True, "sources": sources,
        "source_differences": [p for p in source if sources["old"][p] != sources["new"][p]],
        "artifacts": records, "comparisons": comparisons}
    args.output.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    _main()
