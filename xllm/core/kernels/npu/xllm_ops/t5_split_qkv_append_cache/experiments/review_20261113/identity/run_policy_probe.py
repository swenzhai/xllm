# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Isolated allocation API policy trace; no operator or acceptance timings."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def _main() -> int:
    files = [HERE / n for n in ["trace_alloc_policy.cpp", "trace_alloc_policy.so", "probe_pages.py", "run_policy_probe.py"]]
    files += [Path("/usr/local/Ascend/cann-9.1.0/include/acl/acl_rt.h")]
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    report = HERE / "reports_01"
    report.mkdir(exist_ok=False)
    result = {"scope": "isolated_API_interposition_NO_OPERATOR_NOT_ACCEPTANCE", "status": "running", "records": [], "before": before}
    (report / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    code = '''import runpy,sys
if int(sys.argv[1]):
 import torch,torch_npu
 torch.npu.set_device(0)
 temporary=torch.empty(int(sys.argv[1]),dtype=torch.uint8,device="npu:0")
 del temporary
 torch.npu.synchronize()
sys.argv=[sys.argv[2],"--report",sys.argv[3]]
runpy.run_path(sys.argv[0],run_name="__main__")
'''
    for device in ["0", "1"]:
        for allocator, primer in [("default", 0), ("default", 1073741312), ("page1g", 0)]:
            directory = report / f"chip{device}_{allocator}_{primer}"
            directory.mkdir()
            env = dict(os.environ)
            for key in ["PYTORCH_NPU_ALLOC_CONF", "TASK_QUEUE_ENABLE", "ASCEND_LAUNCH_BLOCKING"]:
                env.pop(key, None)
            env.update(ASCEND_RT_VISIBLE_DEVICES=device, LD_PRELOAD=str(HERE / "trace_alloc_policy.so"),
                       T5QKV_ALLOC_TRACE=str(directory / "allocation_calls.jsonl"))
            if allocator == "page1g":
                env["PYTORCH_NPU_ALLOC_CONF"] = "page_size:1g"
            command = [sys.executable, "-c", code, str(primer), str(HERE / "probe_pages.py"), str(directory / "pages.json")]
            execution = {"command": command, "environment": {k: env.get(k) for k in ["ASCEND_RT_VISIBLE_DEVICES", "PYTORCH_NPU_ALLOC_CONF", "LD_PRELOAD", "T5QKV_ALLOC_TRACE"]}}
            with (directory / "probe.log").open("xb") as log:
                rc = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            execution["exit_code"] = rc
            (directory / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")
            if rc != 0:
                raise RuntimeError("Interposed allocation probe failed; inspect retained logs")
            calls = [json.loads(line) for line in (directory / "allocation_calls.jsonl").read_text().splitlines()]
            pages = json.loads((directory / "pages.json").read_text())
            if not calls or len(pages["records"]) != 6 or any(r["api_return_code"] != 0 for r in pages["records"]):
                raise RuntimeError("Incomplete interception or allocation-attribute records")
            result["records"].append({"visible_device": device, "allocator": allocator, "primer_bytes": primer,
                "directory": str(directory), "calls": calls, "allocator_segments": pages["allocator_snapshot"]})
            (report / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    result["after"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    result["source_and_interposer_unchanged"] = result["before"] == result["after"]
    result["status"] = "completed_diagnostic_NOT_ACCEPTANCE" if result["source_and_interposer_unchanged"] else "invalid"
    (report / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0 if result["status"] == "completed_diagnostic_NOT_ACCEPTANCE" else 1


if __name__ == "__main__":
    raise SystemExit(_main())
