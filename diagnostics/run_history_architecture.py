"""Run exactly four new history x architecture cells; reuse audited BASE artifacts."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from data import file_hash
from diagnostics.run_intermediate_supervision import write
from diagnostics.summarize_history_architecture import (all_specs, audit_cell, summarize,
                                                        verify_common_initialization_and_streams)
from diagnostics.temporal_dependence import read, require


CONFIG = Path("configs/history_architecture_v1.json")
STATUS = Path("outputs/history_architecture_v1_queue.json")


def main():
    torch.set_num_threads(1)
    require(torch.cuda.is_available(), "CUDA required")
    c = read(CONFIG)
    expected = [("loop", "H0"), ("untied", "H0"), ("loop", "H24"), ("untied", "H24")]
    require([(s["variant"], s["group"]) for s in c["new_runs"]] == expected, "Wrong bounded queue")
    lock = Path("outputs/history_architecture_v1_queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(not STATUS.exists(), "Queue exists; inspect before resuming")
    protected = {**c["protected_artifacts"], **c["training_source_sha256"], **c["implementation_sha256"],
                 c["data"]: c["data_sha256"], c["reference_config"]: c["reference_config_sha256"],
                 str(CONFIG): file_hash(CONFIG)}
    def verify():
        require(all(file_hash(p) == h for p, h in protected.items()), "Frozen source or artifact changed")
    verify()
    for spec in c["new_runs"]:
        require(len(spec["segments"]) == 1, "Initial queue starts from scratch")
        path = Path(spec["segments"][0])
        require(not path.exists() and not path.with_suffix(".lineage.json").exists()
                and not path.with_suffix(".log").exists(), "Output already exists")
    require(not Path(c["report_output"]).exists(), "Report already exists")
    specs = all_specs(c)
    references = {s["group"]: audit_cell(s, c) for s in specs if s["variant"] == "base"}
    status = {"status": "running", "pid": os.getpid(), "started_unix": time.time(),
              "completed": [], "active": None, "plan_sha256": file_hash(CONFIG),
              "reused_base": ["H0", "H24"]}
    write(STATUS, status)
    completed = {}
    try:
        for spec in c["new_runs"]:
            verify()
            variant, group = spec["variant"], spec["group"]
            output = Path(spec["segments"][0])
            command = [sys.executable, "train_real.py", "--data", c["data"],
                       "--variant", variant, "--history-group", group, "--attention-scope", "full",
                       "--training-protocol-version", "2", "--ema-start-update", "1000",
                       "--supervision-mode", "final-only", "--loss-weights", "0", "1",
                       "--device", "cuda", "--updates", "3000", "--microbatch-size", "4",
                       "--accumulation-steps", "8", "--validate-every", "250", "--scenarios", "100",
                       "--sampling-steps", "50", "--chunk-size", "16", "--seed", "42", "--output", str(output)]
            lineage = {"variant": variant, "history_group": group, "seed": 42, "command": command,
                       "started_unix": time.time(), "exit_code": None,
                       "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                       "git_status": subprocess.check_output(["git", "status", "--porcelain"], text=True),
                       "training_source_sha256": c["training_source_sha256"], "data_sha256": c["data_sha256"],
                       "plan_sha256": file_hash(CONFIG)}
            write(output.with_suffix(".lineage.json"), lineage)
            with output.with_suffix(".log").open("x") as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                          env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
                status["active"] = {"variant": variant, "history_group": group, "pid": child.pid, "output": str(output)}
                write(STATUS, status)
                code = child.wait()
            lineage.update(exit_code=code, ended_unix=time.time(),
                           sources_unchanged=all(file_hash(p) == h for p, h in protected.items()))
            if code == 0:
                lineage["last_checkpoint_sha256"] = file_hash(output / "checkpoint_last.pt")
            write(output.with_suffix(".lineage.json"), lineage)
            require(code == 0 and lineage["sources_unchanged"], "Run failed; preserve artifacts and inspect")
            item = audit_cell(spec, c)
            verify_common_initialization_and_streams(references[group], item)
            if group == "H24":
                require(item["initialization"] == completed[variant, "H0"]["initialization"], "Same-variant initial tensors differ")
            completed[variant, group] = item
            lineage["engineering_audit_passed"] = True
            write(output.with_suffix(".lineage.json"), lineage)
            status["completed"].append({"variant": variant, "history_group": group})
            status["active"] = None
            write(STATUS, status)
        verify()
        status.update(status="training_complete", training_ended_unix=time.time())
        write(STATUS, status)
        report = summarize(c, Path(c["report_output"]))
        status.update(status="report_complete_pending_review", ended_unix=time.time(),
                      report_output=c["report_output"], report_rows=len(report["rows"]))
        write(STATUS, status)
    except BaseException as error:
        status.update(status="stopped_for_inspection", error=repr(error), ended_unix=time.time())
        write(STATUS, status)
        raise
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
