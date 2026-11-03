"""Bounded H0 -> H1 -> H24 queue, with frozen inputs and no automatic retries."""
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
from diagnostics.summarize_history_information import audit_history_run, summarize
from diagnostics.temporal_dependence import read, require


CONFIG = Path("configs/history_information_v1/execution.json")
STATUS = Path("outputs/history_information_v1_queue.json")


def main():
    torch.set_num_threads(1)
    require(torch.cuda.is_available(), "CUDA required")
    frozen = read(CONFIG)
    require([s["group"] for s in frozen["runs"]] == ["H0", "H1", "H24"], "Unexpected queue")
    lock = Path("outputs/history_information_v1_queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(not STATUS.exists(), "Queue exists; inspect before continuing")
    protected = {**frozen["protected_artifacts"], **frozen["training_source_sha256"],
                 **frozen["implementation_sha256"], frozen["data"]: frozen["data_sha256"],
                 str(CONFIG): file_hash(CONFIG), **frozen["plan_sha256"]}
    def verify():
        require(all(file_hash(p) == h for p, h in protected.items()), "Frozen input/source changed")
    verify()
    for spec in frozen["runs"]:
        require(len(spec["segments"]) == 1, "Initial queue must start from scratch")
        path = Path(spec["segments"][0])
        require(not path.exists() and not path.with_suffix(".lineage.json").exists()
                and not path.with_suffix(".log").exists(), "Run output already exists")
    require(not Path(frozen["report_output"]).exists(), "Report already exists")
    status = {"status": "running", "pid": os.getpid(), "started_unix": time.time(),
              "completed": [], "active": None, "plan_sha256": file_hash(CONFIG)}
    write(STATUS, status)
    first = None
    try:
        for spec in frozen["runs"]:
            verify()
            group, output = spec["group"], Path(spec["segments"][0])
            command = [sys.executable, "train_real.py", "--data", frozen["data"],
                       "--history-group", group, "--variant", "base", "--attention-scope", "full",
                       "--training-protocol-version", "2", "--ema-start-update", "1000",
                       "--supervision-mode", "final-only", "--loss-weights", "0", "1",
                       "--device", "cuda", "--updates", "3000", "--microbatch-size", "4",
                       "--accumulation-steps", "8", "--validate-every", "250", "--scenarios", "100",
                       "--sampling-steps", "50", "--chunk-size", "16", "--seed", "42", "--output", str(output)]
            record = {"group": group, "seed": 42, "command": command, "started_unix": time.time(),
                      "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                      "git_status": subprocess.check_output(["git", "status", "--porcelain"], text=True),
                      "training_source_sha256": frozen["training_source_sha256"],
                      "data_sha256": frozen["data_sha256"], "plan_sha256": file_hash(CONFIG), "exit_code": None}
            write(output.with_suffix(".lineage.json"), record)
            with output.with_suffix(".log").open("x") as log:
                env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
                status["active"] = {"group": group, "pid": child.pid, "output": str(output)}
                write(STATUS, status)
                code = child.wait()
            record.update(exit_code=code, ended_unix=time.time(),
                          sources_unchanged=all(file_hash(p) == h for p, h in protected.items()))
            if code == 0:
                record["last_checkpoint_sha256"] = file_hash(output / "checkpoint_last.pt")
            write(output.with_suffix(".lineage.json"), record)
            require(code == 0 and record["sources_unchanged"], "Run failed; inspect artifacts before resume")
            audited = audit_history_run(spec, frozen)
            if first is None:
                first = audited
            require(audited["initialization"] == first["initialization"], "Initial tensors differ")
            require(torch.equal(audited["state"]["noise_rng"], first["state"]["noise_rng"]), "Training noise differs")
            for key in ("order", "rng"):
                require(torch.equal(audited["state"]["data"][key], first["state"]["data"][key]), "Data order differs")
            require(audited["state"]["data"]["cursor"] == first["state"]["data"]["cursor"], "Data cursor differs")
            record["engineering_audit_passed"] = True
            write(output.with_suffix(".lineage.json"), record)
            status["completed"].append(group)
            status["active"] = None
            write(STATUS, status)
        verify()
        status.update(status="training_complete", training_ended_unix=time.time())
        write(STATUS, status)
        report = summarize(frozen, Path(frozen["report_output"]))
        status.update(status="report_complete_pending_review", ended_unix=time.time(),
                      report_output=frozen["report_output"], report_rows=len(report["rows"]))
        write(STATUS, status)
    except BaseException as error:
        status.update(status="stopped_for_inspection", error=repr(error), ended_unix=time.time())
        write(STATUS, status)
        raise
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
