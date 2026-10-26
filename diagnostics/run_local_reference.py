"""Run the three frozen R02 LOCAL-2 jobs sequentially, stopping on any failure."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from data import file_hash
from diagnostics.temporal_dependence import require, read

CONFIG = Path("configs/local_attention_v1.json")
STATUS = Path("outputs/local_v2_queue.json")


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    import torch
    torch.set_num_threads(1)
    config = read(CONFIG)
    protocol = config["protocol"]
    require(torch.cuda.is_available(), "CUDA required for this frozen run")
    require(config["seeds"] == [42, 43, 44] and len(config["runs"]) == 3, "Unexpected budget")
    lock = Path("outputs/local_v2_queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    require(not STATUS.exists(), "Queue exists: inspect it before any restart")
    for spec in config["runs"]:
        output = Path(spec["segments"][0])
        require(not output.exists() and not output.with_suffix(".lineage.json").exists(),
                "Run exists: never overwrite or restart automatically")
    protected = {**protocol["training_source_sha256"], **config["protected_reference_files"],
                 config["data"]: protocol["data_sha256"], str(CONFIG): file_hash(CONFIG),
                 config["reference_manifest"]: config["reference_manifest_sha256"]}
    def verify():
        require(all(file_hash(p) == h for p, h in protected.items()), "Frozen file changed")
    verify()
    status = {"status": "running", "pid": os.getpid(), "started_unix": time.time(),
              "completed": [], "active": None, "plan_sha256": file_hash(CONFIG)}
    write(STATUS, status)
    try:
        for spec in config["runs"]:
            verify()
            seed, output = spec["seed"], Path(spec["segments"][0])
            command = [sys.executable, "train_real.py", "--data", config["data"],
                       "--variant", "base", "--attention-scope", "diagonal",
                       "--training-protocol-version", "2", "--ema-start-update", "1000",
                       "--device", "cuda", "--updates", "3000", "--microbatch-size", "4",
                       "--accumulation-steps", "8", "--validate-every", "250", "--scenarios", "100",
                       "--sampling-steps", "50", "--chunk-size", "16", "--seed", str(seed),
                       "--output", str(output)]
            record = {"seed": seed, "variant": "base", "attention_scope": "diagonal",
                      "output": str(output), "command": command,
                      "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                      "git_status": subprocess.check_output(["git", "status", "--porcelain"], text=True),
                      "training_source_sha256": protocol["training_source_sha256"],
                      "data_sha256": protocol["data_sha256"], "plan_sha256": file_hash(CONFIG),
                      "started_unix": time.time(), "exit_code": None}
            write(output.with_suffix(".lineage.json"), record)
            with output.with_suffix(".log").open("x") as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                status["active"] = {"seed": seed, "pid": process.pid, "output": str(output)}
                write(STATUS, status)
                code = process.wait()
            record.update(exit_code=code, ended_unix=time.time(),
                          sources_unchanged=all(file_hash(p) == h for p, h in protected.items()))
            if code == 0:
                record["last_checkpoint_sha256"] = file_hash(output / "checkpoint_last.pt")
                record["ema_start_checkpoint_sha256"] = file_hash(output / "checkpoint_ema_start.pt")
            write(output.with_suffix(".lineage.json"), record)
            require(code == 0 and record["sources_unchanged"], "Run failed or source changed")
            # Seed 42 is the first formal run and the engineering gate, regardless of its scores.
            from summarize_runs import audit_run
            audited = audit_run(spec, protocol)
            base_spec = next(r for r in read(config["reference_manifest"])["runs"]
                             if r["seed"] == seed and r["variant"] == "base")
            require(audited["initialization"] == read(Path(base_spec["segments"][0]) / "initialization.json"),
                    "LOCAL initialization differs from historical BASE")
            base = torch.load(Path(base_spec["segments"][-1]) / "checkpoint_last.pt",
                              map_location="cpu", weights_only=True)
            last = audited["checkpoint"]
            for key in ("noise_rng",):
                require(torch.equal(base[key], last[key]), "Training RNG differs")
            for key in ("order", "rng"):
                require(torch.equal(base["data"][key], last["data"][key]), "Data order differs")
            require(base["data"]["cursor"] == last["data"]["cursor"], "Data cursor differs")
            record["engineering_audit_passed"] = True
            write(output.with_suffix(".lineage.json"), record)
            status["completed"].append(str(output))
            status["active"] = None
            write(STATUS, status)
        verify()
        status.update(status="training_complete", ended_unix=time.time())
        write(STATUS, status)
    except BaseException as error:
        status.update(status="stopped_for_inspection", error=repr(error), ended_unix=time.time())
        write(STATUS, status)
        raise
    print(json.dumps(status, indent=2), flush=True)


if __name__ == "__main__":
    main()
