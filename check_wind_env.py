#!/usr/bin/env python3
"""Dependency-free environment report; optional isolated FP32 CUDA smoke check."""

import argparse
import ctypes
from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time


def error_text(error):
    return "{}: {}".format(type(error).__name__, error)


def run_command(command, timeout=15):
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, errors="replace", timeout=timeout
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": error_text(error)}


def memory_info():
    """Report OS-visible memory; container limits may be lower."""
    try:
        if platform.system() == "Linux":
            values = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, value = line.split(":", 1)
                if key in ("MemTotal", "MemAvailable"):
                    values[key] = int(value.split()[0]) * 1024
            return {"source": "/proc/meminfo", "bytes": values}
        if platform.system() == "Windows":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_uint32),
                    ("load", ctypes.c_uint32),
                ] + [(name, ctypes.c_uint64) for name in (
                    "total_physical", "available_physical", "total_pagefile",
                    "available_pagefile", "total_virtual", "available_virtual",
                    "available_extended_virtual",
                )]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                raise OSError("GlobalMemoryStatusEx failed")
            return {
                "source": "GlobalMemoryStatusEx",
                "bytes": {"MemTotal": status.total_physical,
                          "MemAvailable": status.available_physical},
            }
        return {"error": "Memory inspection is implemented for Windows and Linux only"}
    except (OSError, ValueError, AttributeError) as error:
        return {"error": error_text(error)}


def probe_torch(device_index=0):
    result = {"status": "NEEDS_REVIEW"}
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError as error:
        if error.name == "torch":
            result["status"] = "TORCH_NOT_INSTALLED"
        result["error"] = error_text(error)
        return result
    except Exception as error:
        result["error"] = error_text(error)
        return result

    try:
        result.update({
            "version": str(torch.__version__),
            "compiled_cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        })
        if not result["cuda_available"]:
            result["status"] = "CUDA_UNAVAILABLE"
            return result
        result["device_count"] = torch.cuda.device_count()
        if not 0 <= device_index < result["device_count"]:
            raise ValueError("Selected CUDA device index is out of range")
        device = torch.device("cuda", device_index)
        properties = torch.cuda.get_device_properties(device)
        result["selected_device"] = {
            "index": device_index, "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
        }
        torch.manual_seed(0)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 16), torch.nn.SiLU(), torch.nn.Linear(16, 1)
        ).to(device=device, dtype=torch.float32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        x = torch.randn(4, 24, 8, device=device, dtype=torch.float32)
        target = torch.randn(4, 24, 1, device=device, dtype=torch.float32)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        optimizer.zero_grad()
        loss = (model(x) - target).square().mean()
        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite FP32 loss")
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all().item()
                   for p in model.parameters()):
            raise RuntimeError("Missing or non-finite FP32 gradient")
        optimizer.step()
        if not all(torch.isfinite(p).all().item() for p in model.parameters()):
            raise RuntimeError("Non-finite parameter after optimizer step")
        changed = any(not torch.equal(old, new) for old, new in
                      zip(before, model.parameters()))
        if not changed:
            raise RuntimeError("Optimizer did not change any parameters")
        torch.cuda.synchronize(device)
        result["fp32_check"] = {
            "loss": loss.item(), "parameters_changed": changed,
            "elapsed_seconds": time.perf_counter() - started,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "scope": "Small MLP only; not full DiT memory or throughput validation",
        }
        result["status"] = "CUDA_BASIC_OK"
    except Exception as error:
        result["error"] = error_text(error)
    return result


def isolated_torch_probe(device_index, timeout):
    # A crashing or hanging native import must not prevent the main JSON report.
    with tempfile.TemporaryDirectory(prefix="wind_env_") as directory:
        output = Path(directory) / "torch.json"
        process = run_command([
            sys.executable, str(Path(__file__).resolve()),
            "--torch-probe", str(output), "--device", str(device_index),
        ], timeout=timeout)
        if process.get("returncode") == 0 and output.exists():
            try:
                result = json.loads(output.read_text(encoding="utf-8"))
                if not isinstance(result, dict) or result.get("status") not in {
                    "CUDA_BASIC_OK", "TORCH_NOT_INSTALLED", "CUDA_UNAVAILABLE",
                    "NEEDS_REVIEW",
                }:
                    raise ValueError("Invalid torch probe report")
                result["process"] = process
                return result
            except (OSError, ValueError) as error:
                process["report_error"] = error_text(error)
        return {"status": "NEEDS_REVIEW", "process": process}


def collect_report(device_index=0, timeout=45):
    disk = shutil.disk_usage(Path.cwd())
    nvidia_smi = shutil.which("nvidia-smi")
    wsl_smi = Path("/usr/lib/wsl/lib/nvidia-smi")
    if nvidia_smi is None and wsl_smi.is_file():
        nvidia_smi = str(wsl_smi)
    driver = run_command([
        nvidia_smi, "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]) if nvidia_smi else {"error": "nvidia-smi not found"}
    torch_info = isolated_torch_probe(device_index, timeout)
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": torch_info["status"],
        "system": {
            "os": platform.system(), "release": platform.release(),
            "architecture": platform.machine(),
            "is_wsl": "microsoft" in platform.release().lower()
                      or bool(os.environ.get("WSL_INTEROP")),
        },
        "python": {"version": platform.python_version(),
                   "executable": sys.executable,
                   "minimum_version_ok": sys.version_info >= (3, 8)},
        "working_directory": str(Path.cwd()),
        "memory": memory_info(),
        "disk": {"total_bytes": disk.total, "free_bytes": disk.free},
        "nvidia_smi": {"path": nvidia_smi, "memory_unit": "MiB", **driver},
        "torch": torch_info,
        "limitations": [
            "Results describe this process environment, not another host or interpreter.",
            "OS memory may exceed container or scheduler limits.",
            "CUDA_BASIC_OK covers a small FP32 MLP, not the proposed DiT or AMP.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/environment"))
    parser.add_argument("--device", type=int, default=0, help="Visible CUDA device index")
    parser.add_argument("--timeout", type=int, default=45, help="Torch probe timeout in seconds")
    parser.add_argument("--torch-probe", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.device < 0 or args.timeout < 1:
        parser.error("--device must be nonnegative and --timeout must be positive")
    if args.torch_probe:
        args.torch_probe.write_text(json.dumps(probe_torch(args.device)), encoding="utf-8")
        return
    report = collect_report(args.device, args.timeout)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = args.output_dir / ("env_report_" + stamp + ".json")
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print("Status: " + report["status"])
    print("Report: " + str(output.resolve()))
    # Successful diagnostics can report unavailable CUDA; exit 0 means report saved.


if __name__ == "__main__":
    main()
