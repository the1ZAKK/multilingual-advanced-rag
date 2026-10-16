"""Read-only local preflight; reports tool availability, never credential values."""

import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys

from rag.config import Settings
from rag.generation.ollama import OllamaGenerator


def memory_gib():
    if sys.platform == "win32":
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("load", ctypes.c_ulong),
                *[
                    (name, ctypes.c_ulonglong)
                    for name in [
                        "total",
                        "available",
                        "page_total",
                        "page_available",
                        "virtual_total",
                        "virtual_available",
                        "extended",
                    ]
                ],
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return round(status.total / 1024**3, 1)
        return None
    try:
        return round(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024**3, 1)
    except (ValueError, OSError, AttributeError):
        return None


def main():
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "ram_gib": memory_gib(),
        "cpu_threads": os.cpu_count(),
        "tools": {
            name: bool(shutil.which(name)) for name in ["git", "docker", "ollama", "nvidia-smi"]
        },
    }
    if shutil.which("git"):
        report["git"] = subprocess.check_output(["git", "--version"], text=True).strip()
    if importlib.util.find_spec("torch"):
        import torch

        report.update(
            torch=torch.__version__,
            cuda_available=torch.cuda.is_available(),
            cuda_runtime=torch.version.cuda,
        )
    report["ollama"] = OllamaGenerator(Settings()).status()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
