"""Environment diagnostics for ``--doctor``.

Reports the OS/CPU/RAM, the Python stack, the GPU (via the GPU backend) and
runs an end-to-end smoke test of both engines so a broken install is visible
before the first real batch.
"""

from __future__ import annotations

import importlib
import io
import os
import platform
import subprocess
import sys

import numpy as np

PACKAGES = [
    ("numpy", "numpy"),
    ("pillow", "PIL"),
    ("opencv-python-headless", "cv2"),
    ("piexif", "piexif"),
    ("cupy-cuda12x", "cupy"),
    ("nvidia-nvimgcodec-cu12", "nvidia.nvimgcodec"),
]


def _pkg_version(mod: str) -> str:
    try:
        from importlib.metadata import version
        pkg_name = mod.split(".")[0].replace("_", "-")
        return version(pkg_name)
    except Exception:
        pass
    try:
        m = importlib.import_module(mod)
    except Exception:
        return "missing"
    return str(getattr(m, "__version__", "?"))


def _ram_gib() -> str:
    try:
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return f"{stat.ullTotalPhys / 2 ** 30:.1f} GiB"
    except Exception:
        pass
    return "?"


def _driver_version() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0]
    except Exception:
        pass
    return "?"


def _cpu_engine_check() -> tuple[bool, str]:
    try:
        import cv2

        from .cpu_backend import CpuEngine
        from .geometry import Plan

        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[:, :, 2] = 200
        buf = CpuEngine().encode(img, 90, 2)
        eng = CpuEngine()
        out = eng.process(buf, Plan(64, 64, 32, 32, 0.0, 0.0, 64.0, 64.0), 90, 2)
        if not out.startswith(b"\xff\xd8"):
            return False, "encoder did not produce JPEG"
        return True, f"OpenCV {cv2.__version__}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _gpu_engine_check() -> tuple[bool, str, dict]:
    from .gpu_backend import GpuEngine

    try:
        eng = GpuEngine()
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}", {}
    ok, msg = eng.self_test()
    info = eng.info
    return ok, msg, {
        "device": info.device_name,
        "sm": info.compute_capability,
        "vram_gb": f"{info.total_vram_gb:.1f}",
        "vram_free_gb": f"{info.free_vram_gb:.1f}",
        "driver": info.driver_version,
        "cuda_runtime": info.cuda_runtime,
        "nvimgcodec": info.nvimgcodec_version,
    }


def doctor_report() -> int:
    print("=" * 68)
    print("batchresize environment check")
    print("=" * 68)
    print(f"OS           : {platform.system()} {platform.release()} "
          f"({platform.version()})")
    print(f"CPU          : {platform.processor() or '?'} "
          f"({os.cpu_count()} logical cores)")
    print(f"RAM          : {_ram_gib()}")
    print(f"Python       : {sys.version.split()[0]} ({sys.executable})")
    print(f"GPU driver   : {_driver_version()}")
    print("packages     : " + ", ".join(
        f"{name}={_pkg_version(mod)}" for name, mod in PACKAGES))
    print("-" * 68)

    cpu_ok, cpu_msg = _cpu_engine_check()
    print(f"CPU engine   : {'OK' if cpu_ok else 'FAIL'} ({cpu_msg})")

    gpu_ok, gpu_msg, info = _gpu_engine_check()
    if info:
        print(f"GPU device   : {info['device']} (SM {info['sm']}, "
              f"{info['vram_gb']} GiB, free {info['vram_free_gb']} GiB)")
        print(f"CUDA runtime : {info['cuda_runtime']}  "
              f"driver {info['driver']}  nvImageCodec {info['nvimgcodec']}")
    print(f"GPU engine   : {'OK' if gpu_ok else 'unavailable'} ({gpu_msg})")
    print(f"self-test    : {'PASS' if gpu_ok else 'FAIL'} - {gpu_msg}")
    print("=" * 68)

    if cpu_ok and gpu_ok:
        print("Both engines are ready.")
        return 0
    if cpu_ok:
        print("GPU path is not usable; runs will use --backend cpu.")
        return 0
    print("Neither engine works - install the CPU dependencies first.")
    return 2
