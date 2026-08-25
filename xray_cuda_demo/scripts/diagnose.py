"""One-command environment diagnostic for the X-ray CUDA demo project.

Usage:
    python scripts/diagnose.py
"""

import importlib.metadata
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "MISSING"


def _tool_version(cmd: list[str]) -> str:
    exe = shutil.which(cmd[0])
    if exe is None:
        return "MISSING"
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        first_line = (out.stdout or out.stderr).strip().splitlines()
        return first_line[0] if first_line else "UNKNOWN"
    except Exception as exc:  # pragma: no cover - diagnostic best effort
        return f"ERROR ({exc})"


def main() -> int:
    print("=" * 60)
    print("X-RAY CUDA DEMO ENVIRONMENT")
    print("=" * 60)

    print(f"\nPython:      {sys.version.split()[0]}")
    print(f"NVCC:        {_tool_version(['nvcc', '--version'])}")
    print(f"CMake:       {_tool_version(['cmake', '--version'])}")

    try:
        import xray_cuda

        cuda_ok = xray_cuda.cuda_available()
        print(f"CUDA Available: {'YES' if cuda_ok else 'NO'}")

        if cuda_ok:
            info = xray_cuda.device_info()
            print(f"GPU:                  {info['name']}")
            print(f"Compute Capability:   {info['compute_capability']}")
            vram_mb = info["total_global_mem_bytes"] / (1024 * 1024)
            print(f"VRAM:                 {vram_mb:.0f} MiB")
            print(f"Multiprocessors:      {info['multiprocessor_count']}")
            print(f"Max threads/block:    {info['max_threads_per_block']}")
        else:
            print("GPU:                  N/A (no usable CUDA device)")

        try:
            result = xray_cuda.smoke_test()
            expected = [1.0, 2.0, 3.0, 4.0, 5.0]
            passed = list(result["output"]) == expected
            print(f"\nSmoke Test:  {'PASS' if passed else 'FAIL'}")
            print(f"  output={result['output']} kernel_ms={result['kernel_ms']:.4f}")
        except Exception as exc:
            print(f"\nSmoke Test:  FAIL ({exc})")

    except ImportError as exc:
        print(f"CUDA Available: UNKNOWN (xray_cuda extension not built: {exc})")
        print("\nSmoke Test:  SKIPPED (extension not built)")

    print(f"\nPyBind11:    {_pkg_version('pybind11')}")
    print(f"NumPy:       {_pkg_version('numpy')}")
    print(f"OpenCV:      {_pkg_version('opencv-python')}")
    print(f"Streamlit:   {_pkg_version('streamlit')}")
    print(f"Pytest:      {_pkg_version('pytest')}")

    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
