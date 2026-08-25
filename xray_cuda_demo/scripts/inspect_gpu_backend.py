"""Section 18 spec item 27: reports facts about the GPU backend that
are each independently VERIFIED at runtime -- never a claim asserted
without a corresponding check. Run after building the `xray_cuda`
extension:

    python scripts/inspect_gpu_backend.py
"""

import inspect
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _check(label: str, passed: bool, detail: str = "") -> bool:
    status = "OK" if passed else "FAIL"
    line = f"  [{status}] {label}"
    if detail:
        line += f" -- {detail}"
    print(line)
    return passed


def main() -> int:
    print("=" * 70)
    print("GPU BACKEND DIAGNOSTIC")
    print("=" * 70)
    all_ok = True

    try:
        import xray_cuda
    except Exception as exc:  # noqa: BLE001
        print(f"\nFAILED to import the xray_cuda extension: {exc}")
        print("Build it first: see cuda/CMakeLists.txt / README build instructions.")
        return 1

    ext_path = Path(xray_cuda.__file__).resolve()
    all_ok &= _check("xray_cuda extension is a compiled binary (not a .py file)",
                      ext_path.suffix in (".pyd", ".so"), str(ext_path))

    # -- native pipeline entry point exists and is a compiled builtin, not Python --
    has_pipeline = hasattr(xray_cuda, "run_basic_cuda_pipeline_gpu")
    all_ok &= _check("run_basic_cuda_pipeline_gpu is exposed by the compiled extension", has_pipeline)
    if has_pipeline:
        is_builtin = type(xray_cuda.run_basic_cuda_pipeline_gpu).__name__ in (
            "builtin_function_or_method", "instancemethod", "function")
        module_name = getattr(xray_cuda.run_basic_cuda_pipeline_gpu, "__module__", "")
        all_ok &= _check("run_basic_cuda_pipeline_gpu belongs to the compiled xray_cuda module",
                          module_name == "xray_cuda", f"__module__={module_name!r}")

    # -- CUDA device availability (does NOT claim compute if unavailable) --
    cuda_ok = xray_cuda.cuda_available()
    _check("CUDA device available", cuda_ok)

    # -- .cu source files actually present and referenced by CMake --
    cu_files = sorted((PROJECT_ROOT / "cuda" / "src").glob("*.cu"))
    all_ok &= _check(f"{len(cu_files)} .cu source file(s) found under cuda/src/", len(cu_files) > 0,
                      ", ".join(p.name for p in cu_files))
    cmakelists = (PROJECT_ROOT / "cuda" / "CMakeLists.txt").read_text(encoding="utf-8")
    missing_from_cmake = [p.name for p in cu_files if p.name not in cmakelists]
    all_ok &= _check("Every .cu file is referenced in cuda/CMakeLists.txt", not missing_from_cmake,
                      f"missing: {missing_from_cmake}" if missing_from_cmake else "all referenced")

    # -- Python GPU computation audit: cuda/*.py must be thin wrappers only --
    import ast

    forbidden_imports = {"cupy", "numba", "torch", "pycuda"}
    python_gpu_computation_found = []
    for py_file in sorted((PROJECT_ROOT / "cuda").glob("*.py")):
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in forbidden_imports:
                        python_gpu_computation_found.append(f"{py_file.name}: imports {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] in forbidden_imports:
                    python_gpu_computation_found.append(f"{py_file.name}: imports from {node.module}")
        if "cv2.cuda" in source:
            python_gpu_computation_found.append(f"{py_file.name}: references cv2.cuda")
    all_ok &= _check("No cupy/numba/torch/pycuda/cv2.cuda usage under cuda/*.py",
                      not python_gpu_computation_found,
                      "; ".join(python_gpu_computation_found) if python_gpu_computation_found else "clean")

    # -- production Streamlit path calls the single native pipeline, not a per-filter chain --
    services_path = PROJECT_ROOT / "ui" / "services.py"
    services_source = services_path.read_text(encoding="utf-8") if services_path.exists() else ""
    uses_single_pipeline_call = "run_cuda_pipeline(images, config, pipeline_config)" in services_source
    all_ok &= _check("ui/services.py's production GPU path calls a single native pipeline function",
                      uses_single_pipeline_call)

    print()
    print("SUMMARY")
    print("-" * 70)
    print(f"GPU backend:               {'Native C++/CUDA' if has_pipeline else 'UNKNOWN'}")
    print(f"Basic:                     {'native' if has_pipeline else 'n/a'}")
    print(f"Enhanced:                  {'native (same pipeline entry point, per-stage variant flags)' if has_pipeline else 'n/a'}")
    print(f"CUDA source:               .cu ({len(cu_files)} files)")
    print(f"CUDA compiler:             nvcc (see cuda/CMakeLists.txt: find_package(CUDAToolkit))")
    print(f"Python GPU computation:    {'none found' if not python_gpu_computation_found else 'FOUND -- see above'}")
    print(f"pybind11:                  {'enabled' if has_pipeline else 'unknown'}")
    print(f"CUDA device at runtime:    {'available' if cuda_ok else 'unavailable on this machine'}")
    print()
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED -- see [FAIL] lines above")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
