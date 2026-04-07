from __future__ import annotations

import importlib
import platform
import sys

MODULES = [
    "torch",
    "transformers",
    "sentence_transformers",
    "faiss",
    "datasets",
    "scipy",
    "sklearn",
    "numpy",
    "pandas",
    "tqdm",
    "pytest",
]


def main() -> int:
    print("CAEM environment check")
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print("-" * 60)

    failed: list[tuple[str, str]] = []

    for mod in MODULES:
        try:
            imported = importlib.import_module(mod)
            version = getattr(imported, "__version__", "unknown")
            print(f"[OK]   {mod:<22} {version}")
        except Exception as exc:
            failed.append((mod, f"{type(exc).__name__}: {exc}"))
            print(f"[FAIL] {mod:<22} {type(exc).__name__}: {exc}")

    print("-" * 60)
    if failed:
        print(f"Result: FAIL ({len(failed)} module import failures)")
        for mod, err in failed:
            print(f"  - {mod}: {err}")
        return 1

    print("Result: OK (all required modules imported)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
