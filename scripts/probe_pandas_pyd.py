"""Probe which pandas compiled modules are blocked by the OS policy.

Loads every *.pyd under pandas/_libs with a module name that matches the DLL's
exported PyInit_ symbol, then reports OK vs BLOCKED.
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path

libs = Path(sys.prefix) / "Lib/site-packages/pandas/_libs"
results = {}
for pyd in sorted(libs.glob("*.pyd")):
    name = pyd.name.split(".")[0]
    try:
        loader = importlib.machinery.ExtensionFileLoader(name, str(pyd))
        spec = importlib.util.spec_from_loader(loader.name, loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
        results[name] = "OK"
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).splitlines()[0]
        if "Application Control" in msg:
            results[name] = "BLOCKED"
        else:
            results[name] = f"ERR: {msg[:70]}"

ok = sum(1 for v in results.values() if v == "OK")
blocked = sum(1 for v in results.values() if v == "BLOCKED")
print(f"OK={ok} BLOCKED={blocked} OTHER={len(results) - ok - blocked}")
for name, status in results.items():
    print(f"  {name:20s} {status}")
