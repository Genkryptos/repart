"""Small helpers for parametric OpenSCAD source."""

from __future__ import annotations

import re

_ASSIGN = r"^(\s*){name}\s*=\s*[^;]*;"


def bake_params(code: str, params: dict[str, float]) -> str:
    """Rewrite each top-level `name = ...;` to the fitted value, so the saved .scad
    is self-contained; params with no assignment in the code are prepended."""
    missing = []
    for name, value in params.items():
        pat = re.compile(_ASSIGN.format(name=re.escape(name)), re.MULTILINE)
        new, k = pat.subn(lambda m: f"{m.group(1)}{name} = {value:g};", code, count=1)
        if k:
            code = new
        else:
            missing.append(f"{name} = {value:g};")
    return ("\n".join(missing) + "\n" + code) if missing else code


def declared(code: str) -> set[str]:
    """Names assigned at the start of a line (top-level parameters, roughly)."""
    return set(re.findall(r"^([A-Za-z_]\w*)\s*=", code, re.MULTILINE))
