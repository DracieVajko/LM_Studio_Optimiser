#!/usr/bin/env python3
"""Generate docs/LM_STUDIO_PARAMETER_MATRIX.md from the parameter registry.

Single source of truth: edit lm_optimizer/services/parameter_registry.py,
then re-run this script. Never hand-edit the generated document.
"""

import sys
from pathlib import Path

from lm_optimizer.services.parameter_registry import PARAMETERS, matrix_rows


def main() -> int:
    rows = matrix_rows()
    lines = [
        "# LM Studio Parameter Matrix (generated)",
        "",
        "Generated from `lm_optimizer/services/parameter_registry.py` — "
        "do not hand-edit; re-run `scripts/generate_parameter_matrix.py`.",
        "",
        "Control: REST = native API verified live; CLI = `lms` flags verified; "
        "SDK/schema = known but programmatically unverified; none = not exposed. "
        "Verified: yes = apply+verify works; no = rejected/unsupported; "
        "partial = version- or model-dependent.",
        "",
        "| Name | Meaning | Backend | API | CLI | SDK | Optimizer status | "
        "Experimental | Lifecycle | Verification method | Known limitations |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    by_name = {p.canonical: p for p in PARAMETERS}
    for row in rows:
        p = by_name[row["name"]]
        if row["verified"] == "yes":
            how = "live request-by-request + echo compare"
        elif "unrecognized" in p.failure_behavior or "400" in p.failure_behavior:
            how = "safe 400-probe"
        else:
            how = "schema/CLI inspection"
        lines.append(
            f"| {row['name']} | {p.display} | {p.backend} | {p.rest} | "
            f"{p.cli} | {p.sdk} | "
            f"{'searched' if p.optimizer_enabled else 'not searched'} | "
            f"{row['experimental']} | {row['lifecycle']} | {how} | {p.failure_behavior} |"
        )
    out = Path("docs/LM_STUDIO_PARAMETER_MATRIX.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(rows)} parameters)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
