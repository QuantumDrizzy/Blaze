"""Blaze's benchmark card: compression ratio, and the error that bought it.

Every bar is a number already measured and documented. Nothing is re-run here:
    TFIM n = 16, critical and paramagnet (lossless to ~1e-7): SUBSTRATE-validation.md
    paramagnet + int8 / 4-bit cores: PHASE8-results.md (fidelity 0.99994 at int8)
    QuBLAR ghost bits, 20-bit exact posterior: QuBLAR RESULTS-phase6 (TT rank 1-2)
    Haar-random state, the control: SUBSTRATE-validation.md (TT larger than dense)
Blaze compresses only what has structure, so the control is drawn, not dropped.

Writes docs/img/blaze_bench_card.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ROWS = [
    ("Haar-random state (control)", 0.38, "TT larger than dense: not compressed", "#888780"),
    ("TFIM n=16, critical", 19, "lossless to ~1e-7", "#7F77DD"),
    ("TFIM n=16, paramagnet", 77, "lossless to ~1e-7", "#7F77DD"),
    ("paramagnet + int8 cores", 447, "fidelity 0.99994", "#1D9E75"),
    ("paramagnet + 4-bit cores", 705, "quantized, error composed", "#1D9E75"),
    ("QuBLAR ghost bits (2^20 posterior)", 26214, "marginals to 7e-7", "#D85A30"),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 4.6), dpi=130)
    fig.patch.set_facecolor("white")
    names = [r[0] for r in ROWS]
    ax.barh(names, [r[1] for r in ROWS], color=[r[3] for r in ROWS])
    ax.set_xscale("log")
    ax.axvline(1.0, color="#444441", lw=0.8, ls=":")
    for i, (_, ratio, note, _) in enumerate(ROWS):
        ax.text(ratio * 1.12, i, f"{ratio:g}×  ·  {note}", va="center", fontsize=9, color="#2C2C2A")
    ax.set_xlim(0.2, 10**6)
    ax.set_xlabel("compression ratio (dense / TT parameters, log scale)")
    ax.set_title("Blaze · TT/MPS compression, measured · it compresses only what has structure",
                 fontsize=12)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.text(0.01, 0.01, "sources: SUBSTRATE-validation.md, PHASE8-results.md (this repo); "
             "QuBLAR RESULTS-phase6 · RTX 5060 Ti (sm_120)", fontsize=7.5, color="#5F5E5A")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    out = ROOT / "docs" / "img" / "blaze_bench_card.png"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
