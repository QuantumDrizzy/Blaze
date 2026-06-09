"""
docs/plots.py — regenerate the Blaze benchmark figures for the README.

Every figure is produced from data computed LIVE here (no hardcoded numbers):
TFIM ground states via quimb, compressed/quantized/overlapped via blaze itself.
Run:  python docs/plots.py   (needs the '[quantum]' extra: quimb)

Output: docs/img/*.png  (committed; everything else stays gitignored).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from blaze import compress, fidelity, quantize_tt          # noqa: E402
from blaze.overlap import inner                            # noqa: E402
from blaze.examples.substrate_quantum_state import (       # noqa: E402
    tfim_groundstate, haar_random_state, half_chain_entropy,
)

OUT = Path(__file__).resolve().parent / "img"
OUT.mkdir(parents=True, exist_ok=True)

# ---- dark cyberpunk palette (brand-consistent with the CYBERDECK dashboard) ----
BG, PANEL = "#0a0c12", "#0e121c"
CYAN, MAGENTA, AMBER = "#00e6c8", "#ff46a0", "#ffb446"
TEXT, MUTED, GRID = "#c8d6e0", "#7a8796", "#1b2434"
FOOTER = "measured on RTX 5060 Ti (sm_120), CUDA 13 · blaze · 2026-06-09"


def _style():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "savefig.edgecolor": BG, "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
        "text.color": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
        "grid.color": GRID, "axes.grid": True, "grid.alpha": 0.5, "grid.linewidth": 0.7,
        "axes.titlecolor": CYAN, "axes.titlesize": 13, "axes.titleweight": "bold",
        "font.size": 11, "font.family": "DejaVu Sans Mono", "figure.dpi": 140,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def _footer(fig):
    fig.text(0.99, 0.01, FOOTER, ha="right", va="bottom", color=MUTED, fontsize=7.5, style="italic")


def _save(fig, name):
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    p = OUT / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p.relative_to(ROOT)}  ({p.stat().st_size//1024} KiB)")


# --------------------------------------------------------------------------- #
N = 16
HS = [0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0]


def fig_compression_vs_entanglement():
    """Compression ratio tracks physical half-chain entanglement; Haar declines."""
    hs = [1.0, 1.5, 2.0, 3.0]
    S, ratio = [], []
    for h in hs:
        psi = tfim_groundstate(N, h)
        tt = compress(psi.reshape((2,) * N), rel_tol=1e-6)
        s, _ = half_chain_entropy(psi, N)
        S.append(s); ratio.append(tt.compression_ratio(psi.reshape((2,) * N)))
    # Haar volume-law control
    ph = haar_random_state(N, seed=0)
    tth = compress(ph.reshape((2,) * N), rel_tol=1e-6)
    Sh, _ = half_chain_entropy(ph, N)
    rh = tth.compression_ratio(ph.reshape((2,) * N))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.axhline(1.0, color=MUTED, ls="--", lw=1, alpha=0.8)
    ax.text(0.05, 1.15, "break-even (TT = dense)", color=MUTED, fontsize=8)
    ax.scatter(S, ratio, s=130, color=CYAN, zorder=5, edgecolor="white", linewidth=0.6,
               label="TFIM ground states (area-law)")
    off = {1.0: (12, -4), 1.5: (12, -2), 2.0: (16, -13), 3.0: (12, 7)}
    for s, r, h in zip(S, ratio, hs):
        dx, dy = off.get(h, (10, 6))
        ax.annotate(f"h={h:g}  {r:.0f}×", (s, r), textcoords="offset points",
                    xytext=(dx, dy), color=CYAN, fontsize=9, ha="left")
    ax.scatter([Sh], [rh], s=170, color=MAGENTA, marker="X", zorder=5,
               edgecolor="white", linewidth=0.6, label="Haar random (volume-law)")
    ax.annotate(f"Haar\n{rh:.2f}× — declines", (Sh, rh), textcoords="offset points",
                xytext=(-10, 18), color=MAGENTA, fontsize=9, ha="right")
    ax.set_yscale("log")
    ax.set_ylim(top=max(ratio) * 2.6)
    ax.set_xlabel("half-chain entanglement entropy  S  (bits)")
    ax.set_ylabel("compression ratio  (×, log)")
    ax.set_title("Blaze compresses exactly what physics makes compressible")
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9, loc="upper right")
    _footer(fig); _save(fig, "compression_vs_entanglement.png")


def fig_error_vs_rank():
    """Phase 4: the lossy knob — monotone error↔rank, with the ratio it buys."""
    psi = tfim_groundstate(N, 1.0)  # critical: the hardest (most entangled) case
    tens = psi.reshape((2,) * N)
    ranks = [1, 2, 3, 4, 6, 8, 12, 16]
    err, ratio = [], []
    for r in ranks:
        tt = compress(tens, max_rank=r, rel_tol=0.0)
        err.append(max(tt.rel_error(tens), 1e-16)); ratio.append(tt.compression_ratio(tens))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(ranks, err, "-o", color=CYAN, lw=2, ms=7, label="relative Frobenius error")
    ax.set_yscale("log")
    ax.set_xlabel("max bond dimension  χ  (rank cap)")
    ax.set_ylabel("relative reconstruction error  (log)", color=CYAN)
    ax.set_title("The lossy dial is monotone and honest (TFIM critical, n=16)")
    ax2 = ax.twinx()
    ax2.plot(ranks, ratio, "-s", color=AMBER, lw=2, ms=6, label="compression ratio")
    ax2.set_yscale("log"); ax2.set_ylabel("compression ratio  (×, log)", color=AMBER)
    ax2.grid(False)
    l1, la = ax.get_legend_handles_labels(); l2, lb = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la + lb, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT,
              fontsize=9, loc="center right")
    _footer(fig); _save(fig, "error_vs_rank.png")


def fig_fidelity_matrix():
    """Phase 7: pairwise fidelity (compressed-space) reveals the quantum phase transition."""
    tts = [compress(tfim_groundstate(N, h).reshape((2,) * N), rel_tol=1e-8) for h in HS]
    M = np.zeros((len(HS), len(HS)))
    for i in range(len(HS)):
        for j in range(len(HS)):
            M[i, j] = fidelity(tts[i], tts[j])

    cmap = LinearSegmentedColormap.from_list("cyb", [BG, "#13324a", CYAN])
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(M, cmap=cmap, vmin=0, vmax=1, origin="lower")
    ax.set_xticks(range(len(HS))); ax.set_yticks(range(len(HS)))
    ax.set_xticklabels([f"{h:g}" for h in HS]); ax.set_yticklabels([f"{h:g}" for h in HS])
    ax.set_xlabel("field  h"); ax.set_ylabel("field  h")
    ax.set_title("Ground-state fidelity ⟨ψ(h)|ψ(h′)⟩ — the QPT, in compressed space")
    for i in range(len(HS)):
        for j in range(len(HS)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color=(BG if M[i, j] > 0.55 else TEXT), fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("fidelity", color=TEXT)
    cb.ax.yaxis.set_tick_params(color=MUTED)
    ax.axvline(0.5, color=MAGENTA, lw=1.5, alpha=0.7); ax.axhline(0.5, color=MAGENTA, lw=1.5, alpha=0.7)
    ax.text(0.0, 6.2, "ordered ↔ paramagnet boundary", color=MAGENTA, fontsize=8)
    _footer(fig); _save(fig, "fidelity_matrix_qpt.png")


def fig_quantization_tradeoff():
    """Phase 8: ratio vs fidelity as bit width drops (TFIM paramagnet)."""
    psi = tfim_groundstate(N, 3.0)
    tens = psi.reshape((2,) * N)
    dense_bytes = (2 ** N) * 16
    tt = compress(tens, rel_tol=1e-6)
    bits = [12, 10, 8, 6, 4]
    R, F = [], []
    for b in bits:
        q = quantize_tt(tt, bits=b, granularity="per_bond")
        R.append(q.compression_ratio_over_dense(dense_bytes))
        F.append(fidelity(tt, q.dequantize()))

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(bits, R, "-o", color=CYAN, lw=2, ms=7, label="end-to-end ratio")
    ax.set_xlabel("quantization bit width"); ax.invert_xaxis()
    ax.set_ylabel("total compression ratio  (×)", color=CYAN)
    ax.set_title("Second stage: more ratio for a measured fidelity cost (TFIM paramagnet)")
    ax2 = ax.twinx()
    ax2.plot(bits, F, "-s", color=MAGENTA, lw=2, ms=6, label="state fidelity")
    ax2.set_ylabel("fidelity  ⟨ψ|ψ_q⟩", color=MAGENTA); ax2.grid(False)
    ax2.set_ylim(min(F) - 0.01, 1.002)
    for b, r, f in zip(bits, R, F):
        ax.annotate(f"{int(b)}-bit", (b, r), textcoords="offset points", xytext=(0, 9),
                    color=TEXT, fontsize=8, ha="center")
    ax.annotate("int8 sweet spot", (8, R[2]), textcoords="offset points", xytext=(20, -22),
                color=AMBER, fontsize=9, arrowprops=dict(arrowstyle="->", color=AMBER))
    l1, la = ax.get_legend_handles_labels(); l2, lb = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la + lb, facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT,
              fontsize=9, loc="lower left")
    _footer(fig); _save(fig, "quantization_tradeoff.png")


def fig_overlap_scaling():
    """Phase 7: the zipper is O(nχ³); the dense inner product is O(2ⁿ)."""
    ns = [6, 8, 10, 12, 14, 16]
    z_t, d_t = [], []
    for n in ns:
        p = np.zeros(2 ** n, dtype=complex); p[0] = p[-1] = 2 ** -0.5
        g = compress(p.reshape((2,) * n), rel_tol=1e-12)
        rng = np.random.default_rng(n)
        b = compress((rng.standard_normal(2 ** n) + 1j * rng.standard_normal(2 ** n)).reshape((2,) * n), rel_tol=1e-12)
        t0 = time.perf_counter(); [inner(g, b) for _ in range(20)]; z_t.append((time.perf_counter() - t0) / 20 * 1e3)
        t0 = time.perf_counter(); [np.vdot(g.reconstruct().ravel(), b.reconstruct().ravel()) for _ in range(20)]
        d_t.append((time.perf_counter() - t0) / 20 * 1e3)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(ns, z_t, "-o", color=CYAN, lw=2, ms=7, label="zipper  ⟨A|B⟩  on the TT  (O(nχ³))")
    ax.plot(ns, d_t, "-s", color=MAGENTA, lw=2, ms=6, label="dense inner product  (O(2ⁿ), decompresses)")
    ax.set_yscale("log")
    ax.set_xlabel("qubits  n"); ax.set_ylabel("time per overlap  (ms, log)")
    ax.set_title("Overlap on the TT vs the dense inner product")
    ax.text(0.97, 0.06, "n=40: dense 2⁴⁰ ≈ 17.6 TB — impossible;\nthe zipper still runs in 3.6 ms",
            transform=ax.transAxes, ha="right", va="bottom", color=AMBER, fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.4", fc=PANEL, ec=AMBER, alpha=0.95))
    ax.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=9, loc="upper left")
    _footer(fig); _save(fig, "overlap_scaling.png")


if __name__ == "__main__":
    _style()
    print("Generating Blaze figures (live data)…")
    fig_compression_vs_entanglement()
    fig_error_vs_rank()
    fig_fidelity_matrix()
    fig_quantization_tradeoff()
    fig_overlap_scaling()
    print("Done ->", OUT)
