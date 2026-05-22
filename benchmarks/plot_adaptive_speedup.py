# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Plot speedup over the base model from a benchmark_adaptive_draft.py result file.

Usage:
    python benchmarks/plot_adaptive_speedup.py results/adaptive_draft_wave_results.json
    python benchmarks/plot_adaptive_speedup.py results/adaptive_draft_wave_results.json --out results/speedup.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot(data: dict, out_path: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    waves = data["waves"]
    summary = data["summary"]
    variants = [v for v in summary if v != "base"]
    colours = {"int8": "C1", "fp8": "C2", "adaptive": "C3"}

    base_per_wave = [w["base"] for w in waves]
    wave_indices = [w["index"] for w in waves]
    wave_labels = [f"{'S' if w['type'] == 'small' else 'L'}{w['index']}" for w in waves]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8))

    # ── Top: per-wave speedup lines ───────────────────────────────────────────
    ax1.axhline(1.0, color="grey", linewidth=0.8, linestyle="--", label="base (1.0×)")
    for v in variants:
        speedups = [w[v] / w["base"] if w["base"] > 0 else 0.0 for w in waves]
        ax1.plot(wave_indices, speedups, marker="o", label=v, color=colours.get(v))

    for w in waves:
        shade = "#d0e8ff" if w["type"] == "small" else "#ffe8d0"
        ax1.axvspan(w["index"] - 0.5, w["index"] + 0.5, color=shade, alpha=0.3, zorder=0)

    ax1.set_xticks(wave_indices)
    ax1.set_xticklabels(wave_labels)
    ax1.set_xlabel("Wave")
    ax1.set_ylabel("Speedup vs base")
    ax1.set_title("Per-wave speedup over base model")
    ax1.legend()

    # ── Bottom: small / large / overall speedup bars ──────────────────────────
    regimes = ["small_avg", "large_avg", "overall"]
    regime_labels = ["Small waves", "Large waves", "Overall"]
    base_vals = {r: summary["base"][r] for r in regimes}

    x = np.arange(len(variants))
    bar_w = 0.22
    regime_colours = ["#4c9be8", "#e87c4c", "#6abf69"]

    for ri, (regime, label, colour) in enumerate(zip(regimes, regime_labels, regime_colours)):
        speedups = [summary[v][regime] / base_vals[regime] if base_vals[regime] > 0 else 0.0
                    for v in variants]
        offset = (ri - 1) * bar_w
        bars = ax2.bar(x + offset, speedups, bar_w, label=label, color=colour)
        for bar, sp in zip(bars, speedups):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                     f"{sp:.2f}×", ha="center", va="bottom", fontsize=8)

    ax2.axhline(1.0, color="grey", linewidth=0.8, linestyle="--")
    ax2.set_xticks(x)
    ax2.set_xticklabels(variants)
    ax2.set_ylabel("Speedup vs base")
    ax2.set_title("Average speedup over base model by regime")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Plot saved to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot speedup over the base model from a wave benchmark result."
    )
    parser.add_argument("input", help="Path to JSON result file")
    parser.add_argument("--out", default=None,
                        help="Output PNG path (default: same stem as input + _speedup.png)")
    args = parser.parse_args()

    data = load(args.input)
    out = args.out or str(Path(args.input).with_stem(Path(args.input).stem + "_speedup").with_suffix(".png"))
    plot(data, out)


if __name__ == "__main__":
    main()
