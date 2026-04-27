"""
Recreates figs/current_pebble_vs_ours.svg using matplotlib.
Reads the same CSV files as plot_current_results.py and produces
a two-panel figure (Walker + Door Open) comparing PEBBLE vs OURS.
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent

COLORS = {
    "PEBBLE": "#1f77b4",
    "OURS":   "#d62728",
}


def load_series(csv_path: Path):
    steps, rewards = [], []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            steps.append(float(row["step"]))
            rewards.append(float(row["episode_reward"]))
    return np.array(steps), np.array(rewards)


def smooth(y, window: int = 5):
    """Simple moving-average smoothing."""
    if window <= 1:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def fmt_k(x, _pos=None):
    if x >= 1_000_000:
        return f"{x/1_000_000:.0f}M"
    if x >= 1_000:
        return f"{x/1_000:.0f}k"
    return str(int(x))


def main():
    walker_base = (
        ROOT / "exp" / "walker_walk"
        / "H1024_L2_lr0.0005"
        / "teacher_b-1_g1_m0_s0_e0"
        / "init1000_unsup9000_inter20000_seg50_acttanh_Rlr0.0003_Rupdate50_en3_sample1_large_batch10_schedule_0_label_smooth_0.0"
    )
    door_base = (
        ROOT / "exp" / "metaworld_door-open-v2"
        / "H256_L3_lr0.0003"
        / "teacher_b-1_g1_m0_s0_e0"
        / "init1000_unsup9000_inter5000_seg50_acttanh_Rlr0.0003_Rupdate10_en3_sample1_large_batch10_schedule_0_label_smooth_0.0"
    )

    panels = {
        "Walker": {
            "PEBBLE": walker_base / "PEBBLE_maxfeed100_Rbatch10_seed12345" / "train.csv",
            "OURS":   walker_base / "Value_lambda_0.1_fix1_maxfeed100_Rbatch10_seed12345" / "train.csv",
        },
        "Door Open": {
            "PEBBLE": door_base / "PEBBLE_dooropen_fixbaseline_maxfeed1000_Rbatch10_seed12345" / "train.csv",
            "OURS":   door_base / "Value_lambda_0.1_fix1_dooropen_maxfeed1000_Rbatch10_seed12349" / "train.csv",
        },
    }

    plt.rcParams.update({
        "font.family":        "Arial",
        "font.size":          11,
        "axes.titlesize":     15,
        "axes.labelsize":     12,
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "legend.fontsize":    11,
        "axes.grid":          True,
        "grid.color":         "#eeeeee",
        "grid.linewidth":     1.0,
        "grid.linestyle":     "-",
        "axes.facecolor":     "white",
        "figure.facecolor":   "#fafafa",
        "lines.linewidth":    2.5,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.spines.left":   True,
        "axes.spines.bottom": True,
    })

    fig, axes = plt.subplots(1, 2, figsize=(13.3, 5.4))
    fig.suptitle("Current PEBBLE vs OURS Training Reward", fontsize=18, y=0.97)

    for ax, (panel_title, paths) in zip(axes, panels.items()):
        for label, path in paths.items():
            if not path.exists():
                print(f"[WARN] missing: {path}")
                continue
            steps, rewards = load_series(path)
            smoothed = smooth(rewards, window=5)
            ax.plot(steps, smoothed, color=COLORS[label], label=label, linewidth=2.5)

        ax.set_title(panel_title, fontsize=15)
        ax.set_xlabel("Environment Steps", fontsize=12)
        ax.set_ylabel("Episode Reward", fontsize=12)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt_k))
        ax.legend(frameon=False)
        ax.spines["bottom"].set_color("#444444")
        ax.spines["left"].set_color("#444444")

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out_dir = ROOT / "figs"
    out_dir.mkdir(exist_ok=True)

    out_svg = out_dir / "current_pebble_vs_ours_matplotlib.svg"
    out_png = out_dir / "current_pebble_vs_ours_matplotlib.png"
    fig.savefig(out_svg, format="svg", bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved: {out_svg}")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
