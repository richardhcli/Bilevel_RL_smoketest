from pathlib import Path
import csv


ROOT = Path(__file__).resolve().parent
WSL_ROOT = Path(r"\\wsl$\Ubuntu\root\projects\Bi-level-Reinforcement-Learning-main")


def load_eval_points(csv_path: Path):
    steps = []
    rewards = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            steps.append(float(row["step"]))
            rewards.append(float(row["episode_reward"]))
    return steps, rewards


def main():
    source_root = ROOT if (ROOT / "exp").exists() else WSL_ROOT
    base = (
        source_root
        / "exp"
        / "metaworld_door-open-v2"
        / "H256_L3_lr0.0003"
        / "teacher_b-1_g1_m0_s0_e0"
        / "init1000_unsup9000_inter5000_seg50_acttanh_Rlr0.0003_Rupdate10_en3_sample1_large_batch10_schedule_0_label_smooth_0.0"
    )

    series = {
        "PEBBLE": load_eval_points(base / "PEBBLE_dooropen_fixbaseline_maxfeed1000_Rbatch10_seed12345" / "eval.csv"),
        "OURS": load_eval_points(base / "Value_lambda_0.1_fix1_dooropen_maxfeed1000_Rbatch10_seed12349" / "eval.csv"),
    }

    width = 760
    height = 430
    pad_left = 72
    pad_right = 20
    pad_top = 42
    pad_bottom = 56
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    all_steps = [step for steps, _ in series.values() for step in steps]
    all_rewards = [reward for _, rewards in series.values() for reward in rewards]
    min_x, max_x = min(all_steps), max(all_steps)
    min_y, max_y = min(all_rewards), max(all_rewards)
    if min_y == max_y:
        min_y -= 1
        max_y += 1
    y_pad = 0.08 * (max_y - min_y)
    min_y -= y_pad
    max_y += y_pad

    def sx(v):
        return pad_left + (v - min_x) / (max_x - min_x) * plot_w if max_x > min_x else pad_left + plot_w / 2

    def sy(v):
        return pad_top + plot_h - (v - min_y) / (max_y - min_y) * plot_h

    colors = {"PEBBLE": "#1f77b4", "OURS": "#d62728"}

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fafafa"/>',
        '<text x="380" y="28" text-anchor="middle" font-size="22" font-family="Arial">Door Open Eval Episode Reward: PEBBLE vs OURS</text>',
    ]

    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        gx = pad_left + frac * plot_w
        gy = pad_top + frac * plot_h
        parts.append(f'<line x1="{gx:.1f}" y1="{pad_top:.1f}" x2="{gx:.1f}" y2="{pad_top + plot_h:.1f}" stroke="#eeeeee"/>')
        parts.append(f'<line x1="{pad_left:.1f}" y1="{gy:.1f}" x2="{pad_left + plot_w:.1f}" y2="{gy:.1f}" stroke="#eeeeee"/>')

    parts.extend([
        f'<line x1="{pad_left:.1f}" y1="{pad_top + plot_h:.1f}" x2="{pad_left + plot_w:.1f}" y2="{pad_top + plot_h:.1f}" stroke="#444"/>',
        f'<line x1="{pad_left:.1f}" y1="{pad_top:.1f}" x2="{pad_left:.1f}" y2="{pad_top + plot_h:.1f}" stroke="#444"/>',
    ])

    for label, (steps, rewards) in series.items():
        points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(steps, rewards))
        parts.append(f'<polyline fill="none" stroke="{colors[label]}" stroke-width="2.5" points="{points}" />')

    for i in range(5):
        frac = i / 4
        xv = min_x + frac * (max_x - min_x)
        yv = max_y - frac * (max_y - min_y)
        parts.append(f'<text x="{pad_left + frac * plot_w:.1f}" y="{pad_top + plot_h + 22:.1f}" text-anchor="middle" font-size="11" font-family="Arial">{int(xv/1000)}k</text>')
        parts.append(f'<text x="{pad_left - 8:.1f}" y="{pad_top + frac * plot_h + 4:.1f}" text-anchor="end" font-size="11" font-family="Arial">{int(yv)}</text>')

    legend_x = width - 150
    legend_y = 70
    for i, label in enumerate(["PEBBLE", "OURS"]):
        ly = legend_y + i * 20
        parts.append(f'<line x1="{legend_x:.1f}" y1="{ly:.1f}" x2="{legend_x + 26:.1f}" y2="{ly:.1f}" stroke="{colors[label]}" stroke-width="2.5"/>')
        parts.append(f'<text x="{legend_x + 34:.1f}" y="{ly + 4:.1f}" font-size="12" font-family="Arial">{label}</text>')

    parts.extend([
        f'<text x="{width / 2:.1f}" y="{height - 14:.1f}" text-anchor="middle" font-size="12" font-family="Arial">Environment Steps</text>',
        f'<text x="20" y="{height / 2:.1f}" text-anchor="middle" font-size="12" font-family="Arial" transform="rotate(-90 20 {height / 2:.1f})">Eval Episode Reward</text>',
        "</svg>",
    ])

    out_path = ROOT / "figs" / "door_eval_pebble_vs_ours.svg"
    out_path.write_text("\n".join(parts), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
