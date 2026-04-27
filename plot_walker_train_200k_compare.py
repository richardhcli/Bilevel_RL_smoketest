from pathlib import Path
import csv


ROOT = Path(__file__).resolve().parent
WSL_ROOT = Path(r"\\wsl$\Ubuntu\root\projects\Bi-level-Reinforcement-Learning-main")


def load_train_points(csv_path: Path, max_step: float):
    steps = []
    rewards = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            step = row.get("step")
            reward = row.get("episode_reward")
            if not step or not reward:
                continue
            step_f = float(step)
            if step_f > max_step:
                continue
            steps.append(step_f)
            rewards.append(float(reward))
    return steps, rewards


def main():
    source_root = ROOT if (ROOT / "exp").exists() else WSL_ROOT
    base = (
        source_root
        / "exp"
        / "walker_walk"
        / "H1024_L2_lr0.0005"
        / "teacher_b-1_g1_m0_s0_e0"
        / "init1000_unsup9000_inter20000_seg50_acttanh_Rlr0.0003_Rupdate50_en3_sample1_large_batch10_schedule_0_label_smooth_0.0"
    )

    max_step = 200000.0
    series = {
        "PEBBLE": load_train_points(base / "PEBBLE_walker_refresh_maxfeed100_Rbatch10_seed12345" / "train.csv", max_step),
        "OURS": load_train_points(base / "Value_lambda_0.1_fix1_maxfeed100_Rbatch10_seed12345" / "train.csv", max_step),
    }

    width = 860
    height = 460
    pad_left = 78
    pad_right = 24
    pad_top = 44
    pad_bottom = 60
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    all_rewards = [reward for _, rewards in series.values() for reward in rewards]
    min_x, max_x = 0.0, max_step
    min_y, max_y = min(all_rewards), max(all_rewards)
    if min_y == max_y:
        min_y -= 1
        max_y += 1
    y_pad = 0.08 * (max_y - min_y)
    min_y -= y_pad
    max_y += y_pad

    def sx(v):
        return pad_left + (v - min_x) / (max_x - min_x) * plot_w

    def sy(v):
        return pad_top + plot_h - (v - min_y) / (max_y - min_y) * plot_h

    colors = {"PEBBLE": "#1f77b4", "OURS": "#d62728"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fafafa"/>',
        '<text x="430" y="30" text-anchor="middle" font-size="22" font-family="Arial">Walker Training Episode Reward: PEBBLE vs OURS (0-200k)</text>',
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

    tick_steps = [0, 50000, 100000, 150000, 200000]
    for xv in tick_steps:
        parts.append(f'<text x="{sx(xv):.1f}" y="{pad_top + plot_h + 24:.1f}" text-anchor="middle" font-size="11" font-family="Arial">{int(xv/1000)}k</text>')

    for i in range(5):
        frac = i / 4
        yv = max_y - frac * (max_y - min_y)
        parts.append(f'<text x="{pad_left - 8:.1f}" y="{pad_top + frac * plot_h + 4:.1f}" text-anchor="end" font-size="11" font-family="Arial">{int(yv)}</text>')

    legend_x = width - 160
    legend_y = 72
    for i, label in enumerate(["PEBBLE", "OURS"]):
        ly = legend_y + i * 22
        parts.append(f'<line x1="{legend_x:.1f}" y1="{ly:.1f}" x2="{legend_x + 28:.1f}" y2="{ly:.1f}" stroke="{colors[label]}" stroke-width="2.5"/>')
        parts.append(f'<text x="{legend_x + 36:.1f}" y="{ly + 4:.1f}" font-size="12" font-family="Arial">{label}</text>')

    parts.extend([
        f'<text x="{width / 2:.1f}" y="{height - 14:.1f}" text-anchor="middle" font-size="12" font-family="Arial">Environment Steps</text>',
        f'<text x="22" y="{height / 2:.1f}" text-anchor="middle" font-size="12" font-family="Arial" transform="rotate(-90 22 {height / 2:.1f})">Training Episode Reward</text>',
        "</svg>",
    ])

    out_path = ROOT / "figs" / "walker_train_pebble_vs_ours_200k.svg"
    out_path.write_text("\n".join(parts), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
