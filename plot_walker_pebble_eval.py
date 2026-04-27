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
    csv_path = (
        source_root
        / "exp"
        / "walker_walk"
        / "H1024_L2_lr0.0005"
        / "teacher_b-1_g1_m0_s0_e0"
        / "init1000_unsup9000_inter20000_seg50_acttanh_Rlr0.0003_Rupdate50_en3_sample1_large_batch10_schedule_0_label_smooth_0.0"
        / "PEBBLE_walker_refresh_maxfeed100_Rbatch10_seed12345"
        / "eval.csv"
    )

    steps, rewards = load_eval_points(csv_path)
    if not steps:
        raise SystemExit("No eval points found.")

    width = 720
    height = 420
    pad_left = 70
    pad_right = 20
    pad_top = 40
    pad_bottom = 55
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    min_x, max_x = min(steps), max(steps)
    min_y, max_y = min(rewards), max(rewards)
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

    points = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(steps, rewards))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fafafa"/>',
        '<text x="360" y="26" text-anchor="middle" font-size="22" font-family="Arial">Walker PEBBLE Eval Episode Reward</text>',
    ]

    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        gx = pad_left + frac * plot_w
        gy = pad_top + frac * plot_h
        parts.append(f'<line x1="{gx:.1f}" y1="{pad_top:.1f}" x2="{gx:.1f}" y2="{pad_top + plot_h:.1f}" stroke="#eeeeee"/>')
        parts.append(f'<line x1="{pad_left:.1f}" y1="{gy:.1f}" x2="{pad_left + plot_w:.1f}" y2="{gy:.1f}" stroke="#eeeeee"/>')

    parts.extend([
        f'<line x1="{pad_left:.1f}" y1="{pad_top + plot_h:.1f}" x2="{pad_left + plot_w:.1f}" y2="{pad_top + plot_h:.1f}" stroke="#444"/>',
        f'<line x1="{pad_left:.1f}" y1="{pad_top:.1f}" x2="{pad_left:.1f}" y2="{pad_top + plot_h:.1f}" stroke="#444"/>',
        f'<polyline fill="none" stroke="#1f77b4" stroke-width="2.5" points="{points}" />',
    ])

    for i in range(5):
        frac = i / 4
        xv = min_x + frac * (max_x - min_x)
        yv = max_y - frac * (max_y - min_y)
        parts.append(f'<text x="{pad_left + frac * plot_w:.1f}" y="{pad_top + plot_h + 22:.1f}" text-anchor="middle" font-size="11" font-family="Arial">{int(xv/1000)}k</text>')
        parts.append(f'<text x="{pad_left - 8:.1f}" y="{pad_top + frac * plot_h + 4:.1f}" text-anchor="end" font-size="11" font-family="Arial">{int(yv)}</text>')

    parts.extend([
        f'<text x="{width / 2:.1f}" y="{height - 12:.1f}" text-anchor="middle" font-size="12" font-family="Arial">Environment Steps</text>',
        f'<text x="20" y="{height / 2:.1f}" text-anchor="middle" font-size="12" font-family="Arial" transform="rotate(-90 20 {height / 2:.1f})">Eval Episode Reward</text>',
        "</svg>",
    ])

    out_path = ROOT / "figs" / "walker_pebble_eval.svg"
    out_path.write_text("\n".join(parts), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
