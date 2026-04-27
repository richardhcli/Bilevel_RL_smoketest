import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WSL_ROOT = Path(r"\\wsl$\Ubuntu\root\projects\Bi-level-Reinforcement-Learning-main")


def load_series(csv_path: Path):
    steps = []
    rewards = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            steps.append(float(row["step"]))
            rewards.append(float(row["episode_reward"]))
    return steps, rewards


def svg_polyline(points, color):
    return (
        f'<polyline fill="none" stroke="{color}" stroke-width="2.5" '
        f'points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in points)}" />'
    )


def make_panel(title, series_map, x, y, width, height):
    pad_left = 64
    pad_right = 16
    pad_top = 28
    pad_bottom = 42
    plot_x0 = x + pad_left
    plot_y0 = y + pad_top
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    all_steps = [step for steps, _ in series_map.values() for step in steps]
    all_rewards = [reward for _, rewards in series_map.values() for reward in rewards]
    min_x, max_x = min(all_steps), max(all_steps)
    min_y, max_y = min(all_rewards), max(all_rewards)
    if min_y == max_y:
        min_y -= 1
        max_y += 1
    y_pad = 0.08 * (max_y - min_y)
    min_y -= y_pad
    max_y += y_pad

    def sx(v):
        return plot_x0 + (v - min_x) / (max_x - min_x) * plot_w

    def sy(v):
        return plot_y0 + plot_h - (v - min_y) / (max_y - min_y) * plot_h

    parts = [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="white" stroke="#dddddd"/>',
        f'<text x="{x + width / 2:.1f}" y="{y + 18}" text-anchor="middle" font-size="16" font-family="Arial">{title}</text>',
    ]

    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        gx = plot_x0 + frac * plot_w
        gy = plot_y0 + frac * plot_h
        parts.append(f'<line x1="{gx:.1f}" y1="{plot_y0:.1f}" x2="{gx:.1f}" y2="{plot_y0 + plot_h:.1f}" stroke="#eeeeee"/>')
        parts.append(f'<line x1="{plot_x0:.1f}" y1="{gy:.1f}" x2="{plot_x0 + plot_w:.1f}" y2="{gy:.1f}" stroke="#eeeeee"/>')

    parts.extend([
        f'<line x1="{plot_x0:.1f}" y1="{plot_y0 + plot_h:.1f}" x2="{plot_x0 + plot_w:.1f}" y2="{plot_y0 + plot_h:.1f}" stroke="#444"/>',
        f'<line x1="{plot_x0:.1f}" y1="{plot_y0:.1f}" x2="{plot_x0:.1f}" y2="{plot_y0 + plot_h:.1f}" stroke="#444"/>',
    ])

    colors = {"PEBBLE": "#1f77b4", "OURS": "#d62728"}
    for label, (steps, rewards) in series_map.items():
        points = [(sx(step), sy(reward)) for step, reward in zip(steps, rewards)]
        parts.append(svg_polyline(points, colors[label]))

    x_ticks = 5
    for i in range(x_ticks):
        frac = i / (x_ticks - 1)
        tick_val = min_x + frac * (max_x - min_x)
        tick_x = plot_x0 + frac * plot_w
        parts.append(f'<text x="{tick_x:.1f}" y="{plot_y0 + plot_h + 20:.1f}" text-anchor="middle" font-size="11" font-family="Arial">{int(tick_val/1000)}k</text>')

    y_ticks = 5
    for i in range(y_ticks):
        frac = i / (y_ticks - 1)
        tick_val = min_y + (1 - frac) * (max_y - min_y)
        tick_y = plot_y0 + frac * plot_h
        parts.append(f'<text x="{plot_x0 - 8:.1f}" y="{tick_y + 4:.1f}" text-anchor="end" font-size="11" font-family="Arial">{int(tick_val)}</text>')

    legend_x = plot_x0 + plot_w - 110
    legend_y = plot_y0 + 12
    for i, label in enumerate(["PEBBLE", "OURS"]):
        ly = legend_y + i * 18
        parts.append(f'<line x1="{legend_x:.1f}" y1="{ly:.1f}" x2="{legend_x + 26:.1f}" y2="{ly:.1f}" stroke="{colors[label]}" stroke-width="2.5"/>')
        parts.append(f'<text x="{legend_x + 34:.1f}" y="{ly + 4:.1f}" font-size="12" font-family="Arial">{label}</text>')

    parts.append(f'<text x="{x + width / 2:.1f}" y="{y + height - 8:.1f}" text-anchor="middle" font-size="12" font-family="Arial">Environment Steps</text>')
    parts.append(
        f'<text x="{x + 18:.1f}" y="{y + height / 2:.1f}" text-anchor="middle" font-size="12" font-family="Arial" '
        f'transform="rotate(-90 {x + 18:.1f} {y + height / 2:.1f})">Episode Reward</text>'
    )
    return "\n".join(parts)


def main():
    source_root = ROOT if (ROOT / "exp").exists() else WSL_ROOT

    walker_base = source_root / "exp" / "walker_walk" / "H1024_L2_lr0.0005" / "teacher_b-1_g1_m0_s0_e0" / "init1000_unsup9000_inter20000_seg50_acttanh_Rlr0.0003_Rupdate50_en3_sample1_large_batch10_schedule_0_label_smooth_0.0"
    door_base = source_root / "exp" / "metaworld_door-open-v2" / "H256_L3_lr0.0003" / "teacher_b-1_g1_m0_s0_e0" / "init1000_unsup9000_inter5000_seg50_acttanh_Rlr0.0003_Rupdate10_en3_sample1_large_batch10_schedule_0_label_smooth_0.0"

    runs = {
        "Walker": {
            "PEBBLE": walker_base / "PEBBLE_maxfeed100_Rbatch10_seed12345" / "train.csv",
            "OURS": walker_base / "Value_lambda_0.1_fix1_maxfeed100_Rbatch10_seed12345" / "train.csv",
        },
        "Door Open": {
            "PEBBLE": door_base / "PEBBLE_dooropen_fixbaseline_maxfeed1000_Rbatch10_seed12345" / "train.csv",
            "OURS": door_base / "Value_lambda_0.1_fix1_dooropen_maxfeed1000_Rbatch10_seed12349" / "train.csv",
        },
    }

    width = 1280
    height = 520
    panel_w = 590
    panel_h = 420
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fafafa"/>',
        '<text x="640" y="28" text-anchor="middle" font-size="22" font-family="Arial">Current PEBBLE vs OURS Training Reward</text>',
    ]

    positions = [(30, 60), (660, 60)]
    for (title, paths), (x, y) in zip(runs.items(), positions):
        series_map = {}
        for label, path in paths.items():
            if path.exists():
                series_map[label] = load_series(path)
        if series_map:
            svg_parts.append(make_panel(title, series_map, x, y, panel_w, panel_h))

    svg_parts.append("</svg>")

    output_path = ROOT / "figs" / "current_pebble_vs_ours.svg"
    output_path.write_text("\n".join(svg_parts), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
