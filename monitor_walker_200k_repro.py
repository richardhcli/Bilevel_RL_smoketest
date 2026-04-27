import csv
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "exp" / "walker_walk" / "H1024_L2_lr0.0005" / "teacher_b-1_g1_m0_s0_e0" / "init1000_unsup9000_inter20000_seg50_acttanh_Rlr0.0003_Rupdate50_en3_sample1_large_batch10_schedule_0_label_smooth_0.0"
FIGS = ROOT / "figs"
STATUS_PATH = FIGS / "walker_200k_repro_status.txt"
OUT_PATH = FIGS / "walker_200k_fresh_vs_historical.svg"
TARGET_STEP = 200000
POLL_SECONDS = 60


RUNS = {
    "PEBBLE": {
        "fresh": BASE / "PEBBLE_200k_fresh_maxfeed100_Rbatch10_seed12345" / "train.csv",
        "historical": BASE / "PEBBLE_maxfeed100_Rbatch10_seed12345" / "train.csv",
        "color_fresh": "#1f77b4",
        "color_hist": "#9ecae1",
    },
    "OURS": {
        "fresh": BASE / "Value_lambda_0.1_fix1_200k_fresh_maxfeed100_Rbatch10_seed12345" / "train.csv",
        "historical": BASE / "Value_lambda_0.1_fix1_maxfeed100_Rbatch10_seed12345" / "train.csv",
        "color_fresh": "#d62728",
        "color_hist": "#f7b6b2",
    },
}


def load_series(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                rows.append((float(row["step"]), float(row["episode_reward"])))
            except (KeyError, ValueError):
                continue
    return rows


def last_step(rows):
    return int(rows[-1][0]) if rows else 0


def compare_rows(fresh_rows, hist_rows):
    hist_by_step = {int(step): reward for step, reward in hist_rows}
    deltas = []
    for step, reward in fresh_rows:
        step_i = int(step)
        if step_i in hist_by_step:
            deltas.append(abs(reward - hist_by_step[step_i]))
    if not deltas:
        return 0, None, None
    return len(deltas), sum(deltas) / len(deltas), max(deltas)


def svg_polyline(points, color, width=2.5, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<polyline fill="none" stroke="{color}" stroke-width="{width}"{dash_attr} '
        f'points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in points)}" />'
    )


def make_panel(title, series_map, x, y, width, height):
    pad_left = 64
    pad_right = 18
    pad_top = 34
    pad_bottom = 42
    plot_x0 = x + pad_left
    plot_y0 = y + pad_top
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    all_steps = [step for rows, _, _ in series_map.values() for step, _ in rows]
    all_rewards = [reward for rows, _, _ in series_map.values() for _, reward in rows]
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
        f'<text x="{x + width / 2:.1f}" y="{y + 20}" text-anchor="middle" font-size="16" font-family="Arial">{title}</text>',
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

    for _, (rows, color, dash) in series_map.items():
        points = [(sx(step), sy(reward)) for step, reward in rows]
        parts.append(svg_polyline(points, color, dash=dash))

    for i in range(5):
        frac = i / 4
        xv = min_x + frac * (max_x - min_x)
        yv = max_y - frac * (max_y - min_y)
        parts.append(f'<text x="{plot_x0 + frac * plot_w:.1f}" y="{plot_y0 + plot_h + 20:.1f}" text-anchor="middle" font-size="11" font-family="Arial">{int(xv/1000)}k</text>')
        parts.append(f'<text x="{plot_x0 - 8:.1f}" y="{plot_y0 + frac * plot_h + 4:.1f}" text-anchor="end" font-size="11" font-family="Arial">{int(yv)}</text>')

    legend_x = plot_x0 + plot_w - 170
    legend_y = plot_y0 + 12
    legend_items = [
        ("Historical", "#777777", "7,4"),
        ("Fresh", "#777777", None),
    ]
    for i, (label, color, dash) in enumerate(legend_items):
        ly = legend_y + i * 18
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(f'<line x1="{legend_x:.1f}" y1="{ly:.1f}" x2="{legend_x + 26:.1f}" y2="{ly:.1f}" stroke="{color}" stroke-width="2.5"{dash_attr}/>')
        parts.append(f'<text x="{legend_x + 34:.1f}" y="{ly + 4:.1f}" font-size="12" font-family="Arial">{label}</text>')

    parts.append(f'<text x="{x + width / 2:.1f}" y="{y + height - 8:.1f}" text-anchor="middle" font-size="12" font-family="Arial">Environment Steps</text>')
    parts.append(
        f'<text x="{x + 18:.1f}" y="{y + height / 2:.1f}" text-anchor="middle" font-size="12" font-family="Arial" '
        f'transform="rotate(-90 {x + 18:.1f} {y + height / 2:.1f})">Episode Reward</text>'
    )
    return "\n".join(parts)


def write_plot(data):
    width = 1280
    height = 540
    panel_w = 590
    panel_h = 430
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fafafa"/>',
        '<text x="640" y="28" text-anchor="middle" font-size="22" font-family="Arial">Walker 200k Fresh vs Historical Training Reward</text>',
    ]
    positions = [(30, 68), (660, 68)]
    for (title, series_map), (x, y) in zip(data.items(), positions):
        parts.append(make_panel(title, series_map, x, y, panel_w, panel_h))
    parts.append("</svg>")
    OUT_PATH.write_text("\n".join(parts), encoding="utf-8")


def write_status(message_lines):
    FIGS.mkdir(exist_ok=True)
    STATUS_PATH.write_text("\n".join(message_lines) + "\n", encoding="utf-8")


def main():
    while True:
        status_lines = []
        ready = True
        plot_data = {}
        for run_name, paths in RUNS.items():
            fresh_rows = load_series(paths["fresh"])
            hist_rows = load_series(paths["historical"])
            fresh_step = last_step(fresh_rows)
            hist_step = last_step(hist_rows)
            compared_points, mean_abs_delta, max_abs_delta = compare_rows(fresh_rows, hist_rows)
            status_lines.append(
                f"{run_name}: fresh_step={fresh_step} historical_step={hist_step} "
                f"compared_points={compared_points} mean_abs_delta={mean_abs_delta} max_abs_delta={max_abs_delta}"
            )
            if fresh_step < TARGET_STEP:
                ready = False
            plot_data[run_name] = {
                "Historical": (hist_rows, paths["color_hist"], "7,4"),
                "Fresh": (fresh_rows, paths["color_fresh"], None),
            }

        write_status(status_lines)
        if ready:
            write_plot(plot_data)
            write_status(status_lines + [f"done: wrote {OUT_PATH}"])
            break
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
