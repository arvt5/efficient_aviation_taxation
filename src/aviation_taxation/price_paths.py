"""Square average focal price path plots for ticket-tax policies."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np

from aviation_taxation.model import FractionalParams


FOCAL_RANKS = (2, 1, 0)

# Replicated from analytics/paper_figures/thesis_style.py for this hand-written SVG.
COLORWAY = ["#1F3A5F", "#B65E3C", "#7C8C52", "#6E5A7E", "#B08A3E", "#4F6D64"]
PAPER_BG = "#FFFFFF"
FONT_COLOR = "#1F2933"
AXIS_LINE = "#BCCCDC"
GRID_COLOR = "#E9EEF2"
FONT_FAMILY = "Helvetica, Arial, sans-serif"


def _params_from_result(path: Path) -> FractionalParams:
    meta = json.loads(path.read_text())
    return FractionalParams(**meta["params"])


def _weighted_price(P: np.ndarray, mu: np.ndarray, rank: int, n: int) -> float:
    shape = [1, 1, 1]
    shape[rank] = n + 1
    stocked = np.arange(n + 1).reshape(shape) > 0
    weights = mu * stocked
    denom = float(weights.sum())
    if denom <= 0.0:
        return float("nan")
    return float((weights * P[..., rank]).sum() / denom)


def average_price_path(policy: Path, params: FractionalParams) -> list[dict[str, float]]:
    z = np.load(policy, allow_pickle=True)
    phi_grid = np.asarray(z["phi_grid"], dtype=float)
    prices = np.asarray(z["prices"], dtype=float)
    occupancy = np.asarray(z["occupancy"], dtype=float)
    n = int(params.N)
    d = float(params.d)
    entry_phi = float(params.entry_phi)
    out: list[dict[str, float]] = []
    first_active = int(np.argmax(phi_grid >= entry_phi - 1e-10)) if entry_phi > 0.0 else 0

    for life, rank in enumerate(FOCAL_RANKS):
        if life == 0:
            start = first_active
            base = 0.0
            phi_offset = entry_phi
        elif life == 1:
            start = 0
            base = (1.0 - entry_phi) * d
            phi_offset = 0.0
        else:
            start = 0
            base = (2.0 - entry_phi) * d
            phi_offset = 0.0
        for k in range(start, len(phi_grid)):
            phi = float(phi_grid[k])
            x = base + (phi - phi_offset) * d
            y = _weighted_price(prices[k], occupancy[k], rank, n)
            if np.isfinite(y):
                out.append(
                    {
                        "time": x,
                        "price": y,
                        "phase": float(life),
                        "rank": float(rank),
                        "phi": phi,
                    }
                )
    if out and out[0]["time"] > 1e-12:
        first = dict(out[0])
        first["time"] = 0.0
        out.insert(0, first)
    return out


def _scale(value: float, vmin: float, vmax: float, lo: float, hi: float) -> float:
    if vmax <= vmin:
        return 0.5 * (lo + hi)
    return lo + (value - vmin) * (hi - lo) / (vmax - vmin)


def _events(params: FractionalParams) -> tuple[list[tuple[float, str]], list[tuple[float, float]]]:
    d = float(params.d)
    e = float(params.entry_phi)
    if e <= 1e-12:
        return ([(1.0, "departure/entry"), (2.0, "departure/entry")], [])
    departures = [((1.0 - e) * d, "departure"), ((2.0 - e) * d, "departure")]
    entries = [(1.0 * d, "entry"), (2.0 * d, "entry")]
    shaded = [((1.0 - e) * d, 1.0 * d), ((2.0 - e) * d, 2.0 * d)]
    return sorted(departures + entries, key=lambda x: x[0]), shaded


def render_svg(
    rows: list[dict[str, float]],
    params: FractionalParams,
    usd_per_model_unit: float | None = None,
    ymin_model: float = 0.3,
    ymax_model: float = 1.1,
) -> str:
    width = 720
    height = 720
    left = 86
    right = 86
    top = 48
    bottom = 86
    plot_w = width - left - right
    plot_h = height - top - bottom
    price_color = COLORWAY[0]
    entry_color = COLORWAY[1]
    departure_color = COLORWAY[2]
    xs = np.array([r["time"] for r in rows], dtype=float)
    price_scale = float(usd_per_model_unit) if usd_per_model_unit is not None else 1.0
    price_unit = "USD" if usd_per_model_unit is not None else None
    ys = np.array([r["price"] for r in rows], dtype=float) * price_scale
    xmin = 0.0
    xmax = float(params.T)
    ymin = float(ymin_model) * price_scale
    ymax = float(ymax_model) * price_scale
    events, shaded = _events(params)

    def sx(x: float) -> float:
        return _scale(x, xmin, xmax, left, left + plot_w)

    def sy(y: float) -> float:
        return _scale(y, ymin, ymax, top + plot_h, top)

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img">'
    )
    parts.append(f'<rect x="0" y="0" width="720" height="720" fill="{PAPER_BG}"/>')
    parts.append(
        f'<defs><clipPath id="plot-area"><rect x="{left}" y="{top}" '
        f'width="{plot_w}" height="{plot_h}"/></clipPath></defs>'
    )
    for a, b in shaded:
        parts.append(
            f'<rect x="{sx(a):.2f}" y="{top}" width="{sx(b)-sx(a):.2f}" height="{plot_h}" '
            f'fill="{price_color}" opacity="0.10"/>'
        )
        parts.append(
            f'<text x="{0.5 * (sx(a) + sx(b)):.2f}" y="{top + 21}" text-anchor="middle" '
            f'font-family="{FONT_FAMILY}" font-size="12" fill="{price_color}">duopoly</text>'
        )

    for tick in np.linspace(0.0, float(params.T), 7):
        x = sx(float(tick))
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="{GRID_COLOR}"/>')
    for tick in np.linspace(ymin, ymax, 6):
        y = sy(float(tick))
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="{GRID_COLOR}"/>')

    for xval, label in events:
        x = sx(xval)
        dash = "4 4" if label == "entry" else "2 3"
        color = entry_color if label == "entry" else departure_color
        parts.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" '
            f'stroke="{color}" stroke-width="1.4" stroke-dasharray="{dash}"/>'
        )
        parts.append(
            f'<text x="{x:.2f}" y="{top - 9}" text-anchor="middle" '
            f'font-family="{FONT_FAMILY}" font-size="12" fill="{color}">{html.escape(label)}</text>'
        )

    poly = " ".join(
        f"{sx(float(r['time'])):.2f},{sy(float(r['price']) * price_scale):.2f}"
        for r in rows
    )
    parts.append(
        f'<polyline points="{poly}" fill="none" stroke="{price_color}" stroke-width="3.2" '
        f'stroke-linejoin="round" clip-path="url(#plot-area)"/>'
    )

    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="{AXIS_LINE}" stroke-width="1.0"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="{AXIS_LINE}" stroke-width="1.0"/>')

    for tick in np.linspace(0.0, float(params.T), 7):
        x = sx(float(tick))
        parts.append(f'<line x1="{x:.2f}" y1="{top + plot_h}" x2="{x:.2f}" y2="{top + plot_h + 6}" stroke="{AXIS_LINE}"/>')
        parts.append(
            f'<text x="{x:.2f}" y="{top + plot_h + 24}" text-anchor="middle" '
            f'font-family="{FONT_FAMILY}" font-size="12" fill="{FONT_COLOR}">{tick:.1f}</text>'
        )
    for tick in np.linspace(ymin, ymax, 6):
        y = sy(float(tick))
        parts.append(f'<line x1="{left - 6}" y1="{y:.2f}" x2="{left}" y2="{y:.2f}" stroke="{AXIS_LINE}"/>')
        tick_label = f"{tick:.0f}" if price_unit == "USD" else f"{tick:.2f}"
        parts.append(
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-family="{FONT_FAMILY}" font-size="12" fill="{FONT_COLOR}">{tick_label}</text>'
        )

    parts.append(
        f'<text x="{left + plot_w / 2:.2f}" y="{height - 26}" text-anchor="middle" '
        f'font-family="{FONT_FAMILY}" font-size="14" fill="{FONT_COLOR}">time since focal entry</text>'
    )
    y_axis_title = "average posted price [USD]" if price_unit == "USD" else "average posted price"
    parts.append(
        f'<text x="24" y="{top + plot_h / 2:.2f}" text-anchor="middle" '
        f'transform="rotate(-90 24 {top + plot_h / 2:.2f})" '
        f'font-family="{FONT_FAMILY}" font-size="14" fill="{FONT_COLOR}">{y_axis_title}</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


def write_outputs(svg: str, out_html: Path, out_svg: Path) -> None:
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_svg.parent.mkdir(parents=True, exist_ok=True)
    out_svg.write_text(svg)
    out_html.write_text(
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        "<title>Average price path</title>"
        f"<style>body{{margin:24px;font-family:{FONT_FAMILY};background:{PAPER_BG};color:{FONT_COLOR}}}"
        ".wrap{max-width:760px}svg{max-width:100%;height:auto}</style></head>"
        f"<body><div class=\"wrap\">{svg}</div></body></html>"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--result", type=Path, required=True)
    ap.add_argument("--out-html", type=Path, required=True)
    ap.add_argument("--out-svg", type=Path, required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--usd-per-model-unit", type=float)
    ap.add_argument("--ymin-model", type=float, default=0.3)
    ap.add_argument("--ymax-model", type=float, default=1.1)
    args = ap.parse_args()
    if args.ymax_model <= args.ymin_model:
        ap.error("--ymax-model must be greater than --ymin-model")
    params = _params_from_result(args.result)
    rows = average_price_path(args.policy, params)
    svg = render_svg(
        rows,
        params,
        usd_per_model_unit=args.usd_per_model_unit,
        ymin_model=args.ymin_model,
        ymax_model=args.ymax_model,
    )
    write_outputs(svg, args.out_html, args.out_svg)
    print(json.dumps({"html": str(args.out_html), "svg": str(args.out_svg)}, indent=2))


if __name__ == "__main__":
    main()
