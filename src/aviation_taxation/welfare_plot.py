"""Plot the four welfare objectives for the kappa=0.28 ticket-tax run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DATA = Path(__file__).resolve().parents[2] / "data/archived"
DEFAULT_INPUT = DATA / "frequency_sweep.csv"
DEFAULT_SUMMARY = DATA / "welfare_baseline.json"
DEFAULT_ENDPOINTS = DATA / "interpolation_endpoints.json"
DEFAULT_FITS = DATA / "carbon_instrument_fit.json"
DEFAULT_OUTPUT = Path("outputs/welfare_vs_frequency.svg")

def load_data(
    path: Path,
    summary: dict,
) -> tuple[np.ndarray, float, float, list[tuple[str, np.ndarray, str]]]:
    with path.open(newline="") as handle:
        rows = sorted(
            (row for row in csv.DictReader(handle) if float(row["M"]) >= 2.3),
            key=lambda row: float(row["M"]),
        )
    if len(rows) != 15:
        raise ValueError(f"Expected 15 calculated frequency points, found {len(rows)}")

    m = np.array([float(row["M"]) for row in rows])
    baseline = summary["baseline"]
    conversion = float(baseline["usd_per_unit_busd"])
    emissions_mt = 900.0
    conventional = np.array([float(row["calendar_welfare_after_fixed"]) for row in rows])
    fixed_cost = float(baseline["K"])
    avoided_frequency = 1.0 - m / 3.0

    wedge = conventional + 0.28 * fixed_cost * avoided_frequency
    wedge_scc210 = wedge + 210.0 * emissions_mt / (1000.0 * conversion) * avoided_frequency
    wedge_scc336 = wedge + 336.0 * emissions_mt / (1000.0 * conversion) * avoided_frequency
    panels = [
        ("Conventional welfare", conventional * conversion, "#dc2626"),
        ("Conventional welfare + wedge", wedge * conversion, "#7c3aed"),
        ("Conventional welfare + wedge + SCC (USD 210/tCO$_2$)", wedge_scc210 * conversion, "#059669"),
        ("Conventional welfare + wedge + SCC (USD 336/tCO$_2$)", wedge_scc336 * conversion, "#ea580c"),
    ]
    status_welfare = float(baseline["baseline_calW"]) * conversion
    return m, conversion, status_welfare, panels


def quadratic_peak(m: np.ndarray, welfare: np.ndarray) -> tuple[np.ndarray, float, float]:
    coefficients = np.polyfit(m, welfare, 2)
    if coefficients[0] >= 0:
        raise ValueError("Quadratic fit does not have a maximum")
    peak_m = float(-coefficients[1] / (2.0 * coefficients[0]))
    return coefficients, peak_m, float(np.polyval(coefficients, peak_m))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--endpoints", type=Path, default=DEFAULT_ENDPOINTS)
    parser.add_argument("--fits", type=Path, default=DEFAULT_FITS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with args.summary.open() as handle:
        summary = json.load(handle)
    with args.endpoints.open() as handle:
        endpoints = json.load(handle)
    with args.fits.open() as handle:
        saved_fits = json.load(handle)
    m, conversion, status_welfare, panels = load_data(args.input, summary)
    tau_fit = saved_fits["implied_tau_c_model_per_flight"]
    tau_degree = str(tau_fit["selected_degree"])
    tau_coefficients = tau_fit["candidates"][tau_degree]["coefficients"]
    passenger_neutral_m = float(endpoints["passenger_neutral"]["M"])
    free_entry_m = float(endpoints["free_entry"]["M"])
    fits = [quadratic_peak(m, welfare) for _, welfare, _ in panels]
    maxima = []
    for (title, _, _), (_, peak_m, peak_welfare) in zip(panels, fits):
        maxima.append({
            "objective": title, "M": peak_m, "welfare_busd": peak_welfare,
            "carbon_tax_usd_per_tonne": float(np.polyval(tau_coefficients, peak_m)) * conversion / 0.9,
        })

    extrema = [status_welfare]
    extrema.extend(value for _, welfare, _ in panels for value in welfare)
    extrema.extend(peak_welfare for _, _, peak_welfare in fits)
    span = max(extrema) - min(extrema)
    lower = min(extrema) - 0.09 * span
    upper = max(extrema) + 0.08 * span
    label_y = lower + 0.025 * (upper - lower)
    box_y = lower + 0.11 * (upper - lower)

    fig, axes = plt.subplots(4, 1, figsize=(11.2, 12.4), sharex=True)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.105, right=0.895, top=0.975, bottom=0.075, hspace=0.22)
    calculated_x = np.linspace(m.min(), 3.0, 300)
    extrapolated_x = np.linspace(3.0, 3.75, 300)

    for axis, (panel_title, welfare, color), fit in zip(axes, panels, fits):
        coefficients, peak_m, peak_welfare = fit
        peak_tau = float(np.polyval(tau_coefficients, peak_m))
        peak_carbon_price = peak_tau * conversion / 0.9
        instrument = "tax" if peak_carbon_price >= 0 else "subsidy"

        axis.set_facecolor("white")
        axis.axhline(status_welfare, color="#9ca3af", lw=0.9, ls=(0, (1.5, 2.5)), zorder=0)
        axis.text(0.008, status_welfare + 0.012 * (upper - lower), "status quo welfare",
                  color="#9ca3af", fontsize=7.5, va="bottom", transform=axis.get_yaxis_transform())
        axis.plot(calculated_x, np.polyval(coefficients, calculated_x), color=color, lw=1.8)
        axis.plot(extrapolated_x, np.polyval(coefficients, extrapolated_x), color=color,
                  lw=1.8, ls=(0, (2.5, 2.5)))
        axis.scatter(m, welfare, color=color, edgecolor="white", linewidth=0.55, s=28, zorder=3)

        for position in (3.0, free_entry_m, passenger_neutral_m):
            axis.axvline(position, color="#9ca3af", lw=1.0, ls=(0, (1.5, 2.5)))
        axis.text(3.005, label_y, "M=3", color="#4b5563", fontsize=8)
        axis.text(free_entry_m + 0.006, label_y, "free entry", color="#6b7280", fontsize=7.5)
        axis.text(passenger_neutral_m - 0.006, label_y, "passenger-neutral",
                  color="#6b7280", fontsize=7.5, ha="right")

        axis.axvline(peak_m, color="#7c3aed", lw=1.1, ls=(0, (3, 3)))
        axis.scatter([peak_m], [peak_welfare], color="#7c3aed", edgecolor="white",
                     linewidth=0.6, s=52, zorder=4)
        axis.text(
            0.992,
            box_y,
            f"$M^*={peak_m:.3f}$, W={peak_welfare:,.1f} bn USD\n"
            f"carbon {instrument}: {abs(peak_carbon_price):,.1f} USD/tCO$_2$",
            fontsize=8.5,
            linespacing=1.35,
            ha="right",
            transform=axis.get_yaxis_transform(),
            bbox={"facecolor": "white", "edgecolor": "#d1d5db", "boxstyle": "square,pad=0.65"},
        )
        axis.text(2.265, upper - 0.035 * (upper - lower), panel_title,
                  fontsize=10, fontweight="bold", va="top")
        axis.set_xlim(2.25, 3.75)
        axis.set_ylim(lower, upper)
        axis.grid(True, color="#e5e7eb", linewidth=0.8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.spines[["left", "bottom"]].set_color("#4b5563")
        axis.tick_params(labelsize=8, color="#4b5563")

    axes[-1].set_xticks(np.arange(2.3, 3.71, 0.1))
    axes[-1].set_xlabel("M", fontsize=11, fontweight="bold")
    fig.text(0.025, 0.5, "Welfare (bn USD equivalent)", rotation=90,
             va="center", fontsize=11, fontweight="bold")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".maxima.json").write_text(json.dumps(maxima, indent=2) + "\n")
    for suffix in ("svg", "pdf", "png"):
        fig.savefig(args.output.with_suffix("." + suffix), format=suffix, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
