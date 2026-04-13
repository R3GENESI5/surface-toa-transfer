"""
Paper D — Visualization Module
===============================

Publication-quality figures for the surface-to-TOA transfer paper.
All figures are saved at 300 DPI in both PNG and PDF formats.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)

BIOME_COLORS = {
    "Forest": "#1b7837",
    "Shrubland": "#9970ab",
    "Savanna": "#e7d4e8",
    "Grassland": "#dfc27d",
    "Wetland": "#35978f",
    "Cropland": "#bf812d",
    "Barren": "#f4a582",
    "Other": "#999999",
}

BIOME_MARKERS = {
    "Forest": "o", "Shrubland": "s", "Savanna": "D",
    "Grassland": "^", "Wetland": "v", "Cropland": "P",
    "Barren": "*", "Other": ".",
}


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def _save(fig, path, name):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(path / f"{name}.{ext}")
    plt.close(fig)
    logger.info(f"  Saved {name}")


def plot_alpha_vs_cre(
    df: pd.DataFrame,
    transfer_result,
    output_dir: str | Path,
):
    """
    Figure 1: alpha(beta) vs CERES Cloud Radiative Effect.
    The key figure of Paper D.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    alpha_col = "forcing_proxy"
    cre_col = "ceres_toa_cre_net_mean"

    mask = df[alpha_col].notna() & df[cre_col].notna()
    clean = df[mask]

    for biome, group in clean.groupby("biome_group"):
        ax.scatter(
            group[alpha_col], group[cre_col],
            c=BIOME_COLORS.get(biome, "#999"),
            marker=BIOME_MARKERS.get(biome, "o"),
            label=f"{biome} (n={len(group)})",
            s=40, alpha=0.7, edgecolors="white", linewidth=0.3,
        )

    # Regression line
    x = clean[alpha_col].values
    y = clean[cre_col].values
    slope, intercept, r, p, se = sp_stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, "k-", linewidth=2, label=f"$\\gamma$ = {slope:.2f}")

    # 95% CI band
    n = len(x)
    x_mean = np.mean(x)
    se_line = se * np.sqrt(1/n + (x_line - x_mean)**2 / np.sum((x - x_mean)**2))
    t_val = sp_stats.t.ppf(0.975, n - 2)
    ax.fill_between(
        x_line,
        slope * x_line + intercept - t_val * se_line,
        slope * x_line + intercept + t_val * se_line,
        alpha=0.15, color="grey",
    )

    ax.set_xlabel("Surface forcing coefficient $\\alpha(\\beta)$")
    ax.set_ylabel("CERES Cloud Radiative Effect (W/m$^2$)")
    ax.set_title(
        f"Surface Energy Partitioning vs. TOA Cloud Effect\n"
        f"$\\gamma$ = {slope:.3f} $\\pm$ {se:.3f}, "
        f"R$^2$ = {r**2:.3f}, p = {p:.1e}, n = {n}",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    _save(fig, output_dir, "fig1_alpha_vs_cre")


def plot_biome_gammas(
    biome_df: pd.DataFrame,
    output_dir: str | Path,
):
    """
    Figure 2: Transfer coefficient gamma by biome with CI.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    biome_df = biome_df.sort_values("gamma")
    colors = [BIOME_COLORS.get(b, "#999") for b in biome_df["biome"]]

    bars = ax.barh(
        biome_df["biome"], biome_df["gamma"],
        xerr=biome_df["gamma_se"] * 1.96,
        color=colors, edgecolor="white", linewidth=0.5,
        capsize=3, error_kw={"linewidth": 1},
    )

    ax.axvline(0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Transfer coefficient $\\gamma$")
    ax.set_title("Surface-to-TOA Transfer by Biome (95% CI)")

    for i, row in biome_df.iterrows():
        ax.text(
            row["gamma"] + row["gamma_se"] * 2 + 0.5,
            row["biome"],
            f"n={row['n_sites']:.0f}",
            va="center", fontsize=8, color="#666",
        )

    ax.grid(True, axis="x", alpha=0.3)
    _save(fig, output_dir, "fig2_biome_gammas")


def plot_taylor_diagram(
    df: pd.DataFrame,
    output_dir: str | Path,
):
    """
    Figure 3: Taylor diagram comparing observed (FLUXNET) vs
    CERES-implied energy partition across biomes.
    """
    setup_style()

    # Compute biome-level stats for Taylor diagram
    biome_stats = []
    for biome, group in df.groupby("biome_group"):
        mask = group["forcing_proxy"].notna() & group["ceres_toa_net_mean"].notna()
        clean = group[mask]
        if len(clean) < 3:
            continue

        obs = clean["forcing_proxy"].values
        # Normalize TOA net as a proxy for comparison
        toa = clean["ceres_toa_net_mean"].values
        if np.std(obs) > 0 and np.std(toa) > 0:
            r_corr = np.corrcoef(obs, toa)[0, 1]
            std_ratio = np.std(toa) / np.std(obs)
            biome_stats.append({
                "biome": biome,
                "correlation": r_corr,
                "std_ratio": std_ratio,
                "n": len(clean),
            })

    if not biome_stats:
        logger.warning("Insufficient data for Taylor diagram")
        return

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"projection": "polar"})

    for bs in biome_stats:
        theta = np.arccos(bs["correlation"])
        r = bs["std_ratio"]
        color = BIOME_COLORS.get(bs["biome"], "#999")
        ax.scatter(theta, r, c=color, s=80, zorder=5,
                   label=f"{bs['biome']} (n={bs['n']})")
        ax.annotate(bs["biome"][:3], (theta, r),
                    fontsize=7, ha="left", va="bottom")

    # Reference point (perfect match)
    ax.scatter(0, 1, c="red", marker="*", s=200, zorder=10, label="Reference")

    ax.set_thetamax(90)
    ax.set_title("Taylor Diagram: Surface vs. TOA Energy Partition", pad=20)
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(1.3, 1.0))

    _save(fig, output_dir, "fig3_taylor_diagram")


def plot_financial_sensitivity(
    gamma_range: np.ndarray,
    dollars_per_ha: np.ndarray,
    gamma_estimate: float,
    gamma_ci: tuple[float, float],
    output_dir: str | Path,
):
    """
    Figure 4: $/ha/year as a function of gamma.
    Shows how the financial valuation scales with the transfer coefficient.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(gamma_range, dollars_per_ha, "k-", linewidth=2)

    # Highlight gamma estimate and CI
    ax.axvline(gamma_estimate, color="#1b7837", linewidth=1.5, linestyle="-",
               label=f"$\\gamma$ estimate = {gamma_estimate:.3f}")
    ax.axvspan(gamma_ci[0], gamma_ci[1], alpha=0.15, color="#1b7837",
               label=f"95% CI [{gamma_ci[0]:.3f}, {gamma_ci[1]:.3f}]")

    # Reference lines
    ax.axhline(0, color="grey", linewidth=0.5)

    # Current carbon market range
    ax.axhspan(5, 50, alpha=0.08, color="orange",
               label="Carbon market range ($5-50/ha/yr)")

    # Baker et al. rainfall floor
    ax.axhline(59.4, color="blue", linewidth=0.8, linestyle=":", alpha=0.5,
               label="Baker et al. rainfall floor ($59/ha/yr)")

    ax.set_xlabel("Transfer fraction $\\gamma$ (TOA / surface)")
    ax.set_ylabel("Social benefit of forest cooling ($/ha/year)")
    ax.set_title("Financial Sensitivity to Surface-TOA Transfer")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(gamma_range))

    _save(fig, output_dir, "fig4_financial_sensitivity")


def plot_gamma_map(
    df: pd.DataFrame,
    output_dir: str | Path,
):
    """
    Figure 5: Map of sites colored by local gamma contribution.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    alpha_col = "forcing_proxy"
    cre_col = "ceres_toa_cre_net_mean"

    mask = (
        df[alpha_col].notna() & df[cre_col].notna() &
        df["latitude"].notna() & df["longitude"].notna()
    )
    clean = df[mask]

    if len(clean) == 0:
        logger.warning("No data for gamma map")
        return

    # Local gamma proxy: CRE / alpha ratio per site
    clean = clean.copy()
    clean["local_gamma"] = clean[cre_col] / clean[alpha_col]

    sc = ax.scatter(
        clean["longitude"], clean["latitude"],
        c=clean["local_gamma"],
        cmap="RdYlGn",
        s=50, edgecolors="black", linewidth=0.3,
        alpha=0.8,
    )

    cbar = plt.colorbar(sc, ax=ax, shrink=0.7)
    cbar.set_label("Local CRE / $\\alpha(\\beta)$ ratio")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Spatial Pattern of Surface-to-TOA Transfer")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 80)
    ax.grid(True, alpha=0.2)

    # Simple continent outlines would need cartopy/basemap
    # For now, keep it clean without coastlines

    _save(fig, output_dir, "fig5_gamma_map")
