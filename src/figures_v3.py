"""
Paper D v3 Publication Figures
==============================

Seven figures for the surface-to-TOA transfer paper.
Uses cartopy for maps, seaborn for statistical plots.
All figures at 300 DPI, dual format (PNG + PDF).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

BIOME_COLORS = {
    "Forest": "#1b7837", "Shrubland": "#9970ab", "Savanna": "#e7d4e8",
    "Grassland": "#dfc27d", "Wetland": "#35978f", "Cropland": "#bf812d",
    "Barren": "#f4a582", "Other": "#999999",
}
BIOME_MARKERS = {
    "Forest": "o", "Shrubland": "s", "Savanna": "D",
    "Grassland": "^", "Wetland": "v", "Cropland": "P",
    "Barren": "*", "Other": ".",
}


def setup():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 10, "axes.labelsize": 11,
        "axes.titlesize": 12, "figure.dpi": 150, "savefig.dpi": 300,
        "savefig.bbox": "tight", "axes.spines.top": False, "axes.spines.right": False,
    })


def _save(fig, path, name):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(path / f"{name}.{ext}")
    plt.close(fig)
    print(f"  Saved {name}")


def fig1_site_map(df, output_dir):
    """Figure 1: Global site map with alpha coloring on cartopy."""
    setup()
    if not HAS_CARTOPY:
        print("  Skipping fig1 (no cartopy)")
        return

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", edgecolor="none")
    ax.add_feature(cfeature.OCEAN, facecolor="#e8f4f8")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, color="#888888")
    ax.add_feature(cfeature.BORDERS, linewidth=0.2, color="#cccccc")

    mask = df["forcing_proxy"].notna() & df["latitude"].notna()
    clean = df[mask]

    sc = ax.scatter(
        clean["longitude"], clean["latitude"],
        c=clean["forcing_proxy"], cmap="RdYlGn_r",
        vmin=0, vmax=2.5, s=25, edgecolors="black", linewidth=0.3,
        alpha=0.8, transform=ccrs.PlateCarree(), zorder=5,
    )

    cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.02, aspect=30)
    cbar.set_label(r"Surface forcing proxy $\alpha$ (H/R$_{net}$)", fontsize=10)

    ax.set_title(
        f"313 FLUXNET Sites Co-located with CERES EBAF\n"
        f"7 biome classes, 4 continents, $\\alpha$ range: 0.03 - 2.53",
        fontsize=11, fontweight="bold",
    )

    # Add biome legend
    handles = [mpatches.Patch(color=BIOME_COLORS[b], label=f"{b} (n={len(clean[clean['biome_group']==b])})")
               for b in ["Forest", "Grassland", "Cropland", "Shrubland", "Wetland", "Savanna"]]
    ax.legend(handles=handles, loc="lower left", fontsize=7, framealpha=0.9, ncol=2)

    _save(fig, output_dir, "fig1_site_map")


def fig2_alpha_vs_cre(df, output_dir):
    """Figure 2: Alpha vs CRE scatter with regression and CI."""
    setup()
    fig, ax = plt.subplots(figsize=(8, 6))

    alpha_col = "forcing_proxy"
    cre_col = "ceres_toa_cre_net_mean"
    mask = df[alpha_col].notna() & df[cre_col].notna()
    clean = df[mask]

    for biome in ["Forest", "Grassland", "Cropland", "Shrubland", "Wetland", "Savanna", "Barren"]:
        g = clean[clean["biome_group"] == biome]
        if len(g) == 0:
            continue
        ax.scatter(
            g[alpha_col], g[cre_col],
            c=BIOME_COLORS.get(biome, "#999"), marker=BIOME_MARKERS.get(biome, "o"),
            label=f"{biome} (n={len(g)})", s=35, alpha=0.7,
            edgecolors="white", linewidth=0.3, zorder=3,
        )

    # Regression
    x = clean[alpha_col].values
    y = clean[cre_col].values
    slope, intercept, r, p, se = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, "k-", linewidth=2,
            label=f"$\\gamma$ = {slope:.2f} W/m$^2$/unit $\\alpha$", zorder=4)

    # 95% CI
    n = len(x)
    x_mean = np.mean(x)
    se_line = np.sqrt(np.sum((y - slope*x - intercept)**2)/(n-2)) * \
              np.sqrt(1/n + (x_line - x_mean)**2 / np.sum((x - x_mean)**2))
    t_val = stats.t.ppf(0.975, n - 2)
    ax.fill_between(x_line,
                     slope * x_line + intercept - t_val * se_line,
                     slope * x_line + intercept + t_val * se_line,
                     alpha=0.12, color="grey", zorder=2)

    ax.set_xlabel(r"Surface forcing proxy $\alpha$ (H/R$_{net}$)")
    ax.set_ylabel("CERES Cloud Radiative Effect (W/m$^2$)")
    ax.set_title(
        f"Surface Energy Partitioning vs. TOA Cloud Radiative Effect\n"
        f"$\\gamma$ = {slope:.2f} $\\pm$ {se:.2f}, R$^2$ = {r**2:.3f}, "
        f"p = {p:.1e}, n = {n}",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle="--")

    _save(fig, output_dir, "fig2_alpha_vs_cre")


def fig3_seasonal(df, output_dir):
    """Figure 3: JJA vs DJF comparison (side by side)."""
    setup()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    alpha_col = "forcing_proxy"

    for ax, season, col, title, color in [
        (ax1, "JJA", "cre_jja", "Summer (JJA)", "#d73027"),
        (ax2, "DJF", "cre_djf", "Winter (DJF)", "#4575b4"),
    ]:
        if col not in df.columns:
            ax.text(0.5, 0.5, f"No {season} data", ha="center", va="center", transform=ax.transAxes)
            continue

        mask = df[alpha_col].notna() & df[col].notna()
        clean = df[mask]
        x, y = clean[alpha_col].values, clean[col].values

        if len(x) < 10:
            ax.text(0.5, 0.5, f"n = {len(x)} (insufficient)", ha="center", va="center", transform=ax.transAxes)
            continue

        for biome in ["Forest", "Grassland", "Cropland", "Shrubland", "Wetland", "Savanna"]:
            g = clean[clean["biome_group"] == biome]
            if len(g) == 0:
                continue
            ax.scatter(g[alpha_col], g[col], c=BIOME_COLORS.get(biome),
                       marker=BIOME_MARKERS.get(biome, "o"), s=30, alpha=0.6,
                       edgecolors="white", linewidth=0.2)

        slope, intercept, r, p, se = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, slope * x_line + intercept, color=color, linewidth=2.5)

        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        ax.set_title(f"{title}\n$\\gamma$ = {slope:.1f}, R$^2$ = {r**2:.3f}, p = {p:.1e} ({sig})",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel(r"$\alpha$ (H/R$_{net}$)")
        ax.grid(True, alpha=0.2, linestyle="--")

    ax1.set_ylabel("CERES CRE$_{net}$ (W/m$^2$)")

    fig.suptitle("Seasonal Decomposition: Signal Concentrates in Convective Season",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, output_dir, "fig3_seasonal")


def fig4_biome_gammas(df, output_dir):
    """Figure 4: Biome-level gamma with CI."""
    setup()
    alpha_col = "forcing_proxy"
    cre_col = "ceres_toa_cre_net_mean"

    records = []
    for biome, g in df.groupby("biome_group"):
        mask = g[alpha_col].notna() & g[cre_col].notna()
        if mask.sum() < 5:
            continue
        x, y = g.loc[mask, alpha_col].values, g.loc[mask, cre_col].values
        slope, intercept, r, p, se = stats.linregress(x, y)
        records.append({"biome": biome, "gamma": slope, "se": se, "r2": r**2, "p": p, "n": mask.sum()})

    bdf = pd.DataFrame(records).sort_values("gamma")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [BIOME_COLORS.get(b, "#999") for b in bdf["biome"]]

    bars = ax.barh(bdf["biome"], bdf["gamma"], xerr=bdf["se"] * 1.96,
                   color=colors, edgecolor="white", linewidth=0.5,
                   capsize=3, error_kw={"linewidth": 1}, height=0.6)

    ax.axvline(0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)

    for _, row in bdf.iterrows():
        sig = "***" if row["p"] < 0.001 else "**" if row["p"] < 0.01 else "*" if row["p"] < 0.05 else "n.s."
        ax.text(max(row["gamma"] + row["se"] * 2 + 0.5, 1),
                row["biome"], f"n={row['n']:.0f}, R$^2$={row['r2']:.2f} {sig}",
                va="center", fontsize=8, color="#555")

    ax.set_xlabel(r"Transfer coefficient $\gamma$ (W/m$^2$ per unit $\alpha$)")
    ax.set_title("Surface-to-TOA Transfer by Biome (95% CI)", fontweight="bold")
    ax.grid(True, axis="x", alpha=0.2, linestyle="--")

    _save(fig, output_dir, "fig4_biome_gammas")


def fig5_causal_chain(df, output_dir):
    """Figure 5: ERA5 mediation pathway diagram."""
    setup()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Boxes
    boxes = {
        "alpha":  (1, 3.5, r"Surface $\alpha$" + "\n(H/R$_{net}$)"),
        "precip": (5, 5.5, "Precipitation\n(ERA5)"),
        "blh":    (5, 1.5, "Boundary Layer\nHeight (ERA5)"),
        "cre":    (9, 3.5, "TOA Cloud\nRadiative Effect\n(CERES)"),
    }

    for key, (x, y, label) in boxes.items():
        color = "#1b7837" if key == "alpha" else "#35978f" if key == "cre" else "#dfc27d"
        rect = mpatches.FancyBboxPatch((x - 0.9, y - 0.6), 1.8, 1.2,
                                        boxstyle="round,pad=0.1",
                                        facecolor=color, edgecolor="black",
                                        linewidth=1.5, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x, y, label, ha="center", va="center", fontsize=9, fontweight="bold",
                color="white" if key in ["alpha", "cre"] else "black",
                path_effects=[pe.withStroke(linewidth=2, foreground="black" if key in ["alpha","cre"] else "white")])

    # Arrows with mediation percentages
    arrow_props = dict(arrowstyle="-|>", color="black", lw=2)

    # alpha -> precip (R2=0.179)
    ax.annotate("", xy=(4.1, 5.2), xytext=(1.9, 4.0), arrowprops=arrow_props)
    ax.text(2.8, 4.9, "R$^2$=0.18\np=6.8e-15", fontsize=8, ha="center",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    # alpha -> blh (R2=0.165)
    ax.annotate("", xy=(4.1, 1.8), xytext=(1.9, 3.0), arrowprops=arrow_props)
    ax.text(2.8, 2.1, "R$^2$=0.16\np=9.0e-14", fontsize=8, ha="center",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    # precip -> cre
    ax.annotate("", xy=(8.1, 4.2), xytext=(5.9, 5.2), arrowprops=arrow_props)
    ax.text(7.2, 5.0, "mediates\n54%", fontsize=9, ha="center", fontweight="bold",
            color="#d73027", bbox=dict(boxstyle="round", facecolor="#fff5f0", alpha=0.9))

    # blh -> cre
    ax.annotate("", xy=(8.1, 2.8), xytext=(5.9, 1.8), arrowprops=arrow_props)
    ax.text(7.2, 1.7, "mediates\n30%", fontsize=9, ha="center", fontweight="bold",
            color="#d73027", bbox=dict(boxstyle="round", facecolor="#fff5f0", alpha=0.9))

    # Direct arrow alpha -> cre
    ax.annotate("", xy=(8.1, 3.5), xytext=(1.9, 3.5),
                arrowprops=dict(arrowstyle="-|>", color="#888888", lw=1.5, linestyle="--"))
    ax.text(5, 3.5, "Direct: R$^2$=0.13, p=3.7e-11\nAfter ALL controls: p=8.4e-5",
            fontsize=8, ha="center", va="center",
            bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.9))

    ax.set_title("ERA5 Mediation Analysis: Physical Pathway from Surface to TOA",
                 fontsize=12, fontweight="bold", pad=20)

    _save(fig, output_dir, "fig5_causal_chain")


def fig6_financial(df, output_dir):
    """Figure 6: Financial sensitivity curve."""
    setup()
    fig, ax = plt.subplots(figsize=(8, 5))

    gamma_frac = np.linspace(0, 0.5, 200)
    surface_diff = 34.2  # observed delta_alpha * R_net
    dollars = gamma_frac * surface_diff * 11.76

    ax.plot(gamma_frac, dollars, "k-", linewidth=2)

    # Mark empirical estimates
    ax.axvline(0.085, color="#1b7837", linewidth=1.5, label="CRE pathway ($\\Gamma$ = 0.085)")
    ax.axvline(0.206, color="#35978f", linewidth=1.5, label="Combined CRE+LW ($\\Gamma$ = 0.206)")
    ax.axvspan(0.06, 0.11, alpha=0.12, color="#1b7837")
    ax.axvspan(0.17, 0.24, alpha=0.12, color="#35978f")

    # Reference lines
    ax.axhline(59.4, color="#4575b4", linewidth=0.8, linestyle=":",
               label="Baker et al. rainfall floor ($59/ha/yr)")
    ax.axhspan(5, 50, alpha=0.06, color="orange", label="Carbon market ($5-50/ha/yr)")

    ax.set_xlabel("Transfer fraction $\\Gamma$ (TOA / surface)")
    ax.set_ylabel("Social benefit of forest cooling ($/ha/year)")
    ax.set_title("Financial Sensitivity to Surface-TOA Transfer\n(using observed $\\alpha$ range: Forest 0.86, Shrubland 1.21)",
                 fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 250)

    _save(fig, output_dir, "fig6_financial")


def fig7_cre_map(df, output_dir):
    """Figure 7: Spatial pattern of CRE on cartopy with site overlay."""
    setup()
    if not HAS_CARTOPY:
        print("  Skipping fig7 (no cartopy)")
        return

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
    ax.set_global()
    ax.add_feature(cfeature.LAND, facecolor="#f5f5f5", edgecolor="none")
    ax.add_feature(cfeature.OCEAN, facecolor="#e8f4f8")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, color="#888888")

    mask = df["ceres_toa_cre_net_mean"].notna() & df["latitude"].notna()
    clean = df[mask]

    sc = ax.scatter(
        clean["longitude"], clean["latitude"],
        c=clean["ceres_toa_cre_net_mean"], cmap="RdBu_r",
        vmin=-45, vmax=15, s=30, edgecolors="black", linewidth=0.3,
        alpha=0.85, transform=ccrs.PlateCarree(), zorder=5,
    )

    cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.02, aspect=30)
    cbar.set_label("CERES CRE$_{net}$ (W/m$^2$)", fontsize=10)

    ax.set_title(
        "Cloud Radiative Effect at FLUXNET Sites\n"
        "Negative = clouds cool (forests), Positive = weak cloud cooling (arid)",
        fontsize=11, fontweight="bold",
    )

    _save(fig, output_dir, "fig7_cre_map")


def generate_all_figures(df, output_dir="outputs/v3_figures"):
    """Generate all seven figures."""
    print("Generating Paper D v3 figures...")
    fig1_site_map(df, output_dir)
    fig2_alpha_vs_cre(df, output_dir)
    fig3_seasonal(df, output_dir)
    fig4_biome_gammas(df, output_dir)
    fig5_causal_chain(df, output_dir)
    fig6_financial(df, output_dir)
    fig7_cre_map(df, output_dir)
    print("Done.")


if __name__ == "__main__":
    df = pd.read_csv("outputs/toa_tables/site_summary_v3_full.csv")
    generate_all_figures(df)
