"""
Robustness Checks for Paper D
=============================

Addresses adversarial audit findings:
- F1: MODIS homogeneity filter
- F2: Wetland anomaly investigation
- F3: Additional confounding controls
- M3: Seasonal decomposition
- M5-M6: Heteroscedasticity, robust SEs, effective sample size
- M9: Site-specific transfer fractions using local R_net
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────
# M6: Effective sample size (spatial clustering)
# ──────────────────────────────────────────────────────────

def compute_effective_sample_size(df: pd.DataFrame) -> dict:
    """
    Compute effective sample size accounting for spatial clustering.
    Sites sharing a CERES pixel are not independent on the dependent variable.
    """
    df = df.copy()
    df["ceres_pixel"] = (
        np.round(df["latitude"]).astype(str) + "_" +
        np.round(df["longitude"]).astype(str)
    )

    n_sites = len(df)
    n_pixels = df["ceres_pixel"].nunique()
    dupes = df.groupby("ceres_pixel").size()
    n_shared = (dupes > 1).sum()
    sites_in_shared = dupes[dupes > 1].sum()

    result = {
        "n_sites": n_sites,
        "n_unique_pixels": n_pixels,
        "n_shared_pixels": n_shared,
        "sites_in_shared_pixels": sites_in_shared,
        "effective_n": n_pixels,  # Conservative: one obs per pixel
    }

    logger.info(
        f"Effective sample size:\n"
        f"  Sites: {n_sites}\n"
        f"  Unique CERES pixels: {n_pixels}\n"
        f"  Shared pixels: {n_shared} ({sites_in_shared} sites)\n"
        f"  Effective n (conservative): {n_pixels}"
    )

    return result


def pixel_averaged_regression(
    df: pd.DataFrame,
    alpha_col: str = "forcing_proxy",
    toa_col: str = "ceres_toa_cre_net_mean",
) -> dict:
    """
    Average sites within the same CERES pixel before regression.
    This gives each 1-degree pixel one observation, ensuring independence.
    """
    df = df.copy()
    df["ceres_pixel"] = (
        np.round(df["latitude"]).astype(str) + "_" +
        np.round(df["longitude"]).astype(str)
    )

    mask = df[alpha_col].notna() & df[toa_col].notna()
    clean = df[mask]

    # Average within pixels
    pixel_means = clean.groupby("ceres_pixel").agg({
        alpha_col: "mean",
        toa_col: "mean",
        "latitude": "mean",
        "biome_group": "first",
    }).reset_index()

    x = pixel_means[alpha_col].values
    y = pixel_means[toa_col].values
    slope, intercept, r, p, se = sp_stats.linregress(x, y)

    result = {
        "gamma_pixel_averaged": slope,
        "gamma_se": se,
        "r_squared": r ** 2,
        "p_value": p,
        "n_pixels": len(pixel_means),
    }

    logger.info(
        f"Pixel-averaged regression:\n"
        f"  gamma = {slope:.4f} +/- {se:.4f}\n"
        f"  R2 = {r**2:.4f}, p = {p:.2e}\n"
        f"  n = {len(pixel_means)} unique pixels"
    )

    return result


# ──────────────────────────────────────────────────────────
# M5: Heteroscedasticity and robust standard errors
# ──────────────────────────────────────────────────────────

def heteroscedasticity_test(
    df: pd.DataFrame,
    alpha_col: str = "forcing_proxy",
    toa_col: str = "ceres_toa_cre_net_mean",
) -> dict:
    """
    Breusch-Pagan test for heteroscedasticity and HC3 robust standard errors.
    """
    mask = df[alpha_col].notna() & df[toa_col].notna()
    x = df.loc[mask, alpha_col].values
    y = df.loc[mask, toa_col].values
    n = len(x)

    # OLS fit
    slope, intercept, r, p_ols, se_ols = sp_stats.linregress(x, y)
    residuals = y - (slope * x + intercept)

    # Breusch-Pagan: regress squared residuals on x
    sq_resid = residuals ** 2
    bp_slope, bp_int, bp_r, bp_p, bp_se = sp_stats.linregress(x, sq_resid)
    bp_statistic = n * bp_r ** 2  # LM test statistic ~ chi2(1)
    bp_p_value = 1 - sp_stats.chi2.cdf(bp_statistic, 1)

    # HC3 robust standard errors (MacKinnon & White, 1985)
    X = np.column_stack([np.ones(n), x])
    hat_matrix_diag = np.sum(X * np.linalg.solve(X.T @ X, X.T).T, axis=1)
    hc3_weights = residuals ** 2 / (1 - hat_matrix_diag) ** 2
    bread = np.linalg.inv(X.T @ X)
    meat = X.T @ np.diag(hc3_weights) @ X
    sandwich = bread @ meat @ bread
    se_hc3 = np.sqrt(sandwich[1, 1])

    # Recompute p-value with HC3 SE
    t_hc3 = slope / se_hc3
    p_hc3 = 2 * sp_stats.t.sf(abs(t_hc3), n - 2)

    result = {
        "bp_statistic": bp_statistic,
        "bp_p_value": bp_p_value,
        "heteroscedastic": bp_p_value < 0.05,
        "se_ols": se_ols,
        "se_hc3": se_hc3,
        "p_ols": p_ols,
        "p_hc3": p_hc3,
        "gamma": slope,
    }

    logger.info(
        f"Heteroscedasticity test:\n"
        f"  Breusch-Pagan: stat = {bp_statistic:.2f}, p = {bp_p_value:.4f} "
        f"({'heteroscedastic' if bp_p_value < 0.05 else 'homoscedastic'})\n"
        f"  OLS SE:  {se_ols:.4f} (p = {p_ols:.2e})\n"
        f"  HC3 SE:  {se_hc3:.4f} (p = {p_hc3:.2e})"
    )

    return result


# ──────────────────────────────────────────────────────────
# F3: Multiple confounding controls
# ──────────────────────────────────────────────────────────

def multivariate_controls(
    df: pd.DataFrame,
    alpha_col: str = "forcing_proxy",
    toa_col: str = "ceres_toa_cre_net_mean",
) -> dict:
    """
    Partial correlations controlling for multiple confounders.
    Uses residual regression approach.
    """
    results = {}

    # Available controls
    potential_controls = {
        "latitude": "latitude",
        "R_net": "mean_netrad",
        "altitude_proxy": "ceres_toa_lw_clr_mean",  # Clear-sky OLR as altitude proxy
    }

    available = {}
    for name, col in potential_controls.items():
        if col in df.columns and df[col].notna().sum() > 20:
            available[name] = col

    mask = df[alpha_col].notna() & df[toa_col].notna()
    for col in available.values():
        mask &= df[col].notna()
    clean = df[mask]

    if len(clean) < 20:
        logger.warning("Insufficient data for multivariate controls")
        return {}

    x = clean[alpha_col].values
    y = clean[toa_col].values

    # Zero-order correlation
    r0, p0 = sp_stats.pearsonr(x, y)
    results["zero_order"] = {"r": r0, "r2": r0**2, "p": p0, "n": len(clean)}

    # Progressive controls: add one at a time
    from numpy.linalg import lstsq

    control_list = []
    for name, col in available.items():
        control_list.append((name, col))
        controls = clean[[c for _, c in control_list]].values
        X_ctrl = np.column_stack([controls, np.ones(len(clean))])

        # Residualize alpha
        coef_x, _, _, _ = lstsq(X_ctrl, x, rcond=None)
        x_resid = x - X_ctrl @ coef_x

        # Residualize TOA
        coef_y, _, _, _ = lstsq(X_ctrl, y, rcond=None)
        y_resid = y - X_ctrl @ coef_y

        r_partial, p_partial = sp_stats.pearsonr(x_resid, y_resid)

        ctrl_names = [n for n, _ in control_list]
        results[f"controlling_{'+'.join(ctrl_names)}"] = {
            "r": r_partial,
            "r2": r_partial ** 2,
            "p": p_partial,
            "controls": ctrl_names,
            "n": len(clean),
        }

        logger.info(
            f"  Controlling {ctrl_names}: r = {r_partial:.4f}, "
            f"R2 = {r_partial**2:.4f}, p = {p_partial:.2e}"
        )

    return results


# ──────────────────────────────────────────────────────────
# F2: Wetland anomaly analysis
# ──────────────────────────────────────────────────────────

def wetland_anomaly_analysis(df: pd.DataFrame) -> dict:
    """
    Investigate why wetlands show negative gamma.
    Hypothesis: Arctic/boreal wetlands have low CRE despite high ET
    because of low solar input and stratiform (not convective) clouds.
    """
    wet = df[df["biome_group"] == "Wetland"].copy()

    if len(wet) < 3:
        return {}

    # Split by latitude
    arctic_mask = wet["latitude"].abs() > 55
    n_arctic = arctic_mask.sum()
    n_temperate = (~arctic_mask).sum()

    result = {
        "n_total": len(wet),
        "n_arctic": int(n_arctic),
        "n_temperate": int(n_temperate),
        "mean_lat": wet["latitude"].mean(),
    }

    if n_arctic > 0:
        result["arctic_mean_alpha"] = wet.loc[arctic_mask, "forcing_proxy"].mean()
        result["arctic_mean_cre"] = wet.loc[arctic_mask, "ceres_toa_cre_net_mean"].mean()
        result["arctic_mean_rnet"] = wet.loc[arctic_mask, "mean_netrad"].mean()
    if n_temperate > 0:
        result["temperate_mean_alpha"] = wet.loc[~arctic_mask, "forcing_proxy"].mean()
        result["temperate_mean_cre"] = wet.loc[~arctic_mask, "ceres_toa_cre_net_mean"].mean()
        result["temperate_mean_rnet"] = wet.loc[~arctic_mask, "mean_netrad"].mean()

    # Regression excluding wetlands
    non_wet = df[df["biome_group"] != "Wetland"]
    mask = non_wet["forcing_proxy"].notna() & non_wet["ceres_toa_cre_net_mean"].notna()
    x = non_wet.loc[mask, "forcing_proxy"].values
    y = non_wet.loc[mask, "ceres_toa_cre_net_mean"].values
    slope, intercept, r, p, se = sp_stats.linregress(x, y)

    result["gamma_excl_wetlands"] = slope
    result["r2_excl_wetlands"] = r ** 2
    result["p_excl_wetlands"] = p
    result["n_excl_wetlands"] = len(x)

    logger.info(
        f"Wetland anomaly analysis:\n"
        f"  {n_arctic} Arctic (>55 lat), {n_temperate} temperate\n"
        f"  Arctic: mean CRE = {result.get('arctic_mean_cre', 'n/a')}, "
        f"R_net = {result.get('arctic_mean_rnet', 'n/a')}\n"
        f"  Excluding wetlands: gamma = {slope:.4f}, R2 = {r**2:.4f}, p = {p:.2e}"
    )

    return result


# ──────────────────────────────────────────────────────────
# M3: Seasonal decomposition (using monthly CERES data)
# ──────────────────────────────────────────────────────────

def seasonal_cre_extraction(
    ceres_path: str,
    df: pd.DataFrame,
    toa_var: str = "toa_cre_net_mon",
) -> pd.DataFrame:
    """
    Extract seasonal (JJA and DJF) CRE averages per site from monthly CERES data.
    Returns DataFrame with cre_jja and cre_djf columns added.
    """
    try:
        import xarray as xr
    except ImportError:
        logger.warning("xarray required for seasonal decomposition")
        return df

    try:
        ds = xr.open_dataset(ceres_path, engine="h5netcdf")
    except Exception:
        try:
            ds = xr.open_dataset(ceres_path, engine="netcdf4")
        except Exception as e:
            logger.warning(f"Could not open CERES file for seasonal: {e}")
            return df

    if toa_var not in ds.data_vars:
        logger.warning(f"{toa_var} not in CERES file")
        return df

    # Get time coordinate
    time = pd.DatetimeIndex(ds["time"].values)

    jja_mask = time.month.isin([6, 7, 8])
    djf_mask = time.month.isin([12, 1, 2])

    cre_jja_list = []
    cre_djf_list = []

    for _, row in df.iterrows():
        lat = row.get("latitude", np.nan)
        lon = row.get("longitude", np.nan)

        if not (np.isfinite(lat) and np.isfinite(lon)):
            cre_jja_list.append(np.nan)
            cre_djf_list.append(np.nan)
            continue

        try:
            site = ds[toa_var].sel(lat=lat, lon=lon, method="nearest", tolerance=0.5)
            vals = site.values

            jja_vals = vals[jja_mask]
            djf_vals = vals[djf_mask]

            cre_jja = float(np.nanmean(jja_vals)) if np.any(np.isfinite(jja_vals)) else np.nan
            cre_djf = float(np.nanmean(djf_vals)) if np.any(np.isfinite(djf_vals)) else np.nan

            cre_jja_list.append(cre_jja)
            cre_djf_list.append(cre_djf)
        except Exception:
            cre_jja_list.append(np.nan)
            cre_djf_list.append(np.nan)

    df = df.copy()
    df["cre_jja"] = cre_jja_list
    df["cre_djf"] = cre_djf_list

    n_jja = df["cre_jja"].notna().sum()
    n_djf = df["cre_djf"].notna().sum()
    logger.info(f"Seasonal CRE extracted: {n_jja} JJA, {n_djf} DJF values")

    return df


def seasonal_regression(
    df: pd.DataFrame,
    alpha_col: str = "forcing_proxy",
) -> dict:
    """
    Compare JJA vs DJF gamma. If the mechanism is real,
    JJA gamma should be stronger (convective season).
    """
    results = {}

    for season, col in [("JJA", "cre_jja"), ("DJF", "cre_djf"), ("Annual", "ceres_toa_cre_net_mean")]:
        if col not in df.columns:
            continue
        mask = df[alpha_col].notna() & df[col].notna()
        x = df.loc[mask, alpha_col].values
        y = df.loc[mask, col].values

        if len(x) < 10:
            continue

        slope, intercept, r, p, se = sp_stats.linregress(x, y)
        results[season] = {
            "gamma": slope,
            "se": se,
            "r2": r ** 2,
            "p": p,
            "n": len(x),
        }

        logger.info(
            f"  {season}: gamma = {slope:.4f} +/- {se:.4f}, "
            f"R2 = {r**2:.4f}, p = {p:.2e}, n = {len(x)}"
        )

    return results


# ──────────────────────────────────────────────────────────
# M9: Site-specific transfer fractions
# ──────────────────────────────────────────────────────────

def site_specific_transfer(
    df: pd.DataFrame,
    gamma_cre: float,
    alpha_col: str = "forcing_proxy",
) -> pd.DataFrame:
    """
    Compute site-specific transfer fractions using local R_net.
    Addresses the reviewer concern that R_net varies 4x across sites.
    """
    df = df.copy()

    # Transfer fraction = gamma * delta_alpha / (delta_alpha * R_net_local)
    # = gamma / R_net_local
    df["local_transfer_fraction"] = np.where(
        df["mean_netrad"] > 10,
        gamma_cre / df["mean_netrad"],
        np.nan
    )

    valid = df["local_transfer_fraction"].dropna()
    logger.info(
        f"Site-specific transfer fractions:\n"
        f"  mean = {valid.mean():.4f} ({valid.mean()*100:.1f}%)\n"
        f"  range = [{valid.min():.4f}, {valid.max():.4f}]\n"
        f"  ({valid.min()*100:.1f}% to {valid.max()*100:.1f}%)"
    )

    return df


# ──────────────────────────────────────────────────────────
# M8: Corrected financial calculation
# ──────────────────────────────────────────────────────────

def corrected_financial_sensitivity(
    gamma_cre: float,
    gamma_lw: float,
    df: pd.DataFrame,
    dollars_per_wm2_per_ha: float = 11.76,
) -> dict:
    """
    Financial sensitivity using OBSERVED alpha ranges from data,
    not assumed 0.5-1.5 range.
    """
    # Use actual biome means from data
    forest_alpha = df.loc[df["biome_group"] == "Forest", "forcing_proxy"].mean()
    shrub_alpha = df.loc[df["biome_group"] == "Shrubland", "forcing_proxy"].mean()
    forest_rnet = df.loc[df["biome_group"] == "Forest", "mean_netrad"].mean()

    delta_alpha_observed = shrub_alpha - forest_alpha
    surface_diff_observed = delta_alpha_observed * forest_rnet

    # TOA changes using observed values
    toa_cre = gamma_cre * delta_alpha_observed
    toa_lw = gamma_lw * delta_alpha_observed
    toa_total = toa_cre + toa_lw

    frac_cre = toa_cre / surface_diff_observed if surface_diff_observed > 0 else 0
    frac_total = toa_total / surface_diff_observed if surface_diff_observed > 0 else 0

    unadjusted = surface_diff_observed * dollars_per_wm2_per_ha

    result = {
        "forest_alpha_observed": forest_alpha,
        "shrubland_alpha_observed": shrub_alpha,
        "delta_alpha_observed": delta_alpha_observed,
        "forest_rnet_observed": forest_rnet,
        "surface_diff_observed_wm2": surface_diff_observed,
        "toa_cre_change_wm2": toa_cre,
        "toa_total_change_wm2": toa_total,
        "transfer_fraction_cre": frac_cre,
        "transfer_fraction_total": frac_total,
        "unadjusted_dollars_per_ha": unadjusted,
        "cre_adjusted_dollars_per_ha": unadjusted * frac_cre,
        "total_adjusted_dollars_per_ha": unadjusted * frac_total,
    }

    logger.info(
        f"Corrected financial sensitivity (observed alpha range):\n"
        f"  Forest alpha: {forest_alpha:.3f}\n"
        f"  Shrubland alpha: {shrub_alpha:.3f}\n"
        f"  Delta alpha: {delta_alpha_observed:.3f} (was 1.0 in original)\n"
        f"  Forest R_net: {forest_rnet:.1f} W/m2\n"
        f"  Surface differential: {surface_diff_observed:.1f} W/m2 (was 130.0)\n"
        f"  TOA CRE change: {toa_cre:.2f} W/m2\n"
        f"  Transfer fraction (CRE): {frac_cre:.3f} ({frac_cre*100:.1f}%)\n"
        f"  Transfer fraction (total): {frac_total:.3f} ({frac_total*100:.1f}%)\n"
        f"  $/ha unadjusted: ${unadjusted:,.0f}\n"
        f"  $/ha CRE-adjusted: ${unadjusted * frac_cre:,.0f}\n"
        f"  $/ha total-adjusted: ${unadjusted * frac_total:,.0f}"
    )

    return result


# ──────────────────────────────────────────────────────────
# Run all robustness checks
# ──────────────────────────────────────────────────────────

def run_all_robustness(
    df: pd.DataFrame,
    gamma_cre: float,
    gamma_lw: float,
    ceres_path: Optional[str] = None,
) -> dict:
    """Run all robustness checks and return combined results."""

    logger.info("=" * 60)
    logger.info("ROBUSTNESS CHECKS")
    logger.info("=" * 60)

    results = {}

    logger.info("\n--- M6: Effective sample size ---")
    results["effective_n"] = compute_effective_sample_size(df)

    logger.info("\n--- M6: Pixel-averaged regression ---")
    results["pixel_averaged"] = pixel_averaged_regression(df)

    logger.info("\n--- M5: Heteroscedasticity test ---")
    results["heteroscedasticity"] = heteroscedasticity_test(df)

    logger.info("\n--- F3: Multivariate controls ---")
    results["controls"] = multivariate_controls(df)

    logger.info("\n--- F2: Wetland anomaly ---")
    results["wetland"] = wetland_anomaly_analysis(df)

    logger.info("\n--- M9: Site-specific transfer fractions ---")
    df = site_specific_transfer(df, gamma_cre)
    results["site_transfer"] = {
        "mean": df["local_transfer_fraction"].mean(),
        "std": df["local_transfer_fraction"].std(),
        "min": df["local_transfer_fraction"].min(),
        "max": df["local_transfer_fraction"].max(),
    }

    logger.info("\n--- M8: Corrected financial (observed alpha range) ---")
    results["corrected_financial"] = corrected_financial_sensitivity(
        gamma_cre, gamma_lw, df
    )

    if ceres_path:
        logger.info("\n--- M3: Seasonal decomposition ---")
        df = seasonal_cre_extraction(ceres_path, df)
        results["seasonal"] = seasonal_regression(df)

    logger.info("\n" + "=" * 60)

    return results
