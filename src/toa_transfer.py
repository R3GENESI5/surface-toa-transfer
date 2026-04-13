"""
Surface-to-TOA Transfer Coefficient (gamma)
============================================

Tests whether biome-specific surface energy partitioning (alpha(beta))
propagates to top-of-atmosphere radiative signatures measured by CERES.

The transfer coefficient gamma is defined as:

    TOA_anomaly_i = gamma * alpha(beta_i) + intercept + epsilon_i

If gamma > 0, surface forcing differences propagate to TOA.
If gamma ~ 0, surface differences are redistributed locally.

This is the load-bearing test for applying IPCC climate sensitivity
(lambda = 0.8 C per W/m2) to local surface forcing differentials.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


@dataclass
class TransferResult:
    """Result of a single gamma regression."""
    toa_variable: str
    gamma: float
    gamma_se: float
    intercept: float
    r_squared: float
    p_value: float
    n_sites: int
    gamma_ci_low: float = np.nan
    gamma_ci_high: float = np.nan

    def to_dict(self) -> dict:
        return {
            "toa_variable": self.toa_variable,
            "gamma": self.gamma,
            "gamma_se": self.gamma_se,
            "intercept": self.intercept,
            "r_squared": self.r_squared,
            "p_value": self.p_value,
            "n_sites": self.n_sites,
            "gamma_ci_low": self.gamma_ci_low,
            "gamma_ci_high": self.gamma_ci_high,
        }


def compute_transfer_coefficient(
    df: pd.DataFrame,
    alpha_col: str = "forcing_proxy",
    toa_col: str = "ceres_toa_cre_net_mean",
    label: str = "CRE_net",
) -> Optional[TransferResult]:
    """
    Regress a CERES TOA variable against the surface alpha(beta) proxy.

    Parameters
    ----------
    df : DataFrame with alpha_col and toa_col columns
    alpha_col : column name for the surface forcing proxy
    toa_col : column name for the CERES TOA variable
    label : human-readable label for the TOA variable

    Returns
    -------
    TransferResult or None if insufficient data
    """
    mask = df[alpha_col].notna() & df[toa_col].notna()
    clean = df[mask]

    if len(clean) < 10:
        logger.warning(f"Insufficient data for {label}: {len(clean)} sites")
        return None

    x = clean[alpha_col].values
    y = clean[toa_col].values

    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x, y)

    result = TransferResult(
        toa_variable=label,
        gamma=slope,
        gamma_se=std_err,
        intercept=intercept,
        r_squared=r_value ** 2,
        p_value=p_value,
        n_sites=len(clean),
    )

    logger.info(
        f"Transfer coefficient ({label}):\n"
        f"  gamma = {slope:.4f} +/- {std_err:.4f}\n"
        f"  R2    = {r_value**2:.4f}\n"
        f"  p     = {p_value:.2e}\n"
        f"  n     = {len(clean)}"
    )

    return result


def compute_all_transfer_coefficients(df: pd.DataFrame) -> list[TransferResult]:
    """
    Compute gamma for all available CERES TOA variables.

    Tests three pathways:
    - CRE (cloud radiative effect): the cloud pathway
    - TOA LW (outgoing longwave): the thermal pathway
    - TOA net (net radiation): the combined signal
    """
    tests = [
        ("ceres_toa_cre_net_mean", "CRE_net"),
        ("ceres_toa_cre_sw_mean", "CRE_SW"),
        ("ceres_toa_cre_lw_mean", "CRE_LW"),
        ("ceres_toa_lw_up_mean", "TOA_LW_up"),
        ("ceres_toa_net_mean", "TOA_net"),
        ("ceres_toa_net_clr_mean", "TOA_net_clearsky"),
    ]

    results = []
    for toa_col, label in tests:
        if toa_col in df.columns:
            r = compute_transfer_coefficient(df, toa_col=toa_col, label=label)
            if r is not None:
                results.append(r)
        else:
            logger.info(f"Skipping {label}: column {toa_col} not in DataFrame")

    return results


def bootstrap_gamma(
    df: pd.DataFrame,
    alpha_col: str = "forcing_proxy",
    toa_col: str = "ceres_toa_cre_net_mean",
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_seed: int = 42,
) -> tuple[float, float, float]:
    """
    Bootstrap confidence interval for gamma.

    Returns (gamma_median, ci_low, ci_high).
    """
    rng = np.random.RandomState(random_seed)

    mask = df[alpha_col].notna() & df[toa_col].notna()
    clean = df[mask]
    n = len(clean)

    if n < 10:
        return np.nan, np.nan, np.nan

    x = clean[alpha_col].values
    y = clean[toa_col].values

    gammas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        slope, _, _, _, _ = sp_stats.linregress(x[idx], y[idx])
        gammas[i] = slope

    alpha_level = (1 - ci) / 2
    ci_low = np.percentile(gammas, 100 * alpha_level)
    ci_high = np.percentile(gammas, 100 * (1 - alpha_level))

    logger.info(
        f"Bootstrap gamma ({n_bootstrap} iterations):\n"
        f"  median = {np.median(gammas):.4f}\n"
        f"  {ci*100:.0f}% CI = [{ci_low:.4f}, {ci_high:.4f}]"
    )

    return float(np.median(gammas)), float(ci_low), float(ci_high)


def cross_validate_gamma(
    df: pd.DataFrame,
    alpha_col: str = "forcing_proxy",
    toa_col: str = "ceres_toa_cre_net_mean",
    n_folds: int = 5,
    random_seed: int = 42,
) -> dict:
    """
    K-fold cross-validation of the gamma transfer coefficient.

    Returns dict with per-fold gammas, R2 values, and summary stats.
    """
    mask = df[alpha_col].notna() & df[toa_col].notna()
    clean = df[mask]
    x = clean[alpha_col].values
    y = clean[toa_col].values

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    fold_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(x)):
        slope, intercept, _, _, _ = sp_stats.linregress(x[train_idx], y[train_idx])

        predicted = slope * x[test_idx] + intercept
        residuals = y[test_idx] - predicted
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y[test_idx] - np.mean(y[test_idx])) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rmse = np.sqrt(np.mean(residuals ** 2))

        fold_results.append({
            "fold": fold_idx,
            "gamma": slope,
            "intercept": intercept,
            "r2": r2,
            "rmse": rmse,
            "n_test": len(test_idx),
        })

    gammas = [f["gamma"] for f in fold_results]
    r2s = [f["r2"] for f in fold_results]

    result = {
        "fold_results": fold_results,
        "mean_gamma": np.mean(gammas),
        "std_gamma": np.std(gammas),
        "mean_r2": np.mean(r2s),
        "std_r2": np.std(r2s),
    }

    logger.info(
        f"Cross-validation ({n_folds}-fold):\n"
        f"  gamma = {result['mean_gamma']:.4f} +/- {result['std_gamma']:.4f}\n"
        f"  R2    = {result['mean_r2']:.4f} +/- {result['std_r2']:.4f}"
    )

    return result


def compute_biome_gammas(
    df: pd.DataFrame,
    alpha_col: str = "forcing_proxy",
    toa_col: str = "ceres_toa_cre_net_mean",
    biome_col: str = "biome_group",
    min_sites: int = 5,
) -> pd.DataFrame:
    """
    Compute gamma per biome group.

    Returns DataFrame with columns: biome, gamma, gamma_se, r2, p_value, n_sites.
    """
    records = []
    for biome, group in df.groupby(biome_col):
        mask = group[alpha_col].notna() & group[toa_col].notna()
        clean = group[mask]

        if len(clean) < min_sites:
            logger.info(f"  {biome}: {len(clean)} sites (below min {min_sites}), skipping")
            continue

        x = clean[alpha_col].values
        y = clean[toa_col].values
        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x, y)

        records.append({
            "biome": biome,
            "gamma": slope,
            "gamma_se": std_err,
            "r_squared": r_value ** 2,
            "p_value": p_value,
            "n_sites": len(clean),
        })

        logger.info(
            f"  {biome}: gamma = {slope:.4f} +/- {std_err:.4f}, "
            f"R2 = {r_value**2:.4f}, n = {len(clean)}"
        )

    return pd.DataFrame(records)


def partial_correlation_gamma(
    df: pd.DataFrame,
    alpha_col: str = "forcing_proxy",
    toa_col: str = "ceres_toa_cre_net_mean",
    control_cols: Optional[list[str]] = None,
) -> dict:
    """
    Compute partial correlation of alpha with TOA, controlling for
    latitude, altitude, and other confounds.

    Uses residual regression approach:
    1. Regress alpha on controls, get residuals
    2. Regress TOA on controls, get residuals
    3. Correlate the two residual series
    """
    if control_cols is None:
        control_cols = ["latitude"]

    available_controls = [c for c in control_cols if c in df.columns]
    if not available_controls:
        logger.warning("No control variables available for partial correlation")
        return {}

    mask = df[alpha_col].notna() & df[toa_col].notna()
    for c in available_controls:
        mask &= df[c].notna()
    clean = df[mask]

    if len(clean) < 15:
        return {}

    # Residualize alpha
    controls = clean[available_controls].values
    from numpy.linalg import lstsq
    X = np.column_stack([controls, np.ones(len(clean))])

    alpha_vals = clean[alpha_col].values
    coef_alpha, _, _, _ = lstsq(X, alpha_vals, rcond=None)
    alpha_resid = alpha_vals - X @ coef_alpha

    toa_vals = clean[toa_col].values
    coef_toa, _, _, _ = lstsq(X, toa_vals, rcond=None)
    toa_resid = toa_vals - X @ coef_toa

    r, p = sp_stats.pearsonr(alpha_resid, toa_resid)

    result = {
        "partial_r": r,
        "partial_r_squared": r ** 2,
        "partial_p": p,
        "controls": available_controls,
        "n": len(clean),
    }

    logger.info(
        f"Partial correlation (controlling {available_controls}):\n"
        f"  r = {r:.4f}, R2 = {r**2:.4f}, p = {p:.2e}"
    )

    return result


def financial_sensitivity(
    gamma_cre: float,
    gamma_cre_ci: tuple[float, float],
    gamma_lw: float = 0.0,
    r_net_amazon: float = 130.0,
    alpha_intact: float = 0.50,
    alpha_degraded: float = 1.50,
    dollars_per_wm2_per_ha: float = 11.76,
) -> dict:
    """
    Compute revised $/ha/year estimates using the empirical gamma.

    The gamma from regression is in W/m2 of TOA change per unit alpha.
    To get a dimensionless transfer fraction, divide by the surface
    forcing differential in W/m2.

    Parameters
    ----------
    gamma_cre : CRE regression slope (W/m2 per unit alpha)
    gamma_cre_ci : 95% CI bounds on gamma_cre
    gamma_lw : TOA LW regression slope (W/m2 per unit alpha)
    r_net_amazon : mean net radiation over Amazon (W/m2)
    alpha_intact : alpha(beta) for intact tropical forest
    alpha_degraded : alpha(beta) for degraded shrubland
    dollars_per_wm2_per_ha : conversion factor from Social Benefit methodology

    Returns
    -------
    dict with dimensionless transfer fractions and adjusted valuations
    """
    delta_alpha = alpha_degraded - alpha_intact
    surface_differential = delta_alpha * r_net_amazon

    # TOA CRE change from forest to shrubland conversion
    toa_cre_change = gamma_cre * delta_alpha
    toa_lw_change = gamma_lw * delta_alpha
    toa_total_change = toa_cre_change + toa_lw_change

    # Dimensionless transfer fractions
    frac_cre = toa_cre_change / surface_differential
    frac_total = toa_total_change / surface_differential
    frac_cre_low = (gamma_cre_ci[0] * delta_alpha) / surface_differential
    frac_cre_high = (gamma_cre_ci[1] * delta_alpha) / surface_differential

    # Financial: apply transfer fraction to Rob's unadjusted valuation
    unadjusted = surface_differential * dollars_per_wm2_per_ha
    adjusted_cre = unadjusted * frac_cre
    adjusted_total = unadjusted * frac_total
    adjusted_low = unadjusted * frac_cre_low
    adjusted_high = unadjusted * frac_cre_high

    result = {
        "surface_forcing_differential_wm2": surface_differential,
        "delta_alpha": delta_alpha,
        "toa_cre_change_wm2": toa_cre_change,
        "toa_lw_change_wm2": toa_lw_change,
        "toa_total_change_wm2": toa_total_change,
        "gamma_cre_wm2_per_alpha": gamma_cre,
        "gamma_lw_wm2_per_alpha": gamma_lw,
        "transfer_fraction_cre": frac_cre,
        "transfer_fraction_total": frac_total,
        "transfer_fraction_ci": (frac_cre_low, frac_cre_high),
        "unadjusted_dollars_per_ha": unadjusted,
        "dollars_per_ha_cre_only": adjusted_cre,
        "dollars_per_ha_total": adjusted_total,
        "dollars_per_ha_ci": (adjusted_low, adjusted_high),
    }

    logger.info(
        f"Financial sensitivity:\n"
        f"  Surface forcing differential: {surface_differential:.1f} W/m2\n"
        f"  TOA CRE change (forest->shrubland): {toa_cre_change:.1f} W/m2\n"
        f"  TOA LW change: {toa_lw_change:.1f} W/m2\n"
        f"  Transfer fraction (CRE): {frac_cre:.3f} ({frac_cre*100:.1f}%)\n"
        f"  Transfer fraction (total): {frac_total:.3f} ({frac_total*100:.1f}%)\n"
        f"  $/ha/year unadjusted: ${unadjusted:,.0f}\n"
        f"  $/ha/year CRE-adjusted: ${adjusted_cre:,.0f}\n"
        f"  $/ha/year total-adjusted: ${adjusted_total:,.0f}\n"
        f"  $/ha/year CI: ${adjusted_low:,.0f} - ${adjusted_high:,.0f}"
    )

    return result
