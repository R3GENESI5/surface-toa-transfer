"""
α(β) Forcing Coefficient — Curve Fitting and Validation
========================================================

Fits the non-uniform GHG forcing coefficient α as a function of
the Bowen ratio β, using FLUXNET + CERES co-located observations.

Model:
    α(β) = α_min + (α_max − α_min) · β^γ / (β^γ + β₀^γ)

This is a Hill-function (sigmoid on log-β), which:
  - Saturates at α_min for low β (forests, high LE)
  - Saturates at α_max for high β (deserts, high H)
  - Has inflection at β = β₀
  - γ controls steepness

The key empirical test: does the observed forcing proxy (from CERES)
correlate with Bowen ratio (from FLUXNET) in the shape predicted
by this model?
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


@dataclass
class AlphaModelParams:
    """Fitted parameters for α(β)."""
    alpha_min: float
    alpha_max: float
    beta_0: float
    gamma: float
    # Uncertainties (from covariance matrix)
    alpha_min_se: float = np.nan
    alpha_max_se: float = np.nan
    beta_0_se: float = np.nan
    gamma_se: float = np.nan

    def to_dict(self) -> dict:
        return {
            "alpha_min": self.alpha_min,
            "alpha_max": self.alpha_max,
            "beta_0": self.beta_0,
            "gamma": self.gamma,
            "alpha_min_se": self.alpha_min_se,
            "alpha_max_se": self.alpha_max_se,
            "beta_0_se": self.beta_0_se,
            "gamma_se": self.gamma_se,
        }


def alpha_function(beta: np.ndarray, alpha_min: float, alpha_max: float,
                   beta_0: float, gamma: float) -> np.ndarray:
    """
    The α(β) forcing coefficient.

    α(β) = α_min + (α_max − α_min) · β^γ / (β^γ + β₀^γ)
    """
    beta = np.asarray(beta, dtype=float)
    numerator = beta ** gamma
    denominator = numerator + beta_0 ** gamma
    return alpha_min + (alpha_max - alpha_min) * numerator / denominator


def fit_alpha_model(
    beta: np.ndarray,
    alpha_observed: np.ndarray,
    weights: Optional[np.ndarray] = None,
    initial_guess: tuple = (0.80, 1.40, 1.0, 1.5),
    bounds: tuple = ((0.5, 1.1, 0.1, 0.5), (1.0, 2.0, 5.0, 5.0)),
) -> tuple[AlphaModelParams, dict]:
    """
    Fit the α(β) model to observed data using nonlinear least squares.

    Parameters
    ----------
    beta : array of Bowen ratios
    alpha_observed : array of observed forcing proxy values
    weights : optional array of weights (1/variance)
    initial_guess : (α_min, α_max, β₀, γ)
    bounds : ((lower bounds), (upper bounds))

    Returns
    -------
    params : AlphaModelParams with fitted values and standard errors
    diagnostics : dict with R², RMSE, residuals, etc.
    """
    # Clean data
    mask = np.isfinite(beta) & np.isfinite(alpha_observed) & (beta > 0)
    beta_clean = beta[mask]
    alpha_clean = alpha_observed[mask]

    if len(beta_clean) < 10:
        raise ValueError(f"Insufficient data points ({len(beta_clean)}) for fitting")

    sigma = None
    if weights is not None:
        sigma = 1.0 / np.sqrt(weights[mask])

    # Fit
    popt, pcov = curve_fit(
        alpha_function,
        beta_clean,
        alpha_clean,
        p0=initial_guess,
        bounds=bounds,
        sigma=sigma,
        maxfev=10000,
    )

    # Standard errors from covariance diagonal
    perr = np.sqrt(np.diag(pcov))

    params = AlphaModelParams(
        alpha_min=popt[0], alpha_max=popt[1],
        beta_0=popt[2], gamma=popt[3],
        alpha_min_se=perr[0], alpha_max_se=perr[1],
        beta_0_se=perr[2], gamma_se=perr[3],
    )

    # Diagnostics
    predicted = alpha_function(beta_clean, *popt)
    residuals = alpha_clean - predicted
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((alpha_clean - np.mean(alpha_clean)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = np.sqrt(np.mean(residuals ** 2))
    mae = np.mean(np.abs(residuals))

    diagnostics = {
        "n_points": len(beta_clean),
        "r_squared": r_squared,
        "rmse": rmse,
        "mae": mae,
        "residuals": residuals,
        "predicted": predicted,
        "beta_used": beta_clean,
        "alpha_used": alpha_clean,
    }

    logger.info(
        f"α(β) fit results:\n"
        f"  α_min = {params.alpha_min:.4f} ± {params.alpha_min_se:.4f}\n"
        f"  α_max = {params.alpha_max:.4f} ± {params.alpha_max_se:.4f}\n"
        f"  β₀    = {params.beta_0:.4f} ± {params.beta_0_se:.4f}\n"
        f"  γ     = {params.gamma:.4f} ± {params.gamma_se:.4f}\n"
        f"  R²    = {r_squared:.4f}\n"
        f"  RMSE  = {rmse:.4f}\n"
        f"  n     = {len(beta_clean)}"
    )

    return params, diagnostics


def cross_validate_alpha(
    beta: np.ndarray,
    alpha_observed: np.ndarray,
    n_folds: int = 5,
    random_seed: int = 42,
    initial_guess: tuple = (0.80, 1.40, 1.0, 1.5),
    bounds: tuple = ((0.5, 1.1, 0.1, 0.5), (1.0, 2.0, 5.0, 5.0)),
) -> dict:
    """
    K-fold cross-validation of the α(β) model.

    Trains on k-1 folds, predicts on the held-out fold.
    Reports out-of-sample R², RMSE, and parameter stability.

    Returns
    -------
    dict with:
        fold_results : list of per-fold metrics
        mean_r2, std_r2 : cross-validated R²
        mean_rmse, std_rmse : cross-validated RMSE
        param_stability : std of fitted params across folds
    """
    mask = np.isfinite(beta) & np.isfinite(alpha_observed) & (beta > 0)
    beta_clean = beta[mask]
    alpha_clean = alpha_observed[mask]

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    fold_results = []
    all_params = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(beta_clean)):
        beta_train = beta_clean[train_idx]
        alpha_train = alpha_clean[train_idx]
        beta_test = beta_clean[test_idx]
        alpha_test = alpha_clean[test_idx]

        try:
            popt, _ = curve_fit(
                alpha_function,
                beta_train,
                alpha_train,
                p0=initial_guess,
                bounds=bounds,
                maxfev=10000,
            )
        except RuntimeError:
            logger.warning(f"Fold {fold_idx}: fitting failed, skipping")
            continue

        # Out-of-sample predictions
        predicted = alpha_function(beta_test, *popt)
        residuals = alpha_test - predicted

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((alpha_test - np.mean(alpha_test)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rmse = np.sqrt(np.mean(residuals ** 2))

        fold_results.append({
            "fold": fold_idx,
            "r2": r2,
            "rmse": rmse,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "params": popt.tolist(),
        })
        all_params.append(popt)

    if not fold_results:
        return {"error": "All folds failed"}

    r2_values = [f["r2"] for f in fold_results]
    rmse_values = [f["rmse"] for f in fold_results]
    param_array = np.array(all_params)

    result = {
        "fold_results": fold_results,
        "mean_r2": np.mean(r2_values),
        "std_r2": np.std(r2_values),
        "mean_rmse": np.mean(rmse_values),
        "std_rmse": np.std(rmse_values),
        "param_means": param_array.mean(axis=0).tolist(),
        "param_stds": param_array.std(axis=0).tolist(),
        "param_names": ["alpha_min", "alpha_max", "beta_0", "gamma"],
    }

    logger.info(
        f"Cross-validation ({n_folds}-fold):\n"
        f"  R² = {result['mean_r2']:.4f} ± {result['std_r2']:.4f}\n"
        f"  RMSE = {result['mean_rmse']:.4f} ± {result['std_rmse']:.4f}\n"
        f"  Parameter stability (std across folds):\n"
        f"    α_min: {param_array[:, 0].std():.4f}\n"
        f"    α_max: {param_array[:, 1].std():.4f}\n"
        f"    β₀:    {param_array[:, 2].std():.4f}\n"
        f"    γ:     {param_array[:, 3].std():.4f}"
    )

    return result


def compare_against_uniform(
    beta: np.ndarray,
    alpha_observed: np.ndarray,
    params: AlphaModelParams,
) -> dict:
    """
    Compare α(β) predictions against the uniform-forcing assumption (α=1).

    This is the key result: how much error does uniform forcing introduce,
    and how much does the α(β) correction reduce it?

    Returns
    -------
    dict with:
        uniform_rmse, uniform_mae : error under α=1 assumption
        corrected_rmse, corrected_mae : error with α(β) correction
        improvement_pct : percentage reduction in RMSE
        biome_breakdown : per-biome error comparison
    """
    mask = np.isfinite(beta) & np.isfinite(alpha_observed) & (beta > 0)
    beta_clean = beta[mask]
    alpha_clean = alpha_observed[mask]

    # Uniform assumption: α = 1 for all sites
    uniform_pred = np.ones_like(alpha_clean)
    uniform_residuals = alpha_clean - uniform_pred
    uniform_rmse = np.sqrt(np.mean(uniform_residuals ** 2))
    uniform_mae = np.mean(np.abs(uniform_residuals))

    # α(β) corrected
    corrected_pred = alpha_function(
        beta_clean, params.alpha_min, params.alpha_max,
        params.beta_0, params.gamma
    )
    corrected_residuals = alpha_clean - corrected_pred
    corrected_rmse = np.sqrt(np.mean(corrected_residuals ** 2))
    corrected_mae = np.mean(np.abs(corrected_residuals))

    improvement = (1 - corrected_rmse / uniform_rmse) * 100 if uniform_rmse > 0 else 0

    result = {
        "uniform_rmse": uniform_rmse,
        "uniform_mae": uniform_mae,
        "corrected_rmse": corrected_rmse,
        "corrected_mae": corrected_mae,
        "improvement_pct": improvement,
        "n_sites": len(beta_clean),
    }

    logger.info(
        f"Uniform vs. α(β) correction:\n"
        f"  Uniform:   RMSE = {uniform_rmse:.4f}, MAE = {uniform_mae:.4f}\n"
        f"  α(β):      RMSE = {corrected_rmse:.4f}, MAE = {corrected_mae:.4f}\n"
        f"  Improvement: {improvement:.1f}% reduction in RMSE"
    )

    return result
