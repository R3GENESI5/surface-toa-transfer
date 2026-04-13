#!/usr/bin/env python3
"""
Paper D — Surface-to-TOA Transfer Analysis Pipeline
====================================================

Tests whether biome-specific surface energy partitioning (alpha(beta))
propagates to top-of-atmosphere radiative signatures (CERES).

Builds on Paper A's pipeline: reuses FLUXNET loading, CERES co-location,
and alpha(beta) computation from Steps 1-3 of run_analysis.py.

Usage:
    python run_toa_analysis.py                     # Full pipeline
    python run_toa_analysis.py --skip-era5         # Skip ERA5
    python run_toa_analysis.py --fluxnet-dir PATH  # Custom FLUXNET path
    python run_toa_analysis.py --ceres-file PATH   # Local CERES NetCDF
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src.fluxnet_loader import build_site_summary
from src.ceres_loader import enrich_sites_with_ceres, compute_effective_forcing_proxy
from src.alpha_model import fit_alpha_model
from src.toa_transfer import (
    compute_all_transfer_coefficients,
    bootstrap_gamma,
    cross_validate_gamma,
    compute_biome_gammas,
    partial_correlation_gamma,
    financial_sensitivity,
)
from src.toa_visualize import (
    plot_alpha_vs_cre,
    plot_biome_gammas,
    plot_taylor_diagram,
    plot_financial_sensitivity,
    plot_gamma_map,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("paper_d")


def parse_args():
    parser = argparse.ArgumentParser(description="Paper D: Surface-to-TOA Transfer")
    parser.add_argument("--fluxnet-dir", type=str, default="data/raw/fluxnet")
    parser.add_argument("--ceres-file", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--skip-era5", action="store_true", default=True)
    parser.add_argument("--min-months", type=int, default=24)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    return parser.parse_args()


def main():
    args = parse_args()

    fig_dir = Path(args.output_dir) / "toa_figures"
    table_dir = Path(args.output_dir) / "toa_tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Paper D: Surface-to-TOA Transfer Coefficient (gamma)")
    logger.info("=" * 60)

    # ────────────────────────────────────────────────────────────
    # STEP 1-3: Reuse Paper A pipeline (FLUXNET + CERES + alpha)
    # ────────────────────────────────────────────────────────────
    logger.info("\nStep 1: Loading FLUXNET data...")
    site_summary = build_site_summary(
        data_dir=args.fluxnet_dir,
        min_valid_months=args.min_months,
    )
    if len(site_summary) == 0:
        logger.error("No FLUXNET data found.")
        sys.exit(1)
    logger.info(f"  {len(site_summary)} sites loaded")

    logger.info("\nStep 2: Loading CERES EBAF data (including CRE)...")
    site_summary = enrich_sites_with_ceres(
        site_summary,
        ceres_data_path=args.ceres_file,
        match_radius_deg=0.5,
    )

    logger.info("\nStep 3: Computing surface forcing proxy...")
    site_summary = compute_effective_forcing_proxy(site_summary)

    valid = site_summary.dropna(subset=["bowen_ratio", "forcing_proxy"])
    logger.info(f"  {len(valid)} sites with complete data")

    if len(valid) < 20:
        logger.error(f"Only {len(valid)} sites. Need at least 20.")
        sys.exit(1)

    # Fit alpha(beta) from Paper A (needed for context)
    logger.info("\nStep 3b: Fitting alpha(beta) model (Paper A baseline)...")
    beta = valid["bowen_ratio"].values
    alpha_obs = valid["forcing_proxy"].values
    alpha_params, alpha_diag = fit_alpha_model(beta, alpha_obs)

    # ────────────────────────────────────────────────────────────
    # STEP 4: Compute transfer coefficients
    # ────────────────────────────────────────────────────────────
    logger.info("\nStep 4: Computing transfer coefficients (gamma)...")
    transfer_results = compute_all_transfer_coefficients(valid)

    # ────────────────────────────────────────────────────────────
    # STEP 5: Bootstrap CI for primary gamma
    # ────────────────────────────────────────────────────────────
    logger.info(f"\nStep 5: Bootstrap CI ({args.n_bootstrap} iterations)...")
    primary_toa = "ceres_toa_cre_net_mean"

    gamma_median, gamma_ci_low, gamma_ci_high = bootstrap_gamma(
        valid, toa_col=primary_toa, n_bootstrap=args.n_bootstrap,
    )

    # Update the primary transfer result with CI
    for tr in transfer_results:
        if tr.toa_variable == "CRE_net":
            tr.gamma_ci_low = gamma_ci_low
            tr.gamma_ci_high = gamma_ci_high

    # ────────────────────────────────────────────────────────────
    # STEP 6: Cross-validation
    # ────────────────────────────────────────────────────────────
    logger.info(f"\nStep 6: {args.cv_folds}-fold cross-validation...")
    cv_results = cross_validate_gamma(
        valid, toa_col=primary_toa, n_folds=args.cv_folds,
    )

    # ────────────────────────────────────────────────────────────
    # STEP 7: Biome-level decomposition
    # ────────────────────────────────────────────────────────────
    logger.info("\nStep 7: Biome-level gamma decomposition...")
    biome_gammas = compute_biome_gammas(valid, toa_col=primary_toa)

    # ────────────────────────────────────────────────────────────
    # STEP 8: Partial correlation (controlling for latitude)
    # ────────────────────────────────────────────────────────────
    logger.info("\nStep 8: Partial correlation controlling for latitude...")
    partial = partial_correlation_gamma(
        valid, toa_col=primary_toa, control_cols=["latitude"],
    )

    # ────────────────────────────────────────────────────────────
    # STEP 9: Financial sensitivity
    # ────────────────────────────────────────────────────────────
    logger.info("\nStep 9: Financial sensitivity analysis...")
    gamma_cre = next(
        (tr.gamma for tr in transfer_results if tr.toa_variable == "CRE_net"),
        0.0
    )
    gamma_lw = next(
        (tr.gamma for tr in transfer_results if tr.toa_variable == "TOA_LW_up"),
        0.0
    )
    fin = financial_sensitivity(
        gamma_cre=gamma_cre,
        gamma_cre_ci=(gamma_ci_low, gamma_ci_high),
        gamma_lw=gamma_lw,
    )

    # ────────────────────────────────────────────────────────────
    # STEP 10: Generate figures
    # ────────────────────────────────────────────────────────────
    logger.info("\nStep 10: Generating figures...")

    primary_tr = next(
        (tr for tr in transfer_results if tr.toa_variable == "CRE_net"), None
    )
    if primary_tr:
        plot_alpha_vs_cre(valid, primary_tr, fig_dir)

    if len(biome_gammas) > 0:
        plot_biome_gammas(biome_gammas, fig_dir)

    plot_taylor_diagram(valid, fig_dir)

    # Financial sensitivity curve (dimensionless gamma fraction)
    gamma_frac_range = np.linspace(0, 0.5, 200)
    surface_diff = (1.50 - 0.50) * 130.0
    dollars_curve = gamma_frac_range * surface_diff * 11.76
    gamma_frac = fin.get("transfer_fraction_total", 0)
    gamma_frac_ci = fin.get("transfer_fraction_ci", (0, 0))
    plot_financial_sensitivity(
        gamma_frac_range, dollars_curve,
        gamma_frac, gamma_frac_ci,
        fig_dir,
    )

    plot_gamma_map(valid, fig_dir)

    # ────────────────────────────────────────────────────────────
    # STEP 11: Save results
    # ────────────────────────────────────────────────────────────
    logger.info("\nStep 11: Saving results...")

    valid.to_csv(table_dir / "site_summary_with_toa.csv", index=False)

    if len(biome_gammas) > 0:
        biome_gammas.to_csv(table_dir / "biome_gammas.csv", index=False)

    results = {
        "transfer_coefficients": [tr.to_dict() for tr in transfer_results],
        "gamma_cre": {
            "variable": "CRE_net",
            "gamma": gamma_cre,
            "gamma_ci_95": [gamma_ci_low, gamma_ci_high],
            "bootstrap_n": args.n_bootstrap,
        },
        "cross_validation": cv_results,
        "partial_correlation": partial,
        "financial_sensitivity": fin,
        "paper_a_params": alpha_params.to_dict(),
    }

    with open(table_dir / "toa_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # ────────────────────────────────────────────────────────────
    # SUMMARY
    # ────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("PAPER D RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Sites analyzed:          {len(valid)}")

    for tr in transfer_results:
        sig = "***" if tr.p_value < 0.001 else "**" if tr.p_value < 0.01 else "*" if tr.p_value < 0.05 else "ns"
        logger.info(
            f"  {tr.toa_variable:20s}: gamma = {tr.gamma:+.4f} +/- {tr.gamma_se:.4f}, "
            f"R2 = {tr.r_squared:.4f}, p = {tr.p_value:.2e} {sig}"
        )

    logger.info(f"")
    logger.info(f"Primary (CRE_net):")
    logger.info(f"  gamma = {gamma_cre:.4f}")
    logger.info(f"  95% CI = [{gamma_ci_low:.4f}, {gamma_ci_high:.4f}]")
    logger.info(f"  CV gamma = {cv_results.get('mean_gamma', np.nan):.4f} +/- {cv_results.get('std_gamma', np.nan):.4f}")
    logger.info(f"")
    logger.info(f"Transfer fractions:")
    logger.info(f"  CRE pathway:   {fin.get('transfer_fraction_cre', 0):.3f} ({fin.get('transfer_fraction_cre', 0)*100:.1f}%)")
    logger.info(f"  Total (CRE+LW): {fin.get('transfer_fraction_total', 0):.3f} ({fin.get('transfer_fraction_total', 0)*100:.1f}%)")
    logger.info(f"")
    logger.info(f"Financial impact:")
    logger.info(f"  Unadjusted:     ${fin['unadjusted_dollars_per_ha']:,.0f}/ha/year")
    logger.info(f"  CRE-adjusted:   ${fin['dollars_per_ha_cre_only']:,.0f}/ha/year")
    logger.info(f"  Total-adjusted:  ${fin['dollars_per_ha_total']:,.0f}/ha/year")
    ci = fin.get('dollars_per_ha_ci', (0, 0))
    logger.info(f"  CI range:       ${ci[0]:,.0f} - ${ci[1]:,.0f}/ha/year")
    logger.info(f"")
    logger.info(f"Figures saved to:        {fig_dir}/")
    logger.info(f"Tables saved to:         {table_dir}/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
