"""
MODIS Land Cover Homogeneity Check
===================================

Assesses whether the 1x1 degree CERES pixel around each FLUXNET site
is dominated by the same land cover type as the tower. Sites in
heterogeneous pixels may show weaker surface-to-TOA correlations
due to mixed signals.

Data source: MODIS MCD12C1 (Climate Modeling Grid, 0.05 degree)
Product: https://lpdaac.usgs.gov/products/mcd12c1v061/
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# IGBP class mapping (MCD12C1 uses integer codes)
IGBP_CODES = {
    0: "WAT", 1: "ENF", 2: "EBF", 3: "DNF", 4: "DBF", 5: "MF",
    6: "CSH", 7: "OSH", 8: "WSA", 9: "SAV", 10: "GRA", 11: "WET",
    12: "CRO", 13: "URB", 14: "CVM", 15: "SNO", 16: "BSV", 255: "UNC",
}


def compute_pixel_homogeneity(
    modis_lc_path: Optional[str | Path],
    site_lat: float,
    site_lon: float,
    site_igbp: str,
    window_deg: float = 0.5,
) -> Optional[float]:
    """
    Compute the fraction of MODIS pixels within the CERES footprint
    that match the tower's IGBP class.

    Parameters
    ----------
    modis_lc_path : Path to MCD12C1 HDF/NetCDF file
    site_lat, site_lon : tower coordinates
    site_igbp : IGBP class of the tower (e.g., "ENF", "EBF")
    window_deg : half-width of the window in degrees (0.5 = 1x1 degree)

    Returns
    -------
    Fraction (0-1) of pixels matching the tower class, or None if no data
    """
    if modis_lc_path is None:
        return None

    try:
        import xarray as xr
    except ImportError:
        logger.warning("xarray required for MODIS homogeneity check")
        return None

    modis_lc_path = Path(modis_lc_path)
    if not modis_lc_path.exists():
        logger.warning(f"MODIS file not found: {modis_lc_path}")
        return None

    try:
        ds = xr.open_dataset(modis_lc_path)
    except Exception as e:
        logger.warning(f"Could not open MODIS file: {e}")
        return None

    # Find the land cover variable (naming varies by format)
    lc_var = None
    for candidate in ["Majority_Land_Cover_Type_1", "LC_Type1", "land_cover"]:
        if candidate in ds.data_vars:
            lc_var = candidate
            break

    if lc_var is None:
        logger.warning(f"No land cover variable found in {modis_lc_path.name}")
        return None

    # Extract window around site
    try:
        window = ds[lc_var].sel(
            lat=slice(site_lat + window_deg, site_lat - window_deg),
            lon=slice(site_lon - window_deg, site_lon + window_deg),
        )
    except Exception:
        try:
            window = ds[lc_var].sel(
                latitude=slice(site_lat + window_deg, site_lat - window_deg),
                longitude=slice(site_lon - window_deg, site_lon + window_deg),
            )
        except Exception as e:
            logger.warning(f"Could not extract window: {e}")
            return None

    values = window.values.flatten()
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return None

    # Find the IGBP code for the site class
    target_code = None
    for code, name in IGBP_CODES.items():
        if name == site_igbp:
            target_code = code
            break

    if target_code is None:
        return None

    homogeneity = np.sum(values == target_code) / len(values)
    return float(homogeneity)


def add_homogeneity_to_sites(
    site_summary: pd.DataFrame,
    modis_lc_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Add a homogeneity score to each site in the summary.

    If MODIS data is not available, adds a column of NaN
    (the analysis proceeds without the filter).
    """
    if modis_lc_path is None:
        logger.info(
            "No MODIS land cover data provided. "
            "Homogeneity filter will not be applied. "
            "Download MCD12C1 from https://lpdaac.usgs.gov/products/mcd12c1v061/"
        )
        site_summary["pixel_homogeneity"] = np.nan
        return site_summary

    homogeneities = []
    for _, row in site_summary.iterrows():
        h = compute_pixel_homogeneity(
            modis_lc_path,
            row.get("latitude", np.nan),
            row.get("longitude", np.nan),
            row.get("igbp", "UNK"),
        )
        homogeneities.append(h)

    site_summary["pixel_homogeneity"] = homogeneities
    n_computed = sum(h is not None for h in homogeneities)
    logger.info(f"Homogeneity computed for {n_computed}/{len(site_summary)} sites")

    return site_summary


def filter_homogeneous_sites(
    df: pd.DataFrame,
    min_homogeneity: float = 0.70,
) -> pd.DataFrame:
    """
    Filter to only sites where the CERES pixel is dominated
    by the tower's land cover type.

    If homogeneity column is all NaN (no MODIS data), returns
    the full DataFrame unchanged.
    """
    if "pixel_homogeneity" not in df.columns:
        return df

    if df["pixel_homogeneity"].isna().all():
        logger.info("No homogeneity data available, returning all sites")
        return df

    mask = df["pixel_homogeneity"] >= min_homogeneity
    filtered = df[mask]
    logger.info(
        f"Homogeneity filter ({min_homogeneity:.0%}): "
        f"{len(filtered)}/{len(df)} sites retained"
    )
    return filtered
