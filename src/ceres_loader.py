"""
CERES EBAF Data Loader — Radiative Imbalance per Site
=====================================================

Loads CERES EBAF (Energy Balanced and Filled) data and extracts
radiative budget metrics co-located with FLUXNET tower sites.

Data source: https://ceres.larc.nasa.gov/data/
Product: CERES_EBAF-Surface_Edition4.2 (monthly, 1° × 1° grid)

Two access modes:
  1. Local NetCDF files (if pre-downloaded)
  2. OpenDAP remote access (requires internet, no account needed)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# CERES EBAF OpenDAP endpoint (combined TOA + Surface, Edition 4.2)
CERES_OPENDAP_URLS = [
    "https://opendap.larc.nasa.gov/opendap/hyrax/CERES/EBAF/"
    "Edition4.2/CERES_EBAF_Edition4.2_200003-202407.nc",
    "https://opendap.larc.nasa.gov/opendap/CERES/EBAF/"
    "Edition4.2/CERES_EBAF_Edition4.2_200003-202407.nc",
    "http://opendap.larc.nasa.gov/opendap/hyrax/CERES/EBAF/"
    "Edition4.2/CERES_EBAF_Edition4.2_200003-202407.nc",
]
# Keep single URL for backward compat
CERES_OPENDAP_URL = CERES_OPENDAP_URLS[0]

# Variables of interest — try multiple naming conventions
# The combined EBAF file may use different names than the surface-only file
CERES_VARIABLES = {
    # Surface variables
    "sfc_net_sw":  ["sfc_net_sw_all_mon", "sfc_comp_net_sw_all_mon"],
    "sfc_net_lw":  ["sfc_net_lw_all_mon", "sfc_comp_net_lw_all_mon"],
    "sfc_down_sw": ["sfc_down_sw_all_mon", "sfc_comp_sw_down_all_mon"],
    # TOA all-sky
    "toa_net":     ["toa_net_all_mon"],
    "toa_sw_up":   ["toa_sw_all_mon"],
    "toa_lw_up":   ["toa_lw_all_mon"],
    # TOA clear-sky (Paper D: separate cloud from surface/atmosphere effects)
    "toa_net_clr": ["toa_net_clr_t_mon", "toa_net_clr_c_mon", "toa_net_clr_mon"],
    "toa_sw_clr":  ["toa_sw_clr_t_mon", "toa_sw_clr_c_mon", "toa_sw_clr_mon"],
    "toa_lw_clr":  ["toa_lw_clr_t_mon", "toa_lw_clr_c_mon", "toa_lw_clr_mon"],
    # Cloud radiative effect (Paper D: isolates cloud pathway)
    "toa_cre_sw":  ["toa_cre_sw_mon"],
    "toa_cre_lw":  ["toa_cre_lw_mon"],
    "toa_cre_net": ["toa_cre_net_mon"],
}


def load_ceres_local(filepath: str | Path) -> Optional["xarray.Dataset"]:
    """Load CERES EBAF from a local NetCDF file."""
    import xarray as xr

    filepath = Path(filepath)
    if not filepath.exists():
        logger.warning(f"CERES file not found: {filepath}")
        return None

    ds = xr.open_dataset(filepath)
    logger.info(f"Loaded CERES data: {list(ds.data_vars)[:10]}...")
    return ds


def load_ceres_opendap() -> Optional["xarray.Dataset"]:
    """Load CERES EBAF via OpenDAP (remote, no download required)."""
    import xarray as xr

    for url in CERES_OPENDAP_URLS:
        # Try default engine first, then pydap
        for engine in [None, "pydap"]:
            engine_label = engine or "default"
            logger.info(f"Trying CERES EBAF via OpenDAP ({engine_label}): {url}")
            try:
                kwargs = {"engine": engine} if engine else {}
                ds = xr.open_dataset(url, **kwargs)
                logger.info(f"CERES data loaded. Variables: {list(ds.data_vars)[:15]}...")
                logger.info(f"Coordinates: {list(ds.coords)}")
                return ds
            except Exception as e:
                logger.warning(f"  Failed ({engine_label}): {type(e).__name__}")
                continue

    logger.error(
        "Could not access CERES via any OpenDAP URL.\n"
        "Alternative: download CERES EBAF NetCDF from "
        "https://ceres.larc.nasa.gov/data/ and pass the local path via --ceres-file."
    )
    return None


def extract_site_radiation(
    ceres_ds: "xarray.Dataset",
    lat: float,
    lon: float,
    match_radius_deg: float = 0.5,
) -> dict:
    """
    Extract mean radiation budget for a single site location.

    Uses nearest-neighbor matching within match_radius_deg.
    Returns time-averaged radiation components in W/m².
    """
    import xarray as xr

    # Find nearest grid cell
    try:
        site = ceres_ds.sel(lat=lat, lon=lon % 360, method="nearest", tolerance=match_radius_deg)
    except (KeyError, ValueError):
        # Try different coordinate names
        try:
            site = ceres_ds.sel(
                latitude=lat, longitude=lon % 360,
                method="nearest", tolerance=match_radius_deg
            )
        except Exception:
            return {}

    result = {}

    # Extract each variable, taking time mean
    for key, var_candidates in CERES_VARIABLES.items():
        if isinstance(var_candidates, str):
            var_candidates = [var_candidates]
        for var_name in var_candidates:
            if var_name in site.data_vars:
                vals = site[var_name].values
                vals = vals[np.isfinite(vals)]
                if len(vals) > 0:
                    result[f"ceres_{key}_mean"] = float(np.mean(vals))
                    result[f"ceres_{key}_std"] = float(np.std(vals))
                break

    # Compute derived quantities
    sfc_net_sw = result.get("ceres_sfc_net_sw_mean", np.nan)
    sfc_net_lw = result.get("ceres_sfc_net_lw_mean", np.nan)
    toa_net = result.get("ceres_toa_net_mean", np.nan)

    # Surface net radiation (SW + LW)
    if np.isfinite(sfc_net_sw) and np.isfinite(sfc_net_lw):
        result["ceres_sfc_net_rad"] = sfc_net_sw + sfc_net_lw

    # Atmospheric absorption (TOA net minus surface net)
    if np.isfinite(toa_net) and np.isfinite(result.get("ceres_sfc_net_rad", np.nan)):
        result["ceres_atm_absorption"] = toa_net - result["ceres_sfc_net_rad"]

    # Cloud radiative effect (all-sky minus clear-sky) if not directly available
    for component in ["sw", "lw", "net"]:
        cre_key = f"ceres_toa_cre_{component}_mean"
        if cre_key not in result:
            all_key = f"ceres_toa_{component.replace('net', 'net')}_mean"
            clr_key = f"ceres_toa_{component}_clr_mean"
            # For SW: CRE_SW = SW_clr - SW_all (clear-sky minus all-sky upward)
            # For LW: CRE_LW = LW_all - LW_clr (all-sky minus clear-sky upward)
            # For net: CRE_net = net_all - net_clr
            if component == "sw":
                all_val = result.get(f"ceres_toa_sw_up_mean", np.nan)
                clr_val = result.get(f"ceres_toa_sw_clr_mean", np.nan)
                if np.isfinite(all_val) and np.isfinite(clr_val):
                    result[cre_key] = clr_val - all_val
            elif component == "lw":
                all_val = result.get(f"ceres_toa_lw_up_mean", np.nan)
                clr_val = result.get(f"ceres_toa_lw_clr_mean", np.nan)
                if np.isfinite(all_val) and np.isfinite(clr_val):
                    result[cre_key] = all_val - clr_val
            elif component == "net":
                all_val = result.get("ceres_toa_net_mean", np.nan)
                clr_val = result.get("ceres_toa_net_clr_mean", np.nan)
                if np.isfinite(all_val) and np.isfinite(clr_val):
                    result[cre_key] = all_val - clr_val

    return result


def enrich_sites_with_ceres(
    site_summary: pd.DataFrame,
    ceres_data_path: Optional[str | Path] = None,
    match_radius_deg: float = 0.5,
) -> pd.DataFrame:
    """
    Add CERES radiation data to the FLUXNET site summary.

    Parameters
    ----------
    site_summary : DataFrame with columns latitude, longitude
    ceres_data_path : Path to local CERES NetCDF, or None for OpenDAP
    match_radius_deg : Maximum distance for spatial matching

    Returns
    -------
    DataFrame with additional CERES columns
    """
    # Load CERES data
    if ceres_data_path is not None:
        ceres_ds = load_ceres_local(ceres_data_path)
    else:
        ceres_ds = load_ceres_opendap()

    if ceres_ds is None:
        logger.error("Could not load CERES data. Returning sites without radiation data.")
        return site_summary

    ceres_records = []
    for _, row in site_summary.iterrows():
        lat = row.get("latitude", np.nan)
        lon = row.get("longitude", np.nan)

        if not (np.isfinite(lat) and np.isfinite(lon)):
            ceres_records.append({})
            continue

        rad = extract_site_radiation(ceres_ds, lat, lon, match_radius_deg)
        ceres_records.append(rad)
        logger.info(
            f"  {row['site_id']}: "
            f"SFC net = {rad.get('ceres_sfc_net_rad', 'N/A'):.1f} W/m², "
            f"TOA net = {rad.get('ceres_toa_net_mean', 'N/A'):.1f} W/m²"
            if rad else f"  {row['site_id']}: no CERES match"
        )

    ceres_df = pd.DataFrame(ceres_records)
    enriched = pd.concat([site_summary.reset_index(drop=True), ceres_df], axis=1)

    n_matched = ceres_df.dropna(how="all").shape[0]
    logger.info(f"CERES matching complete: {n_matched}/{len(site_summary)} sites matched")

    return enriched


def compute_effective_forcing_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the surface forcing proxy alpha = H / R_net.

    This is the observational alpha(beta) from Paper A (Shahid 2026):
    the fraction of net radiation partitioned to sensible heat.

    alpha < 1: ecosystem attenuates forcing (latent-heat dominated)
    alpha > 1: ecosystem amplifies forcing (sensible-heat dominated)

    Uses FLUXNET-measured H and R_net, NOT CERES ratios.
    The proxy is normalized so temperate sites (beta ~ 0.8-1.2) yield alpha ~ 1.0.
    """
    if "mean_H" not in df.columns or "mean_netrad" not in df.columns:
        logger.warning("Cannot compute forcing proxy — missing mean_H or mean_netrad")
        return df

    # Require meaningful net radiation to avoid division artifacts
    valid_mask = df["mean_netrad"].notna() & (df["mean_netrad"] > 10) & df["mean_H"].notna()

    # Raw alpha: fraction of net radiation going to sensible heat
    df["forcing_proxy_raw"] = np.where(valid_mask, df["mean_H"] / df["mean_netrad"], np.nan)

    # Normalize so temperate sites (beta ~ 0.8-1.2) have alpha ~ 1.0
    temperate_mask = valid_mask & df["bowen_ratio"].between(0.8, 1.2)
    if temperate_mask.sum() > 0:
        baseline = df.loc[temperate_mask, "forcing_proxy_raw"].median()
        if baseline > 0:
            df["forcing_proxy"] = df["forcing_proxy_raw"] / baseline
        else:
            df["forcing_proxy"] = df["forcing_proxy_raw"]
    else:
        df["forcing_proxy"] = df["forcing_proxy_raw"]

    n_valid = df["forcing_proxy"].notna().sum()
    logger.info(
        f"Forcing proxy (alpha = H/R_net): {n_valid} sites, "
        f"range [{df['forcing_proxy'].min():.3f}, {df['forcing_proxy'].max():.3f}]"
    )

    return df
