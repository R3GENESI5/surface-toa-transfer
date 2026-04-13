"""
FLUXNET2015 Data Loader and Bowen Ratio Calculator
===================================================

Loads FLUXNET2015 FULLSET monthly (MM) files, computes site-level
Bowen ratios, and produces a clean site summary table.

Data source: https://fluxnet.org/data/fluxnet2015-dataset/
License: CC-BY-4.0 (requires free registration)

Expected directory layout:
    data/raw/fluxnet/
        FLX_<SITE>_FLUXNET2015_FULLSET_MM_<YEARS>.csv
        ...
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# FLUXNET quality-flag threshold (0 = measured, 1 = good gap-fill, 2 = medium, 3 = poor)
# We accept 0 and 1 only.
MAX_QC = 1

# Physical sanity bounds
LE_MIN, LE_MAX = 0.0, 500.0      # W/m² — latent heat flux
H_MIN, H_MAX = -100.0, 500.0     # W/m² — sensible heat flux
BOWEN_MIN, BOWEN_MAX = 0.01, 50  # dimensionless


def discover_fluxnet_files(data_dir: str | Path) -> list[Path]:
    """Find all FLUXNET monthly CSV files in a directory.

    Supports multiple naming conventions:
      - FLUXNET2015: FLX_<SITE>_FLUXNET2015_FULLSET_MM_*.csv
      - FluxDataKit: FLX_<SITE>_FLUXDATAKIT_FULLSET_MM_*.csv
      - Generic:     FLX_*_MM_*.csv
      - Daily files as fallback: FLX_*_DD_*.csv (aggregated to monthly)
    """
    data_dir = Path(data_dir)

    # Try multiple patterns in order of preference
    patterns = [
        "FLX_*_FULLSET_MM_*.csv",           # Generic FULLSET monthly
        "FLX_*_FLUXNET2015_FULLSET_MM_*.csv",  # FLUXNET2015 specific
        "FLX_*_FLUXDATAKIT_FULLSET_MM_*.csv",  # FluxDataKit specific
        "FLX_*_MM_*.csv",                    # Any monthly
        "*_MM_*.csv",                         # Broader match
    ]

    files = []
    for pattern in patterns:
        files = sorted(data_dir.glob(pattern))
        if files:
            break

    # If no monthly files, try daily (we'll aggregate in the loader)
    if not files:
        daily_patterns = [
            "FLX_*_FULLSET_DD_*.csv",
            "FLX_*_DD_*.csv",
            "*_DD_*.csv",
        ]
        for pattern in daily_patterns:
            files = sorted(data_dir.glob(pattern))
            if files:
                logger.info(f"No monthly files found; using {len(files)} daily files (will aggregate)")
                break

    # Also check subdirectories (FluxDataKit may extract with nesting)
    if not files:
        for pattern in ["**/*_MM_*.csv", "**/*_DD_*.csv"]:
            files = sorted(data_dir.rglob(pattern.replace("**/", "")))
            if files:
                break

    logger.info(f"Found {len(files)} FLUXNET files in {data_dir}")
    return files


def extract_site_id(filepath: Path) -> str:
    """Extract the FLUXNET site ID (e.g. 'US-Ha1') from the filename.

    Handles:
      FLX_US-Ha1_FLUXNET2015_FULLSET_MM_...
      FLX_US-Ha1_FLUXDATAKIT_FULLSET_DD_...
    """
    name = filepath.stem
    parts = name.split("_")
    # Format: FLX_<CC-Xxx>_...  where CC-Xxx is the site ID
    if len(parts) >= 2:
        candidate = parts[1]
        # Check if it looks like a site ID (2 letters, dash, 3+ chars)
        if len(candidate) >= 5 and "-" in candidate:
            return candidate
        # Try joining parts 1 and 2 with dash
        if len(parts) >= 3 and len(parts[1]) == 2:
            return f"{parts[1]}-{parts[2]}"
        return candidate
    return name


def load_site_data(filepath: Path) -> Optional[pd.DataFrame]:
    """
    Load a single FLUXNET file (monthly, daily, or half-hourly) and
    return cleaned flux data aggregated to monthly resolution.

    Returns DataFrame with columns:
        timestamp, H, LE, H_QC, LE_QC, TA, NETRAD, USTAR
    or None if the file cannot be parsed.
    """
    try:
        df = pd.read_csv(filepath, na_values=["-9999", "-9999.0"])
    except Exception as e:
        logger.warning(f"Could not read {filepath.name}: {e}")
        return None

    # Identify the timestamp column
    ts_col = "TIMESTAMP" if "TIMESTAMP" in df.columns else None
    if ts_col is None:
        for c in df.columns:
            if "timestamp" in c.lower():
                ts_col = c
                break
    if ts_col is None:
        logger.warning(f"No timestamp column in {filepath.name}")
        return None

    # Parse timestamp — handle multiple formats
    ts_str = df[ts_col].astype(str)
    sample = ts_str.iloc[0]
    if len(sample) == 6:
        df["timestamp"] = pd.to_datetime(ts_str, format="%Y%m", errors="coerce")
        temporal = "MM"
    elif len(sample) == 8:
        df["timestamp"] = pd.to_datetime(ts_str, format="%Y%m%d", errors="coerce")
        temporal = "DD"
    elif len(sample) >= 12:
        df["timestamp"] = pd.to_datetime(ts_str, format="%Y%m%d%H%M", errors="coerce")
        temporal = "HH"
    else:
        df["timestamp"] = pd.to_datetime(ts_str, errors="coerce")
        temporal = "UNK"

    # Select flux columns — try multiple naming conventions
    # Prefer energy-balance-corrected (_CORR) over gap-filled (_F_MDS)
    col_map = {}
    for target, candidates in {
        "H": ["H_CORR", "H_F_MDS", "H"],
        "LE": ["LE_CORR", "LE_F_MDS", "LE"],
        "H_QC": ["H_CORR_QC", "H_F_MDS_QC", "H_QC"],
        "LE_QC": ["LE_CORR_QC", "LE_F_MDS_QC", "LE_QC"],
        "TA": ["TA_F_MDS", "TA_F", "TA"],
        "NETRAD": ["NETRAD", "NETRAD_F"],
        "USTAR": ["USTAR_50", "USTAR"],
    }.items():
        for c in candidates:
            if c in df.columns and df[c].notna().any():
                col_map[target] = c
                break

    if "H" not in col_map or "LE" not in col_map:
        logger.warning(f"Missing H or LE columns in {filepath.name}")
        return None

    result = pd.DataFrame({"timestamp": df["timestamp"]})
    for target, source in col_map.items():
        result[target] = df[source].values

    # Fill missing QC columns with NaN (will be filtered later)
    for qc_col in ["H_QC", "LE_QC"]:
        if qc_col not in result.columns:
            result[qc_col] = np.nan

    # Aggregate to monthly if needed
    if temporal in ("DD", "HH"):
        result = result.dropna(subset=["timestamp"])
        result = result.set_index("timestamp")
        monthly = result.resample("ME").agg({
            "H": "mean",
            "LE": "mean",
            "H_QC": "mean",
            "LE_QC": "mean",
            **({col: "mean" for col in ["TA", "NETRAD", "USTAR"] if col in result.columns}),
        }).reset_index()
        monthly = monthly.rename(columns={"timestamp": "timestamp"})
        return monthly

    return result


def compute_bowen_ratio(df: pd.DataFrame) -> dict:
    """
    Compute the mean Bowen ratio and related statistics for one site.

    Filters by quality flags and physical bounds before computing.

    Returns dict with:
        bowen_ratio, bowen_std, mean_H, mean_LE, mean_netrad,
        n_valid_months, total_months
    """
    mask = pd.Series(True, index=df.index)

    # Quality filter (if QC columns exist and aren't all NaN)
    if df["H_QC"].notna().any():
        mask &= df["H_QC"] <= MAX_QC
    if df["LE_QC"].notna().any():
        mask &= df["LE_QC"] <= MAX_QC

    # Physical bounds
    mask &= df["LE"].between(LE_MIN, LE_MAX)
    mask &= df["H"].between(H_MIN, H_MAX)

    # Require non-trivial LE to avoid division by near-zero
    mask &= df["LE"] > 5.0  # At least 5 W/m²

    filtered = df[mask]

    if len(filtered) < 6:  # Need at least 6 valid months
        return None

    mean_h = filtered["H"].mean()
    mean_le = filtered["LE"].mean()
    bowen = mean_h / mean_le

    if not (BOWEN_MIN <= bowen <= BOWEN_MAX):
        return None

    # Monthly Bowen ratios for standard deviation
    monthly_bowen = filtered["H"] / filtered["LE"]
    monthly_bowen = monthly_bowen[monthly_bowen.between(BOWEN_MIN, BOWEN_MAX)]

    return {
        "bowen_ratio": bowen,
        "bowen_std": monthly_bowen.std(),
        "bowen_median": monthly_bowen.median(),
        "mean_H": mean_h,
        "mean_LE": mean_le,
        "mean_netrad": filtered["NETRAD"].mean() if "NETRAD" in filtered.columns else np.nan,
        "mean_TA": filtered["TA"].mean() if "TA" in filtered.columns else np.nan,
        "n_valid_months": len(filtered),
        "total_months": len(df),
    }


def load_site_metadata(data_dir: str | Path) -> Optional[pd.DataFrame]:
    """
    Load FLUXNET site metadata (coordinates, IGBP class, etc.).

    Looks for a metadata CSV in the data directory. If not found,
    returns None and metadata must be supplied separately.

    Handles:
      - FLUXNET2015 BIF files: FLX_AA-Flx_BIF_ALL_*.csv
      - FluxDataKit: fdk_site_info.csv / site_metadata.csv
    """
    data_dir = Path(data_dir)
    candidates = (
        list(data_dir.glob("*BIF*.csv"))
        + list(data_dir.glob("*metadata*.csv"))
        + list(data_dir.glob("*site_info*.csv"))
        + list(data_dir.glob("fdk_*.csv"))
    )
    # Also check parent directory
    candidates += list(data_dir.parent.glob("fdk_site_info.csv"))

    if not candidates:
        return None

    df = pd.read_csv(candidates[0])

    # Normalize column names
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if cl in ("sitename", "site_id", "site"):
            col_map["site_id"] = col
        elif cl in ("lat", "latitude"):
            col_map["latitude"] = col
        elif cl in ("lon", "longitude"):
            col_map["longitude"] = col
        elif cl in ("igbp_land_use", "igbp"):
            col_map["igbp"] = col
        elif cl in ("koeppen_code",):
            col_map["koeppen"] = col
        elif cl in ("country",):
            col_map["country"] = col

    result = pd.DataFrame()
    for target, source in col_map.items():
        result[target] = df[source].values

    logger.info(f"Loaded metadata for {len(result)} sites from {candidates[0].name}")
    return result


# ── Hard-coded FLUXNET2015 site metadata (subset) ──────────────────
# This covers the most commonly used sites. If the full metadata CSV
# is available, it takes precedence.
KNOWN_SITES = {
    # site_id: (lat, lon, igbp_class, name)
    "AR-SLu": (-33.46, -66.46, "MF", "San Luis"),
    "AT-Neu": (47.12, 11.32, "GRA", "Neustift"),
    "AU-Tum": (-35.66, 148.15, "EBF", "Tumbarumba"),
    "AU-How": (-12.49, 131.15, "WSA", "Howard Springs"),
    "AU-DaS": (-14.16, 131.39, "SAV", "Daly River Savanna"),
    "BE-Vie": (50.31, 5.99, "MF", "Vielsalm"),
    "BR-Sa1": (-2.86, -54.96, "EBF", "Santarem Km67"),
    "BR-Sa3": (-3.02, -54.97, "EBF", "Santarem Km83"),
    "CA-Man": (55.88, -98.48, "ENF", "Manitoba BOREAS"),
    "CA-Oas": (53.63, -106.20, "DBF", "Saskatchewan Aspen"),
    "CA-Obs": (53.99, -105.12, "ENF", "Saskatchewan Spruce"),
    "CH-Oe1": (47.29, 7.73, "GRA", "Oensingen1"),
    "CN-HaM": (37.37, 101.18, "GRA", "Haibei Alpine Meadow"),
    "CZ-BK1": (49.50, 18.54, "ENF", "Bily Kriz"),
    "DE-Gri": (50.95, 13.51, "GRA", "Grillenburg"),
    "DE-Hai": (51.08, 10.45, "DBF", "Hainich"),
    "DE-Tha": (50.96, 13.57, "ENF", "Tharandt"),
    "DK-Sor": (55.49, 11.64, "DBF", "Soroe"),
    "ES-LMa": (39.94, -5.77, "SAV", "Las Majadas"),
    "FI-Hyy": (61.85, 24.29, "ENF", "Hyytiala"),
    "FI-Sod": (67.36, 26.64, "ENF", "Sodankyla"),
    "FR-Gri": (48.84, 1.95, "CRO", "Grignon"),
    "FR-LBr": (44.72, -0.77, "ENF", "Le Bray"),
    "FR-Pue": (43.74, 3.60, "EBF", "Puechabon"),
    "GF-Guy": (5.28, -52.92, "EBF", "Guyaflux"),
    "GH-Ank": (5.27, -2.69, "EBF", "Ankasa"),
    "IT-Col": (41.85, 13.59, "DBF", "Collelongo"),
    "IT-Cpz": (41.71, 12.38, "EBF", "Castelporziano"),
    "IT-MBo": (46.01, 11.05, "GRA", "Monte Bondone"),
    "IT-Ren": (46.59, 11.43, "ENF", "Renon"),
    "IT-SRo": (43.73, 10.28, "ENF", "San Rossore"),
    "JP-SMF": (35.26, 137.08, "MF", "Seto Mixed Forest"),
    "MY-PSO": (2.97, 102.31, "EBF", "Pasoh"),
    "NL-Loo": (52.17, 5.74, "ENF", "Loobos"),
    "RU-Fyo": (56.46, 32.92, "ENF", "Fyodorovskoye"),
    "SD-Dem": (13.28, 30.48, "SAV", "Demokeya"),
    "SN-Dhr": (15.40, -15.43, "SAV", "Dahra"),
    "US-ARM": (36.61, -97.49, "CRO", "ARM SGP"),
    "US-Atq": (70.47, -157.41, "WET", "Atqasuk"),
    "US-Bar": (44.06, -71.29, "DBF", "Bartlett"),
    "US-Blo": (38.90, -120.63, "ENF", "Blodgett"),
    "US-Bo1": (40.01, -88.29, "CRO", "Bondville"),
    "US-Ha1": (42.54, -72.17, "DBF", "Harvard Forest"),
    "US-MMS": (39.32, -86.41, "DBF", "Morgan Monroe"),
    "US-Me2": (44.45, -121.56, "ENF", "Metolius"),
    "US-NR1": (40.03, -105.55, "ENF", "Niwot Ridge"),
    "US-Ne1": (41.17, -96.48, "CRO", "Mead Irrigated"),
    "US-Ne2": (41.16, -96.47, "CRO", "Mead Irrigated Rotation"),
    "US-Ne3": (41.18, -96.44, "CRO", "Mead Rainfed"),
    "US-PFa": (45.95, -90.27, "MF", "Park Falls"),
    "US-SRM": (31.82, -110.87, "WSA", "Santa Rita Mesquite"),
    "US-Ton": (38.43, -120.97, "WSA", "Tonzi Ranch"),
    "US-UMB": (45.56, -84.71, "DBF", "UMBS"),
    "US-Var": (38.41, -120.95, "GRA", "Vaira Ranch"),
    "US-WCr": (45.81, -90.08, "DBF", "Willow Creek"),
    "US-Wkg": (31.74, -109.94, "GRA", "Walnut Gulch Kendall"),
    "ZA-Kru": (-25.02, 31.50, "SAV", "Kruger"),
    "ZM-Mon": (-15.44, 28.25, "DBF", "Mongu"),
}


def get_site_metadata(site_id: str, metadata_df: Optional[pd.DataFrame] = None) -> dict:
    """Get lat, lon, IGBP class for a site from metadata or known list."""
    if metadata_df is not None:
        row = metadata_df[metadata_df["site_id"] == site_id]
        if len(row) > 0:
            r = row.iloc[0]
            return {
                "latitude": r.get("latitude", np.nan),
                "longitude": r.get("longitude", np.nan),
                "igbp": r.get("igbp", "UNK"),
            }

    if site_id in KNOWN_SITES:
        lat, lon, igbp, _ = KNOWN_SITES[site_id]
        return {"latitude": lat, "longitude": lon, "igbp": igbp}

    return {"latitude": np.nan, "longitude": np.nan, "igbp": "UNK"}


def build_site_summary(
    data_dir: str | Path,
    min_valid_months: int = 24,
) -> pd.DataFrame:
    """
    Process all FLUXNET sites and return a summary DataFrame.

    Columns:
        site_id, latitude, longitude, igbp, biome_group,
        bowen_ratio, bowen_std, bowen_median,
        mean_H, mean_LE, mean_netrad, mean_TA,
        n_valid_months, total_months
    """
    data_dir = Path(data_dir)
    all_files = discover_fluxnet_files(data_dir)

    # Prefer DD (daily) files over HH (half-hourly) — faster to process, same result
    dd_files = [f for f in all_files if "_DD_" in f.name]
    mm_files = [f for f in all_files if "_MM_" in f.name]
    files = mm_files if mm_files else (dd_files if dd_files else all_files)

    if not files:
        logger.error(
            f"No FLUXNET files found in {data_dir}. "
            "Download from https://fluxnet.org/data/fluxnet2015-dataset/ "
            "and place monthly (MM) CSV files in this directory."
        )
        return pd.DataFrame()

    metadata_df = load_site_metadata(data_dir)

    records = []
    for filepath in files:
        site_id = extract_site_id(filepath)
        logger.info(f"Processing {site_id}...")

        df = load_site_data(filepath)
        if df is None:
            continue

        stats = compute_bowen_ratio(df)
        if stats is None:
            logger.info(f"  {site_id}: insufficient valid data, skipping")
            continue

        if stats["n_valid_months"] < min_valid_months:
            logger.info(
                f"  {site_id}: only {stats['n_valid_months']} valid months "
                f"(need {min_valid_months}), skipping"
            )
            continue

        meta = get_site_metadata(site_id, metadata_df)

        record = {"site_id": site_id, **meta, **stats}
        records.append(record)
        logger.info(
            f"  {site_id}: β = {stats['bowen_ratio']:.2f} "
            f"({stats['n_valid_months']} months, IGBP={meta['igbp']})"
        )

    summary = pd.DataFrame(records)

    if len(summary) == 0:
        logger.warning("No valid sites found.")
        return summary

    # Add biome group classification
    forest_types = {"ENF", "EBF", "DNF", "DBF", "MF"}
    shrub_types = {"CSH", "OSH"}
    savanna_types = {"WSA", "SAV"}

    def classify_biome(igbp):
        if igbp in forest_types:
            return "Forest"
        elif igbp in shrub_types:
            return "Shrubland"
        elif igbp in savanna_types:
            return "Savanna"
        elif igbp == "GRA":
            return "Grassland"
        elif igbp == "WET":
            return "Wetland"
        elif igbp in {"CRO", "CVM"}:
            return "Cropland"
        elif igbp == "URB":
            return "Urban"
        elif igbp in {"BSV", "SNO"}:
            return "Barren"
        return "Other"

    summary["biome_group"] = summary["igbp"].apply(classify_biome)

    logger.info(
        f"\nSummary: {len(summary)} sites processed.\n"
        f"Bowen ratio range: {summary['bowen_ratio'].min():.2f} – "
        f"{summary['bowen_ratio'].max():.2f}\n"
        f"Biome groups: {summary['biome_group'].value_counts().to_dict()}"
    )

    return summary.sort_values("bowen_ratio").reset_index(drop=True)
