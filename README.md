# Surface-to-TOA Radiative Transfer Coefficient

Analysis pipeline for **"Does biome-specific surface energy partitioning propagate to the top of atmosphere? Empirical evidence from co-located FLUXNET and CERES observations"** (Shahid, 2026).

## Summary

This paper tests whether the biome-specific surface forcing coefficient from [Shahid (2026)](https://github.com/R3GENESI5/biome-specific-forcing-coefficients) propagates to the top of atmosphere. We co-locate 314 FLUXNET flux tower sites with CERES EBAF Ed4.2 satellite observations and ERA5 reanalysis to measure the transfer coefficient.

## Key Results

- **314 flux tower sites** across 7 biomes, including BR-Sa1 (Santarem Km67), the most studied Amazon tower
- **CRE_SW gamma = 11.80 W/m2 per unit alpha** (R2 = 0.176, p = 8.9e-15)
- **Seasonal proof**: JJA gamma 14x DJF, consistent with convective coupling
- **Homogeneity filter**: R2 = 0.633 at strictest filter (n = 71)
- **ERA5 mediation**: precipitation mediates 54%, signal survives all controls (p = 8.4e-5)
- **Defensible transfer fraction**: 6-10% (CRE pathway)
- **Financial implication**: reduces Social Benefit of Forest Cooling from $402 to $84/ha/year using observed biome alpha ranges

## Relationship to Paper A

This paper builds on the biome-specific forcing coefficient alpha(beta) from:

> Shahid, A. B. (2026). Biome-specific radiative forcing coefficients reveal ecosystems as active climate regulators.
> Code: [github.com/R3GENESI5/biome-specific-forcing-coefficients](https://github.com/R3GENESI5/biome-specific-forcing-coefficients)
> DOI: [10.5281/zenodo.19328341](https://doi.org/10.5281/zenodo.19328341)

The shared source modules (`fluxnet_loader.py`, `ceres_loader.py`, `alpha_model.py`) are forked from Paper A's repository and included here for self-containment. This repository extends them with the following improvements:

### Improvements to shared modules

**`ceres_loader.py`** (extended from Paper A):
- Added 6 new CERES variables: cloud radiative effect (CRE SW/LW/net) and clear-sky TOA (SW/LW/net)
- Added pydap engine fallback for OpenDAP access via HTTPS
- Added derived CRE computation from all-sky minus clear-sky when direct CRE variables are unavailable
- Replaced the forcing proxy definition from `SFC_net / TOA_net` (which produces artifacts near TOA_net = 0) to `H / R_net` (the physically meaningful surface energy partitioning fraction)

**`fluxnet_loader.py`** (bugfix from Paper A):
- Fixed column selection to skip columns that exist but contain only NaN values (`if c in df.columns and df[c].notna().any()`). This recovers 58 additional sites (including GF-Guy, BR-Npw, ID-Pag) where `H_CORR` columns exist but are empty, allowing fallback to `H_F_MDS` gap-filled data.

## Data Sources (All Free, Public)

| Dataset | What it provides | How to get it |
|---|---|---|
| **FluxDataKit v3** | Sensible (H) and latent (LE) heat flux, net radiation | [Zenodo](https://doi.org/10.5281/zenodo.10885933) (automated download) |
| **CERES EBAF Ed4.2** | TOA radiation, cloud radiative effect | [NASA LaRC](https://ceres.larc.nasa.gov/data/) (requires Earthdata login) |
| **ERA5** | CAPE, column water vapour, boundary layer height, precipitation | [CDS](https://cds.climate.copernicus.eu/) (free account + API key) |
| **AmeriFlux** | BR-Sa1 (Santarem Km67) flux data | [AmeriFlux](https://ameriflux-data.lbl.gov/) (free registration) |

## Setup

```bash
pip install -r requirements.txt
```

## Running the Analysis

```bash
# Step 1: Download FluxDataKit (automated, ~6 GB)
python -m src.download_data

# Step 2: Download CERES EBAF from NASA (manual, ~2 GB)
# Save to: data/raw/ceres/CERES_EBAF_Edition4.2_200003-202407.nc

# Step 3: Run the full pipeline
python run_toa_analysis.py \
    --fluxnet-dir data/raw/fluxnet \
    --ceres-file data/raw/ceres/CERES_EBAF_Edition4.2_200003-202407.nc

# Step 4: Run robustness checks + ERA5 causal chain
python run_v3_analysis.py

# Step 5: Generate publication figures
python src/figures_v3.py

# Step 6: Run audit
python audit_final.py
```

## Project Structure

```
surface-toa-transfer/
├── run_toa_analysis.py       <- Main pipeline (Steps 1-11)
├── run_v3_analysis.py        <- Robustness + ERA5 + seasonal analysis
├── audit_final.py            <- Manuscript-data consistency audit
├── rebuild_final.py          <- Manuscript figure insertion
├── config_toa.yaml           <- Configuration
├── requirements.txt          <- Python dependencies
├── LICENSE                   <- MIT License
├── src/
│   ├── toa_transfer.py       <- Transfer coefficient regression, bootstrap, mediation
│   ├── robustness.py         <- Homogeneity, heteroscedasticity, controls, seasonal
│   ├── figures_v3.py         <- 9 publication figures (cartopy maps, scatter, bars)
│   ├── toa_visualize.py      <- Additional visualization functions
│   ├── homogeneity.py        <- MODIS land cover pixel homogeneity check
│   ├── ceres_loader.py       <- CERES EBAF data extraction (from Paper A)
│   ├── fluxnet_loader.py     <- FLUXNET data loading (from Paper A)
│   └── alpha_model.py        <- Alpha(beta) sigmoid fitting (from Paper A)
├── paper_d/
│   ├── manuscript/           <- Final DOCX and PDF
│   └── figures/              <- 9 figures in PNG (300 DPI) + PDF (vector)
└── data/                     <- Downloaded data (not tracked in git)
```

## Citation

```bibtex
@article{shahid2026surface,
  author  = {Shahid, Ali B.},
  title   = {Does biome-specific surface energy partitioning propagate to the
             top of atmosphere? {E}mpirical evidence from co-located {FLUXNET}
             and {CERES} observations},
  year    = {2026},
  note    = {Preprint}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgements

Analysis pipeline developed with assistance from Anthropic's Claude. All scientific decisions, interpretations, and conclusions are the sole work of the author.
