"""Paper D v3: Full analysis on 313 sites with ERA5 causal chain."""
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

import pandas as pd
import numpy as np
from scipy import stats
from numpy.linalg import lstsq
import xarray as xr

df = pd.read_csv('outputs/toa_tables/site_summary_with_toa.csv')
print(f'Sites: {len(df)}')
print(f'Biomes: {df["biome_group"].value_counts().to_dict()}')
print()

alpha = 'forcing_proxy'
cre = 'ceres_toa_cre_net_mean'

# Pixel-averaged
df2 = df.copy()
df2['px'] = np.round(df2['latitude']).astype(str) + '_' + np.round(df2['longitude']).astype(str)
mask = df2[alpha].notna() & df2[cre].notna()
pm = df2[mask].groupby('px').agg({alpha: 'mean', cre: 'mean'}).reset_index()
s, i, r, p, se = stats.linregress(pm[alpha], pm[cre])
print(f'Pixel-averaged: gamma={s:.4f}, R2={r**2:.4f}, p={p:.2e}, n={len(pm)}')

# Heteroscedasticity
mask = df[alpha].notna() & df[cre].notna()
x, y = df.loc[mask, alpha].values, df.loc[mask, cre].values
s, i, r, p, se = stats.linregress(x, y)
resid = y - (s * x + i)
bp_s, bp_i, bp_r, bp_p, bp_se = stats.linregress(x, resid**2)
bp_stat = len(x) * bp_r**2
bp_pv = 1 - stats.chi2.cdf(bp_stat, 1)
print(f'Breusch-Pagan: stat={bp_stat:.2f}, p={bp_pv:.4f}')

n = len(x)
X = np.column_stack([np.ones(n), x])
h = np.sum(X * np.linalg.solve(X.T @ X, X.T).T, axis=1)
hc3_w = resid**2 / (1 - h)**2
bread = np.linalg.inv(X.T @ X)
sandwich = bread @ (X.T @ np.diag(hc3_w) @ X) @ bread
se_hc3 = np.sqrt(sandwich[1, 1])
t_hc3 = s / se_hc3
p_hc3 = 2 * stats.t.sf(abs(t_hc3), n - 2)
print(f'HC3 SE: {se_hc3:.4f} (OLS: {se:.4f}), HC3 p: {p_hc3:.2e}')

# Controls
print()
print('=== Controls ===')
for ctrl_name, ctrl_col in [('latitude', 'latitude'), ('R_net', 'mean_netrad'), ('altitude', 'ceres_toa_lw_clr_mean')]:
    mask = df[alpha].notna() & df[cre].notna() & df[ctrl_col].notna()
    c = df[mask]
    X2 = np.column_stack([c[ctrl_col].values, np.ones(len(c))])
    cx, _, _, _ = lstsq(X2, c[alpha].values, rcond=None)
    cy, _, _, _ = lstsq(X2, c[cre].values, rcond=None)
    xr2 = c[alpha].values - X2 @ cx
    yr2 = c[cre].values - X2 @ cy
    rp, pp = stats.pearsonr(xr2, yr2)
    print(f'  Controlling {ctrl_name}: r={rp:.4f}, R2={rp**2:.4f}, p={pp:.2e}')

mask = df[alpha].notna() & df[cre].notna() & df['latitude'].notna() & df['mean_netrad'].notna() & df['ceres_toa_lw_clr_mean'].notna()
c = df[mask]
X2 = np.column_stack([c[['latitude', 'mean_netrad', 'ceres_toa_lw_clr_mean']].values, np.ones(len(c))])
cx, _, _, _ = lstsq(X2, c[alpha].values, rcond=None)
cy, _, _, _ = lstsq(X2, c[cre].values, rcond=None)
xr2 = c[alpha].values - X2 @ cx
yr2 = c[cre].values - X2 @ cy
rp, pp = stats.pearsonr(xr2, yr2)
print(f'  Controlling ALL: r={rp:.4f}, R2={rp**2:.4f}, p={pp:.2e}')

# Wetland exclusion
nw = df[df['biome_group'] != 'Wetland']
mask = nw[alpha].notna() & nw[cre].notna()
s, i, r, p, se = stats.linregress(nw.loc[mask, alpha], nw.loc[mask, cre])
print(f'Excl wetlands: gamma={s:.4f}, R2={r**2:.4f}, p={p:.2e}, n={mask.sum()}')

# Seasonal
print()
print('=== Seasonal ===')
ds = xr.open_dataset('data/raw/ceres/CERES_EBAF_Edition4.2_200003-202407.nc', engine='h5netcdf')
time = pd.DatetimeIndex(ds['time'].values)
jja = time.month.isin([6, 7, 8])
djf = time.month.isin([12, 1, 2])

cre_jja, cre_djf = [], []
for _, row in df.iterrows():
    lat, lon = row['latitude'], row['longitude']
    if not (np.isfinite(lat) and np.isfinite(lon)):
        cre_jja.append(np.nan); cre_djf.append(np.nan); continue
    try:
        site = ds['toa_cre_net_mon'].sel(lat=lat, lon=lon, method='nearest', tolerance=0.5)
        v = site.values
        jv = v[jja]; dv = v[djf]
        cre_jja.append(float(np.nanmean(jv)) if np.any(np.isfinite(jv)) else np.nan)
        cre_djf.append(float(np.nanmean(dv)) if np.any(np.isfinite(dv)) else np.nan)
    except:
        cre_jja.append(np.nan); cre_djf.append(np.nan)

df['cre_jja'] = cre_jja
df['cre_djf'] = cre_djf

for season, col in [('JJA', 'cre_jja'), ('DJF', 'cre_djf'), ('Annual', cre)]:
    mask = df[alpha].notna() & df[col].notna()
    x2, y2 = df.loc[mask, alpha].values, df.loc[mask, col].values
    if len(x2) > 10:
        s, i, r, p, se = stats.linregress(x2, y2)
        print(f'  {season}: gamma={s:.4f}, R2={r**2:.4f}, p={p:.2e}, n={len(x2)}')

# ERA5 causal chain
print()
print('=== ERA5 causal chain ===')
ds1 = xr.open_dataset('data/raw/era5_extracted/data_stream-moda_stepType-avgua.nc')
ds2 = xr.open_dataset('data/raw/era5_extracted/data_stream-moda_stepType-avgad.nc')

cape_v, tcwv_v, blh_v, tp_v = [], [], [], []
for _, row in df.iterrows():
    lat, lon = row['latitude'], row['longitude']
    if not (np.isfinite(lat) and np.isfinite(lon)):
        cape_v.append(np.nan); tcwv_v.append(np.nan); blh_v.append(np.nan); tp_v.append(np.nan); continue
    try:
        s1 = ds1.sel(latitude=lat, longitude=lon, method='nearest', tolerance=1.0)
        s2 = ds2.sel(latitude=lat, longitude=lon, method='nearest', tolerance=1.0)
        cape_v.append(float(np.nanmean(s1['cape'].values)))
        tcwv_v.append(float(np.nanmean(s1['tcwv'].values)))
        blh_v.append(float(np.nanmean(s1['blh'].values)))
        tp_v.append(float(np.nanmean(s2['tp'].values)) * 1000 * 365)
    except:
        cape_v.append(np.nan); tcwv_v.append(np.nan); blh_v.append(np.nan); tp_v.append(np.nan)

df['era5_cape'] = cape_v
df['era5_tcwv'] = tcwv_v
df['era5_blh'] = blh_v
df['era5_precip'] = tp_v
print(f'ERA5 matched: {df["era5_cape"].notna().sum()}/{len(df)}')

# Direct
mask = df[alpha].notna() & df[cre].notna()
s_d, _, r_d, p_d, _ = stats.linregress(df.loc[mask, alpha], df.loc[mask, cre])
print(f'Direct alpha->CRE: R2={r_d**2:.4f}, p={p_d:.2e}')

# Causal links
for cn, cc in [('alpha->TCWV', 'era5_tcwv'), ('alpha->CAPE', 'era5_cape'),
               ('alpha->BLH', 'era5_blh'), ('alpha->Precip', 'era5_precip')]:
    mask = df[alpha].notna() & df[cc].notna()
    s, i, r, p, se = stats.linregress(df.loc[mask, alpha], df.loc[mask, cc])
    print(f'  {cn}: slope={s:.2f}, R2={r**2:.4f}, p={p:.2e}')

# Mediation
print()
print('=== Mediation ===')
for cn, cc in [('Precip', 'era5_precip'), ('BLH', 'era5_blh'), ('TCWV', 'era5_tcwv'), ('CAPE', 'era5_cape')]:
    mask = df[alpha].notna() & df[cre].notna() & df[cc].notna()
    c = df[mask]
    X2 = np.column_stack([c[cc].values, np.ones(len(c))])
    cx, _, _, _ = lstsq(X2, c[alpha].values, rcond=None)
    cy, _, _, _ = lstsq(X2, c[cre].values, rcond=None)
    xr2 = c[alpha].values - X2 @ cx
    yr2 = c[cre].values - X2 @ cy
    rp, pp = stats.pearsonr(xr2, yr2)
    med = (1 - rp**2 / r_d**2) * 100
    print(f'  Controlling {cn:8s}: partial R2={rp**2:.4f}, p={pp:.2e}, mediation={med:.1f}%')

# ALL
mask = df[alpha].notna() & df[cre].notna()
for c2 in ['era5_tcwv', 'era5_cape', 'era5_precip', 'era5_blh', 'latitude']:
    mask &= df[c2].notna()
c = df[mask]
X2 = np.column_stack([c[['era5_tcwv', 'era5_cape', 'era5_precip', 'era5_blh', 'latitude']].values, np.ones(len(c))])
cx, _, _, _ = lstsq(X2, c[alpha].values, rcond=None)
cy, _, _, _ = lstsq(X2, c[cre].values, rcond=None)
xr2 = c[alpha].values - X2 @ cx
yr2 = c[cre].values - X2 @ cy
rp, pp = stats.pearsonr(xr2, yr2)
print(f'  Controlling ALL ERA5+lat: partial R2={rp**2:.4f}, p={pp:.2e}, n={len(c)}')

# Corrected financials
print()
print('=== Corrected financials (313 sites) ===')
f_alpha = df.loc[df['biome_group'] == 'Forest', alpha].mean()
s_alpha = df.loc[df['biome_group'] == 'Shrubland', alpha].mean()
f_rnet = df.loc[df['biome_group'] == 'Forest', 'mean_netrad'].mean()
da = s_alpha - f_alpha
sd = da * f_rnet
gamma_cre = 8.4725
gamma_lw = 12.0109
toa_cre = gamma_cre * da
toa_total = (gamma_cre + gamma_lw) * da
fc = toa_cre / sd
ft = toa_total / sd
unadj = sd * 11.76
print(f'Forest alpha={f_alpha:.3f}, Shrub alpha={s_alpha:.3f}, delta={da:.3f}')
print(f'Forest R_net={f_rnet:.1f}, Surface diff={sd:.1f} W/m2')
print(f'Transfer CRE={fc:.3f} ({fc*100:.1f}%), Total={ft:.3f} ({ft*100:.1f}%)')
print(f'Unadjusted: ${unadj:.0f}/ha/yr')
print(f'CRE-adjusted: ${unadj*fc:.0f}/ha/yr')
print(f'Total-adjusted: ${unadj*ft:.0f}/ha/yr')

# Biome gammas
print()
print('=== Biome gammas ===')
for b, g in df.groupby('biome_group'):
    mask = g[alpha].notna() & g[cre].notna()
    if mask.sum() >= 5:
        s, i, r, p, se = stats.linregress(g.loc[mask, alpha], g.loc[mask, cre])
        print(f'  {b:12s}: gamma={s:+8.4f}, R2={r**2:.4f}, p={p:.2e}, n={mask.sum()}')

# Save enriched dataset
df.to_csv('outputs/toa_tables/site_summary_v3_full.csv', index=False)
print('\nSaved: outputs/toa_tables/site_summary_v3_full.csv')
