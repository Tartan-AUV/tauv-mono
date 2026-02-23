#!/usr/bin/env python3
"""
Thruster Sweep Visualizer
Generates per-voltage 2D plots, an all-voltage overlay, a 3D scatter,
a 3D matplotlib surface, and an interactive Plotly HTML.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import BallTree
import plotly.graph_objects as go

# =============================================================================
# CONFIG
# =============================================================================

DATA_FILE   = 'thruster_sweep.csv'
OUTPUT_DIR  = 'sweep_plots'
MAX_VOLTAGE = 19.0          # ignore data above this voltage
POLY_DEGREE = 3             # polynomial degree for surface fit
MAX_DIST    = 500           # mask grid cells further than this from real data
GRID_RES    = 60            # surface grid resolution

# Data selection: True = only settled (is_last) points, False = all samples
SURFACE_3D_SETTLED_ONLY = True
SQUASHED_SETTLED_ONLY   = True

# Outlier removal
IQR_FACTOR  = 2           # higher = less aggressive removal
IQR_BUCKETS = 100            # number of gain buckets for per-bucket IQR

# =============================================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load & filter ---
df = pd.read_csv(DATA_FILE)
df['rpm'] = df['rpm'].abs()
df = df[df['voltage'] <= MAX_VOLTAGE].copy()
df['voltage_floor'] = np.floor(df['voltage']).astype(int)

last_all = df[df['is_last'] == 1].copy().sort_values(['voltage', 'gain'])

SWEEP_COUNT_THRESHOLD = 4914

# Count how many times gain crosses the threshold going upward/downward
# by looking at consecutive rows where gain transitions from below to at/above
df_sorted = df.sort_values(['timestamp']).reset_index(drop=True)

pos_crossings = ((df_sorted['gain'].shift(1) < SWEEP_COUNT_THRESHOLD) & 
                 (df_sorted['gain'] >= SWEEP_COUNT_THRESHOLD)).sum()

neg_crossings = ((df_sorted['gain'].shift(1) > -SWEEP_COUNT_THRESHOLD) & 
                 (df_sorted['gain'] <= -SWEEP_COUNT_THRESHOLD)).sum()

sweep_count = pos_crossings + neg_crossings
print(f'Total sweeps: {sweep_count}  (pos: {pos_crossings}, neg: {neg_crossings})  ({len(df)} total samples, {len(last_all)} settled points)')

def remove_outliers(data, col='rpm', factor=IQR_FACTOR, buckets=IQR_BUCKETS):
    clean_parts = []
    data = data.copy()
    data['gain_bucket'] = pd.cut(data['gain'], bins=buckets)
    for _, group in data.groupby('gain_bucket', observed=True):
        if len(group) < 4:
            clean_parts.append(group)
            continue
        q1  = group[col].quantile(0.25)
        q3  = group[col].quantile(0.75)
        iqr = q3 - q1
        clean_parts.append(group[(group[col] >= q1 - factor * iqr) &
                                  (group[col] <= q3 + factor * iqr)])
    return pd.concat(clean_parts).drop(columns='gain_bucket')

# Data pools for 3D and squashed plots
data_3d       = last_all if SURFACE_3D_SETTLED_ONLY else remove_outliers(df.sort_values(['voltage', 'gain']))
data_squashed = last_all if SQUASHED_SETTLED_ONLY   else remove_outliers(df)

# =============================================================================
# 2D: Per-voltage scatter
# =============================================================================

for voltage, group in df.groupby('voltage_floor'):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(group['gain'], group['rpm'], alpha=0.2, s=10,
               color='steelblue', label='all samples')
    last = group[group['is_last'] == 1]
    ax.scatter(last['gain'], last['rpm'], alpha=0.9, s=40,
               color='red', zorder=5, label='settled (is_last)')
    ax.set_xlabel('Gain (raw, -8191 to 8191)')
    ax.set_ylabel('RPM (absolute)')
    ax.set_title(f'Gain vs RPM  —  ~{voltage}V')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fname = os.path.join(OUTPUT_DIR, f'sweep_{voltage}V.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {fname}  ({len(group)} samples, {len(last)} settled)')

# =============================================================================
# 2D: All voltages overlay
# =============================================================================

fig, ax = plt.subplots(figsize=(12, 7))
colors = plt.cm.plasma(np.linspace(0.1, 0.9, df['voltage_floor'].nunique()))
for (voltage, group), color in zip(df.groupby('voltage_floor'), colors):
    last = group[group['is_last'] == 1].sort_values('gain')
    ax.plot(last['gain'], last['rpm'], color=color, linewidth=1.5, label=f'{voltage}V')
    ax.scatter(last['gain'], last['rpm'], color=color, s=20, alpha=0.6)
ax.set_xlabel('Gain (raw, -8191 to 8191)')
ax.set_ylabel('RPM (absolute)')
ax.set_title('Gain vs RPM — All Voltages')
ax.legend(title='Voltage')
ax.grid(True, alpha=0.3)
fname = os.path.join(OUTPUT_DIR, 'sweep_all_voltages.png')
fig.savefig(fname, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {fname}')

# =============================================================================
# 3D: Scatter only (matplotlib)
# =============================================================================

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_3d['gain'], data_3d['voltage'], data_3d['rpm'],
           c=data_3d['voltage'], cmap='plasma', s=15, alpha=0.8)
ax.set_xlabel('Gain')
ax.set_ylabel('Voltage (V)')
ax.set_zlabel('RPM (absolute)')
ax.set_title('Gain vs Voltage vs RPM')
fname = os.path.join(OUTPUT_DIR, 'sweep_3d.png')
fig.savefig(fname, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {fname}')

# =============================================================================
# 3D: Surface fit (matplotlib)
# =============================================================================

def fit_surface_poly(data, gain_sign, degree=POLY_DEGREE, max_dist=MAX_DIST):
    """
    Fit a polynomial surface to one side (pos/neg) of gain data.
    Returns (G, V, RPM_surf) meshgrids or None on failure.
    gain_sign: +1 for positive side, -1 for negative side.
    """
    if len(data) < 10:
        return None
    data = data.copy()
    data['gain_abs'] = data['gain'].abs()

    X = np.column_stack([data['gain_abs'].values, data['voltage'].values])
    y = data['rpm'].values

    poly  = PolynomialFeatures(degree=degree)
    model = LinearRegression().fit(poly.fit_transform(X), y)

    # Print equation
    side  = 'Positive' if gain_sign > 0 else 'Negative'
    terms = poly.get_feature_names_out(['gain', 'voltage'])
    r2    = model.score(poly.transform(X), y)
    print(f'\n{side} side equation (R²={r2:.4f}):')
    print(f'  RPM = {model.intercept_:.4f}')
    for coef, term in zip(model.coef_, terms):
        if abs(coef) > 1e-6:
            print(f'      + {coef:.6f} * {term}')

    gain_grid = np.linspace(0, data['gain_abs'].max(), GRID_RES)
    volt_grid = np.linspace(data['voltage'].min(), data['voltage'].max(), GRID_RES)
    G, V = np.meshgrid(gain_grid * gain_sign, volt_grid)

    X_grid   = np.column_stack([np.abs(G.ravel()), V.ravel()])
    RPM_surf = model.predict(poly.transform(X_grid)).reshape(G.shape)
    RPM_surf = np.clip(RPM_surf, 0, None)

    # Mask cells too far from real data
    tree = BallTree(X)
    dists, _ = tree.query(np.column_stack([np.abs(G.ravel()), V.ravel()]), k=1)
    RPM_surf[dists.reshape(G.shape) > max_dist] = np.nan
    predictions = model.predict(poly.transform(X))
    residuals   = y - predictions
    std_err     = np.std(residuals)
    mae         = np.mean(np.abs(residuals))
    print(f'  ± {std_err:.1f} RPM (1σ),  MAE = {mae:.1f} RPM')
    return G, V, RPM_surf


fig = plt.figure(figsize=(12, 8))
ax  = fig.add_subplot(111, projection='3d')
ax.scatter(data_3d['gain'], data_3d['voltage'], data_3d['rpm'],
           c=data_3d['voltage'], cmap='plasma', s=10, alpha=0.4)

result = fit_surface_poly(data_3d[data_3d['gain'] >= 0], gain_sign=1)
if result:
    G, V, R = result
    ax.plot_surface(G, V, R, alpha=0.5, color='royalblue', edgecolor='none')

result = fit_surface_poly(data_3d[data_3d['gain'] <= 0], gain_sign=-1)
if result:
    G, V, R = result
    ax.plot_surface(G, V, R, alpha=0.5, color='salmon', edgecolor='none')

ax.set_xlabel('Gain')
ax.set_ylabel('Voltage (V)')
ax.set_zlabel('RPM (absolute)')
ax.set_title('Gain vs Voltage vs RPM — Surface Fit')
fname = os.path.join(OUTPUT_DIR, 'sweep_3d_surface.png')
fig.savefig(fname, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {fname}')

# =============================================================================
# 3D: Interactive Plotly HTML
# =============================================================================

fig = go.Figure()

fig.add_trace(go.Scatter3d(
    x=data_3d['gain'],
    y=data_3d['voltage'],
    z=data_3d['rpm'],
    mode='markers',
    marker=dict(size=3, color=data_3d['voltage'], colorscale='Plasma', opacity=0.6),
    name='settled points'
))

def add_plotly_surface(data, gain_sign, color, name):
    result = fit_surface_poly(data, gain_sign)
    if result is None:
        return
    G, V, RPM_surf = result
    fig.add_trace(go.Surface(
        x=G, y=V, z=RPM_surf,
        colorscale=[[0, color], [1, color]],
        opacity=0.5,
        showscale=False,
        name=name
    ))

add_plotly_surface(data_3d[data_3d['gain'] >= 0], gain_sign=1,  color='royalblue', name='positive fit')
add_plotly_surface(data_3d[data_3d['gain'] <= 0], gain_sign=-1, color='salmon',    name='negative fit')

fig.update_layout(
    title='Gain vs Voltage vs RPM — Interactive',
    scene=dict(
        xaxis_title='Gain',
        yaxis_title='Voltage (V)',
        zaxis_title='RPM (absolute)',
    ),
    width=1200,
    height=800,
)

fname = os.path.join(OUTPUT_DIR, 'sweep_3d_interactive.html')
fig.write_html(fname)
print(f'Saved {fname} — open in browser to rotate')

# =============================================================================
# 2D: Squashed — all settled points on one plot, gradient by voltage, outliers removed
# =============================================================================

clean = data_squashed

fig, ax = plt.subplots(figsize=(12, 7))
sc = ax.scatter(
    clean['gain'], clean['rpm'],
    c=clean['voltage'], cmap='plasma',
    s=15, alpha=0.7, linewidths=0
)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Voltage (V)')
ax.set_xlabel('Gain (raw, -8191 to 8191)')
ax.set_ylabel('RPM (absolute)')
ax.set_title('Gain vs RPM — All Voltages (gradient colored)')
ax.grid(True, alpha=0.3)
fname = os.path.join(OUTPUT_DIR, 'sweep_squashed.png')
fig.savefig(fname, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f'Saved {fname}')
