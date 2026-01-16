"""
CLEAN STREAM CLASS ANALYSIS
============================
Analyzes groundwater inflow by stream type (main, tributary, ditch)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
from matplotlib.colors import ListedColormap, BoundaryNorm

# ==============================================================================
# STEP 1: LOAD DATA
# ==============================================================================

print("=" * 70)
print("  STEP 1: LOADING MODEL RESULTS")
print("=" * 70)

nc_file = r"C:\Users\pparvizi24\OneDrive - University of Oulu and Oamk\Parsa-PHD-OneDrive\DTS\Input\spafhy_input\pallas\results\final_run_2_2D_202511261633\202511261633.nc"
results = xr.open_dataset(nc_file)

print(f"\n✓ Loaded: {nc_file}")
print(f"  Time: {pd.Timestamp(results.time.values[0]).date()} to {pd.Timestamp(results.time.values[-1]).date()}")
print(f"  Grid: {results['deep_netflow_to_ditch'].shape[1]} × {results['deep_netflow_to_ditch'].shape[2]}")

# ==============================================================================
# STEP 2: CREATE INFLOW VARIABLE (POSITIVE VALUES ONLY)
# ==============================================================================

print("\n" + "=" * 70)
print("  STEP 2: CREATING INFLOW-ONLY VARIABLE")
print("=" * 70)

netflow = results['deep_netflow_to_ditch']
print(f"\nOriginal netflow: min={float(netflow.min()):.3f}, max={float(netflow.max()):.3f} mm/d")

inflow_to_ditch = netflow.where(netflow > 0, 0)
print(f"Filtered inflow: min={float(inflow_to_ditch.min()):.3f}, max={float(inflow_to_ditch.max()):.3f} mm/d")

results['inflow_to_ditch'] = inflow_to_ditch
print("✓ Created 'inflow_to_ditch' variable")

# ==============================================================================
# STEP 3: CLASSIFY STREAMS BY DEPTH
# ==============================================================================

print("\n" + "=" * 70)
print("  STEP 3: CLASSIFYING STREAMS BY DEPTH")
print("=" * 70)

model_streams = results['parameters_streams'].values  # Shape: (221, 136)

print(f"\nStream info:")
print(f"  Grid: {model_streams.shape}")
print(f"  Stream pixels: {np.sum(model_streams < 0)}")
print(f"  Depth range: {model_streams[model_streams < 0].min():.3f} to {model_streams[model_streams < 0].max():.3f} m")

# Classification by depth
stream_classes = np.zeros_like(model_streams, dtype=int)

# ADJUST THESE THRESHOLDS:
stream_classes[(model_streams < 0) & (model_streams > -0.25)] = 3    # Shallow = Ditch
stream_classes[(model_streams <= -0.25) & (model_streams > -0.35)] = 2  # Medium = Tributary
stream_classes[model_streams <= -0.35] = 1                             # Deep = Main

print(f"\n✓ Classification:")
print(f"  Main (< -0.35 m):         {np.sum(stream_classes == 1)} pixels")
print(f"  Tributary (-0.35 to -0.25): {np.sum(stream_classes == 2)} pixels")
print(f"  Ditch (> -0.25 m):         {np.sum(stream_classes == 3)} pixels")

# ==============================================================================
# STEP 4: VISUALIZE CLASSIFICATION
# ==============================================================================

print("\n" + "=" * 70)
print("  STEP 4: VISUALIZING CLASSIFICATION")
print("=" * 70)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Stream depths
ax1 = axes[0]
im1 = ax1.imshow(model_streams, cmap='Blues_r')
plt.colorbar(im1, ax=ax1, label='Stream Depth [m]')
ax1.set_title('(a) Model Stream Depths', fontweight='bold')

# Classification
ax2 = axes[1]
colors = ['white', 'blue', 'green', 'red']
cmap = ListedColormap(colors)
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
im2 = ax2.imshow(stream_classes, cmap=cmap, norm=norm)
cbar = plt.colorbar(im2, ax=ax2, ticks=[0, 1, 2, 3])
cbar.set_ticklabels(['Background', 'Main', 'Tributary', 'Ditch'])
ax2.set_title('(b) Stream Classification', fontweight='bold')

plt.tight_layout()
plt.show()
print("✓ Visualization complete")

# ==============================================================================
# STEP 5: EXTRACT INFLOWS BY CLASS
# ==============================================================================

print("\n" + "=" * 70)
print("  STEP 5: EXTRACTING INFLOWS")
print("=" * 70)

main_mask = (stream_classes == 1)
trib_mask = (stream_classes == 2)
ditch_mask = (stream_classes == 3)

main_inflow = results['inflow_to_ditch'].where(main_mask)
trib_inflow = results['inflow_to_ditch'].where(trib_mask)
ditch_inflow = results['inflow_to_ditch'].where(ditch_mask)

main_total = main_inflow.sum(dim=['lat', 'lon'], skipna=True)
trib_total = trib_inflow.sum(dim=['lat', 'lon'], skipna=True)
ditch_total = ditch_inflow.sum(dim=['lat', 'lon'], skipna=True)

main_mean = main_inflow.mean(dim=['lat', 'lon'], skipna=True)
trib_mean = trib_inflow.mean(dim=['lat', 'lon'], skipna=True)
ditch_mean = ditch_inflow.mean(dim=['lat', 'lon'], skipna=True)

print("✓ Extracted time series for each class")

# ==============================================================================
# STEP 6: SUMMARY STATISTICS
# ==============================================================================

print("\n" + "=" * 70)
print("  STEP 6: SUMMARY STATISTICS")
print("=" * 70)

stats = pd.DataFrame({
    'Stream Class': ['Main Stream', 'Tributary', 'Ditch'],
    'Pixels': [int(main_mask.sum()), int(trib_mask.sum()), int(ditch_mask.sum())],
    'Mean Inflow (mm/d)': [float(main_mean.mean()), float(trib_mean.mean()), float(ditch_mean.mean())],
    'Max Inflow (mm/d)': [float(main_mean.max()), float(trib_mean.max()), float(ditch_mean.max())],
    'Total Cumulative': [float(main_total.sum()), float(trib_total.sum()), float(ditch_total.sum())]
})

print("\n" + stats.to_string(index=False))

total = stats['Total Cumulative'].sum()
if total > 0:
    print(f"\n{'─'*70}")
    print("RELATIVE CONTRIBUTIONS:")
    for idx, row in stats.iterrows():
        pct = (row['Total Cumulative'] / total) * 100
        print(f"  {row['Stream Class']:15s}: {pct:5.1f}%")
    print("─"*70)

# ==============================================================================
# STEP 7: TIME SERIES PLOTS
# ==============================================================================

print("\n" + "=" * 70)
print("  STEP 7: CREATING TIME SERIES PLOTS")
print("=" * 70)

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
time_dates = pd.to_datetime(results.time.values)

ax1 = axes[0]
ax1.plot(time_dates, main_mean, 'b-', label='Main', lw=2)
ax1.plot(time_dates, trib_mean, 'g-', label='Tributary', lw=2)
ax1.plot(time_dates, ditch_mean, 'r-', label='Ditch', lw=2)
ax1.set_ylabel('Mean Inflow [mm/d]', fontsize=13)
ax1.set_title('Inflow Intensity by Stream Class', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

ax2 = axes[1]
ax2.plot(time_dates, main_total, 'b-', label='Main', lw=2)
ax2.plot(time_dates, trib_total, 'g-', label='Tributary', lw=2)
ax2.plot(time_dates, ditch_total, 'r-', label='Ditch', lw=2)
ax2.set_ylabel('Total Inflow [mm/d × pixels]', fontsize=13)
ax2.set_xlabel('Date', fontsize=13)
ax2.set_title('Total Contribution by Stream Class', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()
print("✓ Plots complete")

# ==============================================================================
# STEP 8: EXPORT
# ==============================================================================

print("\n" + "=" * 70)
print("  STEP 8: EXPORTING RESULTS")
print("=" * 70)

export_ts = pd.DataFrame({
    'Date': time_dates,
    'Main_Mean': main_mean.values,
    'Trib_Mean': trib_mean.values,
    'Ditch_Mean': ditch_mean.values,
    'Main_Total': main_total.values,
    'Trib_Total': trib_total.values,
    'Ditch_Total': ditch_total.values
})

export_ts.to_csv('stream_class_timeseries.csv', index=False)
stats.to_csv('stream_class_statistics.csv', index=False)

print("✓ Saved: stream_class_timeseries.csv")
print("✓ Saved: stream_class_statistics.csv")

print("\n" + "=" * 70)
print("  ✓✓✓ ANALYSIS COMPLETE! ✓✓✓")
print("=" * 70)
