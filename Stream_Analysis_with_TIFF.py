"""
STREAM CLASS ANALYSIS USING TIFF CLASSIFICATION
================================================
Analyzes groundwater inflow by stream type using pre-classified TIFF file
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import rasterio
from rasterio.warp import reproject, Resampling
from matplotlib.colors import ListedColormap, BoundaryNorm

# ==============================================================================
# STEP 1: LOAD MODEL DATA
# ==============================================================================

print("=" * 70)
print("  STEP 1: LOADING MODEL RESULTS")
print("=" * 70)

nc_file = r"C:\Users\pparvizi24\OneDrive - University of Oulu and Oamk\Parsa-PHD-OneDrive\DTS\Input\spafhy_input\pallas\results\final_run_2_2D_202511261633\202511261633.nc"
results = xr.open_dataset(nc_file)

print(f"\n✓ Loaded model: {nc_file.split('\\')[-1]}")
print(f"  Time: {pd.Timestamp(results.time.values[0]).date()} to {pd.Timestamp(results.time.values[-1]).date()}")
print(f"  Grid shape: {results['deep_netflow_to_ditch'].shape[1]} × {results['deep_netflow_to_ditch'].shape[2]}")
print(f"  Lat range: {float(results.lat.min()):.6f} to {float(results.lat.max()):.6f}")
print(f"  Lon range: {float(results.lon.min()):.6f} to {float(results.lon.max()):.6f}")

# ==============================================================================
# STEP 2: LOAD TIFF CLASSIFICATION
# ==============================================================================

print("\n" + "=" * 70)
print("  STEP 2: LOADING TIFF STREAM CLASSIFICATION")
print("=" * 70)

tiff_file = r"C:\Users\pparvizi24\OneDrive - University of Oulu and Oamk\Documents\ArcGIS\Projects\SpaFHy-Pallas\LJOSPAFHYnetwork.tif"

with rasterio.open(tiff_file) as src:
    tiff_data = src.read(1)
    tiff_transform = src.transform
    tiff_crs = src.crs
    tiff_bounds = src.bounds
    
    print(f"\n✓ Loaded TIFF: {tiff_file.split('\\')[-1]}")
    print(f"  Shape: {tiff_data.shape}")
    print(f"  CRS: {tiff_crs}")
    print(f"  Bounds: {tiff_bounds}")
    print(f"  Transform: {tiff_transform}")
    
    # Check unique values in TIFF
    unique_vals = np.unique(tiff_data[~np.isnan(tiff_data)])
    print(f"\n  Unique classification values in TIFF:")
    for val in unique_vals:
        count = np.sum(tiff_data == val)
        print(f"    Value {int(val):3d}: {count:6d} pixels")

# ==============================================================================
# STEP 3: CHECK AND ALIGN GRIDS
# ==============================================================================

print("\n" + "=" * 70)
print("  STEP 3: CHECKING GRID ALIGNMENT")
print("=" * 70)

# Get model grid info
model_shape = (len(results.lat), len(results.lon))
model_lats = results.lat.values
model_lons = results.lon.values

print(f"\nModel grid: {model_shape}")
print(f"TIFF grid:  {tiff_data.shape}")

# Check if shapes match
if tiff_data.shape == model_shape:
    print("\n✓ Grids have same shape! Checking coordinate alignment...")
    
    # For now, assume they align if shapes match
    # You can add more sophisticated coordinate checking here
    stream_classes = tiff_data.copy()
    print("✓ Using TIFF data directly")
    
else:
    print(f"\n⚠ Grid mismatch detected!")
    print(f"   Model: {model_shape}")
    print(f"   TIFF:  {tiff_data.shape}")
    print("\n   Need to resample TIFF to match model grid...")
    
    # Create target grid matching model
    from rasterio.transform import from_bounds
    
    # Get model bounds from lat/lon
    model_bounds = (
        float(model_lons.min()),  # left
        float(model_lats.min()),  # bottom
        float(model_lons.max()),  # right
        float(model_lats.max())   # top
    )
    
    # Create transform for target grid
    target_transform = from_bounds(
        *model_bounds,
        model_shape[1],  # width
        model_shape[0]   # height
    )
    
    # Prepare arrays for reprojection
    stream_classes = np.zeros(model_shape, dtype=tiff_data.dtype)
    
    # Reproject using nearest neighbor (preserves class values)
    reproject(
        source=tiff_data,
        destination=stream_classes,
        src_transform=tiff_transform,
        src_crs=tiff_crs,
        dst_transform=target_transform,
        dst_crs=tiff_crs,  # Assuming same CRS
        resampling=Resampling.nearest
    )
    
    print(f"   ✓ Resampled to {stream_classes.shape}")

# ==============================================================================
# STEP 4: VERIFY CLASSIFICATION
# ==============================================================================

print("\n" + "=" * 70)
print("  STEP 4: VERIFYING STREAM CLASSIFICATION")
print("=" * 70)

# Get model's stream locations
model_streams = results['parameters_streams'].values
stream_pixels = np.sum(model_streams < 0)

print(f"\nModel stream pixels: {stream_pixels}")
print(f"\nClassification pixel counts:")

unique_classes = np.unique(stream_classes[~np.isnan(stream_classes)])
for cls in unique_classes:
    count = np.sum(stream_classes == cls)
    print(f"  Class {int(cls)}: {count:6d} pixels")

# Mask classification to only model stream locations
stream_classes_masked = stream_classes.copy()
stream_classes_masked[model_streams >= 0] = 0  # Set non-stream pixels to 0

print(f"\n✓ Classification masked to model streams:")
for cls in unique_classes:
    if cls == 0:
        continue
    count = np.sum(stream_classes_masked == cls)
    print(f"  Class {int(cls)}: {count:6d} pixels")

# ==============================================================================
# STEP 5: CREATE INFLOW VARIABLE
# ==============================================================================

print("\n" + "=" * 70)
print("  STEP 5: CREATING INFLOW-ONLY VARIABLE")
print("=" * 70)

netflow = results['deep_netflow_to_ditch']
print(f"\nOriginal netflow: min={float(netflow.min()):.3f}, max={float(netflow.max()):.3f} mm/d")

inflow_to_ditch = netflow.where(netflow > 0, 0)
print(f"Filtered inflow: min={float(inflow_to_ditch.min()):.3f}, max={float(inflow_to_ditch.max()):.3f} mm/d")

results['inflow_to_ditch'] = inflow_to_ditch
print("✓ Created 'inflow_to_ditch' variable")

# ==============================================================================
# STEP 6: VISUALIZE CLASSIFICATION
# ==============================================================================

print("\n" + "=" * 70)
print("  STEP 6: VISUALIZING CLASSIFICATION")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Model streams
ax1 = axes[0]
im1 = ax1.imshow(model_streams, cmap='Blues_r')
plt.colorbar(im1, ax=ax1, label='Stream Depth [m]')
ax1.set_title('(a) Model Stream Depths', fontweight='bold', fontsize=12)

# Original TIFF classification
ax2 = axes[1]
colors = ['white', 'blue', 'green', 'red', 'orange', 'purple']
cmap = ListedColormap(colors[:len(unique_classes)])
im2 = ax2.imshow(stream_classes, cmap=cmap)
plt.colorbar(im2, ax=ax2, label='Class')
ax2.set_title('(b) TIFF Classification (Original)', fontweight='bold', fontsize=12)

# Masked classification (only on model streams)
ax3 = axes[2]
im3 = ax3.imshow(stream_classes_masked, cmap=cmap)
plt.colorbar(im3, ax=ax3, label='Class')
ax3.set_title('(c) Classification (Masked to Model Streams)', fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('stream_classification_alignment.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved: stream_classification_alignment.png")

# ==============================================================================
# STEP 7: USER INPUT FOR CLASS MAPPING
# ==============================================================================

print("\n" + "=" * 70)
print("  STEP 7: DEFINE CLASS MAPPING")
print("=" * 70)

print("\n⚠ IMPORTANT: Please define which TIFF values correspond to each stream type")
print("   Looking at the TIFF, unique values are:", [int(x) for x in unique_classes if x != 0])
print("\n   Please update the code with the correct mapping:")
print("   Example: If 1=Main, 2=Tributary, 3=Ditch, set:")
print("     main_class_value = 1")
print("     trib_class_value = 2")
print("     ditch_class_value = 3")

# USER: UPDATE THESE VALUES BASED ON YOUR TIFF
main_class_value = 1   # ← UPDATE THIS
trib_class_value = 2   # ← UPDATE THIS
ditch_class_value = 3  # ← UPDATE THIS

print(f"\n✓ Using mapping:")
print(f"  Main Stream  = TIFF value {main_class_value}")
print(f"  Tributary    = TIFF value {trib_class_value}")
print(f"  Ditch        = TIFF value {ditch_class_value}")

# ==============================================================================
# STEP 8: EXTRACT INFLOWS BY CLASS
# ==============================================================================

print("\n" + "=" * 70)
print("  STEP 8: EXTRACTING INFLOWS BY CLASS")
print("=" * 70)

# Use masked classification
main_mask = (stream_classes_masked == main_class_value)
trib_mask = (stream_classes_masked == trib_class_value)
ditch_mask = (stream_classes_masked == ditch_class_value)

print(f"\nMask pixel counts:")
print(f"  Main Stream:  {np.sum(main_mask)} pixels")
print(f"  Tributary:    {np.sum(trib_mask)} pixels")
print(f"  Ditch:        {np.sum(ditch_mask)} pixels")

# Extract inflows
main_inflow = results['inflow_to_ditch'].where(main_mask)
trib_inflow = results['inflow_to_ditch'].where(trib_mask)
ditch_inflow = results['inflow_to_ditch'].where(ditch_mask)

# Calculate totals and means
main_total = main_inflow.sum(dim=['lat', 'lon'], skipna=True)
trib_total = trib_inflow.sum(dim=['lat', 'lon'], skipna=True)
ditch_total = ditch_inflow.sum(dim=['lat', 'lon'], skipna=True)

main_mean = main_inflow.mean(dim=['lat', 'lon'], skipna=True)
trib_mean = trib_inflow.mean(dim=['lat', 'lon'], skipna=True)
ditch_mean = ditch_inflow.mean(dim=['lat', 'lon'], skipna=True)

print("✓ Extracted time series for each class")

# ==============================================================================
# STEP 9: SUMMARY STATISTICS
# ==============================================================================

print("\n" + "=" * 70)
print("  STEP 9: SUMMARY STATISTICS")
print("=" * 70)

stats = pd.DataFrame({
    'Stream Class': ['Main Stream', 'Tributary', 'Ditch'],
    'Pixels': [int(main_mask.sum()), int(trib_mask.sum()), int(ditch_mask.sum())],
    'Mean Inflow (mm/d)': [
        float(main_mean.mean()) if not np.isnan(main_mean.mean()) else 0,
        float(trib_mean.mean()) if not np.isnan(trib_mean.mean()) else 0,
        float(ditch_mean.mean()) if not np.isnan(ditch_mean.mean()) else 0
    ],
    'Max Inflow (mm/d)': [
        float(main_mean.max()) if not np.isnan(main_mean.max()) else 0,
        float(trib_mean.max()) if not np.isnan(trib_mean.max()) else 0,
        float(ditch_mean.max()) if not np.isnan(ditch_mean.max()) else 0
    ],
    'Total Cumulative': [
        float(main_total.sum()) if not np.isnan(main_total.sum()) else 0,
        float(trib_total.sum()) if not np.isnan(trib_total.sum()) else 0,
        float(ditch_total.sum()) if not np.isnan(ditch_total.sum()) else 0
    ]
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
# STEP 10: TIME SERIES PLOTS
# ==============================================================================

print("\n" + "=" * 70)
print("  STEP 10: CREATING TIME SERIES PLOTS")
print("=" * 70)

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
time_dates = pd.to_datetime(results.time.values)

# Mean inflow intensity
ax1 = axes[0]
if np.sum(main_mask) > 0:
    ax1.plot(time_dates, main_mean, 'b-', label='Main', lw=2)
if np.sum(trib_mask) > 0:
    ax1.plot(time_dates, trib_mean, 'g-', label='Tributary', lw=2)
if np.sum(ditch_mask) > 0:
    ax1.plot(time_dates, ditch_mean, 'r-', label='Ditch', lw=2)
ax1.set_ylabel('Mean Inflow [mm/d]', fontsize=13)
ax1.set_title('Inflow Intensity by Stream Class', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Total contribution
ax2 = axes[1]
if np.sum(main_mask) > 0:
    ax2.plot(time_dates, main_total, 'b-', label='Main', lw=2)
if np.sum(trib_mask) > 0:
    ax2.plot(time_dates, trib_total, 'g-', label='Tributary', lw=2)
if np.sum(ditch_mask) > 0:
    ax2.plot(time_dates, ditch_total, 'r-', label='Ditch', lw=2)
ax2.set_ylabel('Total Inflow [mm/d × pixels]', fontsize=13)
ax2.set_xlabel('Date', fontsize=13)
ax2.set_title('Total Contribution by Stream Class', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('stream_class_timeseries.png', dpi=150, bbox_inches='tight')
plt.show()
print("✓ Saved: stream_class_timeseries.png")

# ==============================================================================
# STEP 11: EXPORT RESULTS
# ==============================================================================

print("\n" + "=" * 70)
print("  STEP 11: EXPORTING RESULTS")
print("=" * 70)

# Time series
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

# Also save the aligned classification as numpy array for future use
np.save('stream_classes_aligned.npy', stream_classes_masked)
print("✓ Saved: stream_classes_aligned.npy")

print("\n" + "=" * 70)
print("  ✓✓✓ ANALYSIS COMPLETE! ✓✓✓")
print("=" * 70)
print("\n📌 NEXT STEPS:")
print("   1. Check the plots to verify alignment")
print("   2. Update class mapping values if needed (Step 7)")
print("   3. Use the CSV files for further analysis")
print("=" * 70)
