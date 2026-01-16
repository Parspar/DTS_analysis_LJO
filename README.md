# DTS Analysis - LJO Catchment

This repository contains analysis scripts for Distributed Temperature Sensing (DTS) data from the LJO catchment in Northern Finland.

## Overview

The scripts analyze DTS temperature measurements along a stream to identify:
- **Groundwater (GW) inflows**: Identified by seasonal slope contrasts in temperature profiles
- **Surface water (SW) inflows**: Identified by melt-season slope variability and cold anomalies
- **Hydrological connectivity**: Using upslope contributing areas (UCAs) and topographic wetness index (TWI)

## Repository Structure

```
DTS_analysis_LJO/
├── DTS GW & SW Classifications.ipynb  # Main classification script
├── Final model results.ipynb          # Model output visualization
├── UCAs & TWI.ipynb                   # Topographic analysis
├── Climatology.ipynb                  # Climatology and forcing data visualization
├── GW_inflow_locations_slope.csv      # Groundwater inflow zones
├── SW_inflow_locations_slope.csv      # Surface water inflow zones
├── utils.py                           # Utility functions
├── README.md                          # This file
└── data/                              # Data directory (not in repo)
    ├── DTS/
    │   └── pallas_dts_data_f_6.nc    # DTS temperature measurements
    ├── forcing/
    │   ├── 30_min_interval_discharge.csv
    │   ├── snow_depth_kittila.csv
    │   ├── precipitation_lompolonvuoma_2021_2024.csv
    │   └── air_temperature_lompolonvuoma_2021_2024.csv
    ├── model_results/                 # SpaFHy-2D model outputs
    │   └── 202601021751_with_inflow.nc
    └── WBT_data/                      # WhiteboxTools geospatial data
        ├── pallas_8/                  # 8m DEM products
        ├── pallas_16/                 # 16m DEM products
        │   └── stream_lengths_burned_clipped.tif
        ├── flow_accumulation_cleaned_data.csv
        └── br_elev_20250715_out/      # Bedrock elevation analysis
            └── flowacc_out/
                └── uca_profiles/
```

## Data Requirements

### Required Data Files

The scripts require the following data files, which should be placed in a `./data/` directory:

1. **DTS Temperature Data**
   - `pallas_dts_data_f_6.nc` - NetCDF file with distributed temperature measurements
   - Time coverage: 2021-06-15 to 2024-09-25 (1198 days)

2. **Forcing Data**
   - `30_min_interval_discharge.csv` - Catchment discharge (30-min intervals)
   - `snow_depth_kittila.csv` - Snow depth observations
   - `precipitation_lompolonvuoma_2021_2024.csv` - Precipitation time series
   - `air_temperature_lompolonvuoma_2021_2024.csv` - Air temperature time series

3. **Model Results**
   - `model_results/202601021751_with_inflow.nc` - SpaFHy-2D hydrological model outputs

4. **Geospatial Data (WhiteboxTools)**
   - Stream network rasters
   - Flow accumulation grids
   - TWI profiles for 8m and 16m DEMs
   - Bedrock UCA profiles

### Data Not Included

Due to file size constraints, the data files are **not included in this repository**. You will need to:
1. Create a `data/` directory in the repository root
2. Place your data files following the structure above
3. Update the path variables at the top of each notebook if needed

## Installation

### Requirements

```bash
# Create conda environment
conda create -n dts-analysis python=3.9
conda activate dts-analysis

# Install required packages
pip install xarray netcdf4 numpy pandas matplotlib seaborn rasterio
```

## Usage

### 1. Classify GW and SW Inflows

Run `DTS GW & SW Classifications.ipynb` to:
- Load DTS temperature data
- Identify groundwater inflow zones using seasonal slope contrasts
- Identify surface water inflows using melt-season signatures
- Export classified zones to CSV files

**Outputs:**
- `GW_inflow_locations_slope.csv`
- `SW_inflow_locations_slope.csv`

### 2. Visualize Model Results

Run `Final model results.ipynb` to:
- Load SpaFHy-2D model outputs
- Plot groundwater inflow along the stream network
- Compare model predictions with DTS observations

### 3. Analyze Topographic Controls

Run `UCAs & TWI.ipynb` to:
- Calculate upslope contributing areas (surface and subsurface)
- Compute topographic wetness index (TWI)
- Overlay with GW/SW classification results

### 4. Visualize Climatology and Forcing Data

Run `Climatology.ipynb` to:
- Plot discharge (Q), precipitation (P), and snow depth (SD) time series
- Visualize air temperature and stream temperature relationships
- Create publication-quality climatology figures

## Methods

### Groundwater Inflow Classification

GW inflows are identified by **seasonal slope reversal**:
- **Winter**: dT/dx < 0 (warming downstream due to GW input)
- **Summer**: dT/dx > 0 (cooling downstream as stream warms faster than GW)

### Surface Water Inflow Classification

SW inflows are identified during snowmelt by:
- High temperature variability (unstable melt-driven input)
- Cold temperature anomalies (cold meltwater signature)
- Strong positive temperature gradients

### Esker Zone

The analysis highlights an **esker region** (1000-1300 m along stream) - a glacial landform that influences groundwater flow patterns.

## Configuration

Each notebook contains a configuration section at the top:

```python
# ============================================================
# DATA PATHS - Update these paths for your local system
# ============================================================
DATA_BASE_DIR = "./data"  # Modify if your data is elsewhere
```

Update these paths if your data is stored in a different location.

## Output Files

- `GW_inflow_locations_slope.csv` - Groundwater inflow segment locations
  - Columns: `x_start_m`, `x_end_m`, `x_mid_m`, `length_m`, `method`, `intersects_esker`
  
- `SW_inflow_locations_slope.csv` - Surface water inflow segment locations
  - Columns: `x_start_m`, `x_end_m`, `x_mid_m`, `length_m`, `method`

## Citation

If you use this code, please cite:

```
[Your paper citation here]
```

## Contact

For questions or issues, please contact:
- **Author**: [Your name]
- **Email**: [Your email]
- **Institution**: University of Oulu

## License

[Specify your license here, e.g., MIT, GPL, etc.]

---

**Note**: This analysis is part of ongoing research on stream-aquifer interactions in northern Finland. The methods are designed for DTS measurements along small headwater streams in subarctic/boreal environments.

