# DTS Analysis of Groundwater and Surface Water Inflows — LJO Catchment-Northern Finland

**Author:** Parsa Parvizi  
**Affiliation:** University of Oulu, Finland  
**Contact:** parsa.parvizi@oulu.fi  
**Repository:** [github.com/Parspar/DTS_analysis_LJO](https://github.com/Parspar/DTS_analysis_LJO)

---

## Project

This repository contains the complete analysis pipeline for a study that uses **Distributed Temperature Sensing (DTS)** — fiber-optic cables that measure temperature continuously along a stream — to pinpoint groundwater and surface water inflow locations in a subarctic headwater stream in **Pallas, Northern Finland** (the LJO catchment).

The idea is simple: groundwater has a different temperature than surface water, and that difference flips between seasons. By tracking these seasonal temperature patterns along ~1940 m of stream, we can identify where groundwater enters (and where surface water through snowmelt-driven surface regime temperature analysis enters during spring).

## How the Detection Works

### Groundwater Inflows

Groundwater maintains a relatively stable temperature year-round. This creates a predictable pattern:

- **In winter**, groundwater is *warmer* than the cold stream, so the stream warms slightly downstream of an inflow (negative temperature gradient).
- **In summer**, groundwater is *cooler* than the warm stream, so the stream cools downstream of an inflow (positive temperature gradient).


### Surface Water Inflows

During snowmelt, cold meltwater enters the stream from the surrounding landscape. These inflows stand out by their:

- High temperature variability (pulses of melt-driven input)
- Cold anomalies relative to the stream
- Sharp positive temperature gradients


## Repository Structure

```
DTS_analysis_LJO/
│
├── DTS GW & SW Classifications.ipynb    # Main analysis — classifies GW and SW inflows
├── Final model results.ipynb            # Compares DTS observations with SpaFHy-2D model outputs
├── UCAs & TWI.ipynb                     # Topographic controls: upslope contributing areas & wetness index
├── Meteo and Model Sim.ipynb            # Meteorological forcing data and model simulation overview
├── 2D Heat Map.ipynb                    # Spatial-temporal temperature heatmaps
├── Flow ACC & TWI - Maps.ipynb          # Flow accumulation and TWI maps
├── Classified Stream LJO inflow to ditch for Monthly MEANs.ipynb
│                                        # Monthly mean analysis of stream-to-ditch inflows
├── Overland flow along stream.ipynb     # Overland flow analysis along the stream
├── Bedrock-Pallas-Flow acc.ipynb        # Bedrock-level flow accumulation analysis
├── comparsion of Forcings to raw data.ipynb  # Comparing forcing data with raw observations
│
├── utils.py                             # Shared utility functions (DTS I/O, plotting, grid reading)
│
├── GW_inflow_locations_slope.csv        # Output: classified groundwater inflow zones
├── SW_inflow_locations_slope.csv        # Output: classified surface water inflow zones
├── lateral_inflows_for_DTS.csv          # Lateral inflow data (ditch and tributary)
├── lateral_inflows_for_DTS_Monthly.csv  # Monthly lateral inflow data
├── distance_along_stream_DTS.csv        # Distance coordinates along the DTS cable
│
├── plotting_example.ipynb               # Example plotting routines
├── visu_tests.ipynb                     # Visualization experiments
├── process_files.ipynb                  # Data processing utilities
│
└── data/                                # Data directory (not in repo — see below)
```

### Where to Start

If you want to understand the core analysis, start with **`DTS GW & SW Classifications.ipynb`** — that's where the groundwater and surface water inflow classification happens. From there, `Final model results.ipynb` shows how the DTS findings compare to hydrological model predictions, and `UCAs & TWI.ipynb` explores the topographic reasons behind the observed inflow patterns.

## Data

The raw data files are too large for Git and are **not included** in this repository. To run the notebooks, you'll need to set up a `data/` folder with the following structure:

| Data | Path | Description |
|------|------|-------------|
| DTS measurements | `data/DTS/pallas_dts_data_f_6.nc` | NetCDF file, ~1940 m of stream, 2021–2024 |
| Discharge | `data/forcing/30_min_interval_discharge.csv` | 30-minute interval catchment discharge |
| Snow depth | `data/forcing/snow_depth_kittila.csv` | Daily snow depth from Kittila station |
| Precipitation | `data/forcing/precipitation_lompolonvuoma_2021_2024.csv` | Precipitation time series |
| Air temperature | `data/forcing/air_temperature_lompolonvuoma_2021_2024.csv` | Air temperature time series |
| Model results | `data/model_results/.nc` | SpaFHy-2D hydrological model outputs |
| Geospatial data | `data/WBT_data/` | DEMs, flow accumulation grids, TWI rasters (8 m & 16 m) |




## Installation

```bash
# Clone the repository
git clone https://github.com/Parspar/DTS_analysis_LJO.git
cd DTS_analysis_LJO

# Create a conda environment (recommended)
conda create -n dts-analysis python=3.9
conda activate dts-analysis

# Install dependencies
pip install xarray netcdf4 numpy pandas matplotlib seaborn rasterio
```

Then place your data files in the `data/` directory following the structure above, and you should be good to go.

## Key Outputs

The main classification notebook produces two CSV files:

**`GW_inflow_locations_slope.csv`** — Groundwater inflow zones:

| Column | Meaning |
|--------|---------|
| `x_start_m` | Start position along stream (m) |
| `x_end_m` | End position along stream (m) |
| `x_mid_m` | Midpoint of the segment (m) |
| `length_m` | Length of the inflow zone (m) |
| `method` | Detection method used |
| `intersects_esker` | Whether the segment overlaps with the esker zone |

**`SW_inflow_locations_slope.csv`** — Surface water inflow zones (same columns, minus `intersects_esker`).

## Utility Functions

The `utils.py` module provides reusable functions shared across notebooks:

- **`convert_ddf_to_monthly_csv`** — Converts raw DTS `.ddf` files into monthly CSV files
- **`read_and_combine_dts_files`** — Reads and merges DTS CSV files into a single pivoted DataFrame
- **`read_fmi_meteo_obs`** — Reads Finnish Meteorological Institute observation files
- **`plot_2D_dts_colormap`** — Creates 2D temperature heatmaps with meteorological context
- **`read_AsciiGrid`** — Reads ArcGIS ASCII grid files for geospatial data
- **`histogram_match`** — Computes histogram matching scores for model validation

## Study Area

The LJO catchment is a small headwater catchment near **Pallas** in Finnish Lapland. The DTS cable runs along approximately 1940 m of the main stream channel. The area features boreal/subarctic vegetation, seasonal snow cover (typically October–May), and a prominent esker formation that influences subsurface water flow. The measurement period spans from June 2021 to September 2024.

## Citation

If you use this code or build on this work, please cite:

```
[Citation will be added upon publication]
```

## License
 MIT

---

*This repository is part of PhD research at the University of Oulu on stream–aquifer interactions in subarctic catchments. If you have questions or ideas for collaboration, feel free to reach out.*
