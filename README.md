# DTS Analysis — Groundwater & Surface Water Inflows in the LJO Catchment

**Author:** Parsa Parvizi · University of Oulu, Finland
**Contact:** parsa.parvizi@oulu.fi
**Repo:** [github.com/Parspar/DTS_analysis_LJO](https://github.com/Parspar/DTS_analysis_LJO)

---

## Project

Analysis of a fiber-optic Distributed Temperature Sensing (DTS) cable laid along ~2000 m of a small subarctic stream in **Pallas, Northern Finland** (the LJO headwater catchment, DTS corvering data 2021–2024).

The goal: locate where **groundwater (GW)** and **surface water (SW)** enter the stream, and check those locations against a SpaFHy-2D hydrological model and topographic predictors (UCA, TWI, flow accumulation).

## The idea behind the methods

Groundwater holds a fairly steady temperature year-round, while stream temperature swings with the seasons. That contrast flips sign between seasons, and DTS is sensitive enough to see it.

- **GW detection** — uses the *seasonal gradient-reversal* signature. Cold-season (25th-quantile) and warm-season (75th-quantile) profiles are compared along the stream. Where the two gradients reverse (cold profile cools downstream, warm profile warms downstream) and the local temperature sits in the 4–5 °C spring-water band, the cell is flagged as a GW zone.

- **SW detection** — uses the *melt-season mean profile*. A trailing upstream baseline is built (with the GW spring zone masked out), then locations are flagged where the stream is cooler than the baseline by ≥ 0.03 °C **and** the second spatial derivative is positive (a clear "dip" in the longitudinal profile).

SW methods filtered for night-time + active melt (snow > 0, Q > 0.2 m³/s). GW analysis has been done for the night time only so daytime solar heating doesn't contaminate the signal.

## Pipeline

Notebooks are numbered to be run in order. Each one is self-contained and produces figures/CSVs used by later steps.

| # | Notebook | What it does |
|---|---|---|
| 01 | `2D Heat Map` | Space–time temperature heatmap of the full DTS record |
| 02 | `Meteo and Model Sim` | Forcing data (Q, snow, precip, air T) overview |
| 03 | `GW-SW methods` | **Core analysis** — detects GW & SW zones, writes `GW_zones.csv` and `SW_zones.csv` |
| 04 | `UCAs & TWI` | Upslope contributing area + topographic wetness index along the stream |
| 05 | `Flow ACC & TWI – Maps` | Catchment maps of flow accumulation and TWI |
| 06 | `Classified Stream LJO inflow to ditch for Monthly MEANs` | Monthly mean inflow at the ditch confluence |
| 07 | `Final model results` | DTS observations vs. SpaFHy-2D model outputs |
| 08 | `Overland flow along stream` | Modelled overland flow along the stream, with GW/SW zones overlaid |

`DTS preprocessing/` holds the raw-file conversion utilities and `utils.py` (DTS I/O, FMI meteo readers, ASCII-grid reader, plotting helpers).

## Quick start

```bash
git clone https://github.com/Parspar/DTS_analysis_LJO.git
cd DTS_analysis_LJO

conda create -n dts-analysis python=3.9 -y
conda activate dts-analysis
pip install xarray netcdf4 numpy pandas matplotlib geopandas rasterio
```

Open the notebooks in order, starting with `03-GW-SW methods.ipynb` if you only want the core result.

## Data

The raw data is too large for Git and is not in the repo. The SpaFHy model used in this study is available at https://doi.org/10.5281/zenodo.19186743 with meteorological forcing files and geospatial input rasters.

## Key outputs

- **`GW_zones.csv`** — groundwater inflow zones (`x_start_m`, `x_end_m`, `width_m`, `type`)
- **`SW_zones.csv`** — surface water inflow zones 

## Citation

Citation will be added on publication.

## License

MIT

---

*Part of PhD work at the University of Oulu - DIWA PhD flagship on stream–aquifer interactions in subarctic catchments. Questions and collaboration ideas are welcome.*
