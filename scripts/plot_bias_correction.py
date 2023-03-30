# SPDX-FileCopyrightText: 2023 Aleksander Grochowicz & Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""This script is used to bias correct reanalysis wind data according
to production data from Norwegian wind farms.

Grid cells that contain wind farms are bias corrected according to the
production data from the wind farms. For grid cells with no wind
farms, we use the bias correction from the nearest grid cell with a
wind farm."""


import math
import os

import atlite
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
import xarray as xr
from statsmodels.distributions.empirical_distribution import ECDF

# Load the input data: cutout (both for corrector year and the one to
# be corrected), shapefile, production data, and wind park locations.

# Start with the shape file to extract Norway.
europe = gpd.read_file(snakemake.input.europeshape)
norway = europe.loc[europe.NUTS_ID == "NO"]
# Remove svalbard by intersecting with box
norway = norway.intersection(shapely.geometry.box(0, 0, 60, 73))

# NB: We add a buffer because the resolution of the grid cells does
# not capture the fractal nature of the coastline.
norway_unbuffered = norway
norway = norway.buffer(0.4)

# Load the cutouts.
# NB: We need two cutouts, one for the year which is the base for the
# bias correction, and one for the year which is to be corrected.
cutout = atlite.Cutout(snakemake.input.era5)

# Create a grid based on the cutout grid.
norway_grid = cutout.grid.loc[cutout.grid.intersects(norway.unary_union)]
norway_grid.reset_index(inplace=True)

# Load the wind park data (from NVE).
wind_parks = gpd.read_file(snakemake.input.wind_parks)
wind_parks["anleggNavn"] = wind_parks["anleggNavn"].str.rstrip()  # Index by name.
wind_parks.set_index("anleggNavn", inplace=True)
wind_parks = wind_parks.sort_index()


# Load the production data (from NVE).
production_data = pd.read_csv(
    snakemake.input.production, index_col=0, parse_dates=True, encoding="latin-1"
).filter(like="production", axis="columns")
# NB: The data from NVE exists from 2003 until 2021.
production_data = production_data.loc[str(snakemake.params.corrector_year)]
# Columns are wind park names; format to align with the "wind_parks" dataframe.
new_cols = production_data.columns.str.split("_", n=1, expand=True)
new_cols = new_cols.droplevel(1)
production_data.rename(
    columns=dict(zip(production_data.columns, new_cols)), inplace=True
)

# Normalise the production data for capacity factors; note that the
# reported installed capacities seem unreliable.
NVE_cap_factors = production_data / production_data.max()

# Find the set of all wind parks for which we have both production
# data and geographical data.
wind_parks = wind_parks.loc[wind_parks.index.intersection(production_data.columns)]
production_data = production_data[wind_parks.index]

# Collect all grid cells with wind turbines
turbine_cells = gpd.sjoin(
    norway_grid, wind_parks[["geometry"]], how="left", predicate="intersects"
).dropna()
turbine_cells.rename(
    columns={"index": "gridcell_code", "index_right": "wind_park"}, inplace=True
)
turbine_cells.set_index("wind_park", inplace=True)


uncorrected_data = xr.open_dataarray(
    snakemake.input.uncorrected_cap_factors
).to_pandas()
corrected_data = xr.open_dataarray(snakemake.input.corrected_cap_factors).to_pandas()


fig, ax = plt.subplots(figsize=(15, 6))
ERA5_mean_caps = pd.DataFrame(
    [
        uncorrected_data[turbine_cells.loc[wind_park, "gridcell_code"]].mean()
        for wind_park in wind_parks.index
    ],
    index=wind_parks.index,
)
production_mean_caps = (production_data / production_data.max()).mean()
corrected_caps = pd.DataFrame(corrected_data).mean()
corrected_caps = corrected_caps[turbine_cells["gridcell_code"].unique()]
corrected_caps = pd.DataFrame(
    [
        corrected_caps.loc[turbine_cells.loc[wind_park, "gridcell_code"]]
        for wind_park in wind_parks.index
    ],
    index=wind_parks.index,
)
mean_caps = pd.concat([ERA5_mean_caps, production_mean_caps, corrected_caps], axis=1)
mean_caps.columns = ["ERA5", "NVE", "corrected"]
mean_caps.sort_values(by="ERA5", inplace=True)

mean_caps.plot(
    title="Mean capacity factor by data source", ax=ax, alpha=1, label="NVE", kind="bar"
)

os.makedirs(snakemake.output[0], exist_ok=True)
fig.savefig(os.path.join(snakemake.output[0], "mean_cap_factors.png"))


# Collect the empirical cumulative distribution functions (ECDFs) for
# the reanalysis and NVE data.
ECDFs_reanalysis = {}
ECDFS_nve = {}
ECDFS_corrected = {}
for wind_park in wind_parks.index:
    ECDFs_reanalysis[wind_park] = ECDF(
        uncorrected_data[turbine_cells.loc[wind_park, "gridcell_code"]]
    )
    ECDFS_nve[wind_park] = ECDF(NVE_cap_factors[wind_park])
    ECDFS_corrected[wind_park] = ECDF(
        corrected_data[turbine_cells.loc[wind_park, "gridcell_code"]]
    )

xs = np.linspace(0, 1, 1001)
len_farms = len(wind_parks)
fig, axs = plt.subplots(math.ceil(len_farms / 4), 4, figsize=(20, 20), sharex=True)
for i, wind_park in enumerate(wind_parks.index):
    F = ECDFs_reanalysis[wind_park]
    G = ECDFS_nve[wind_park]
    H = ECDFS_corrected[wind_park]
    ax = axs.flatten()[i]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.plot(xs, G(xs), label="NVE", ls="dashed")
    ax.plot(xs, F(xs), label="ERA5", ls="dotted")
    ax.plot(xs, H(xs), label="ERA5 Corrected")
    ax.set_title(wind_park)
    ax.legend()

fig.savefig(os.path.join(snakemake.output[0], "ECDFs.png"), dpi=300)


fig, axs = plt.subplots(1, 3, figsize=(15, 6))

G = norway_grid.loc[norway_grid.geometry.intersects(norway_unbuffered.unary_union)]
G.set_index("index", inplace=True)
G["corrected_cap_factor"] = corrected_data.mean()
G.plot(column="corrected_cap_factor", figsize=(7, 7), legend=True, ax=axs[0])

G["correction"] = corrected_data.mean() - uncorrected_data.mean()
G.plot(column="correction", figsize=(7, 7), legend=True, cmap="RdBu", ax=axs[1])

G["ERA5_cap_factor"] = uncorrected_data.mean()
G.plot(column="ERA5_cap_factor", figsize=(7, 7), legend=True, ax=axs[2])

fig.savefig(os.path.join(snakemake.output[0], "capacity_factors_map.png"), dpi=300)
