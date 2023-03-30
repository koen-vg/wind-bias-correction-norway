# SPDX-FileCopyrightText: 2023 Aleksander Grochowicz & Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""This script is used to bias correct reanalysis wind data according
to production data from Norwegian wind farms.

Grid cells that contain wind farms are bias corrected according to the
production data from the wind farms. For grid cells with no wind
farms, we use the bias correction from the nearest grid cell with a
wind farm."""

import atlite
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr


def bias_correct(production_cfs, era5_cfs):
    """Estimates bias correction function from prod. and ERA5 data."""
    N = 1001
    production_quantiles = production_cfs.quantile(np.linspace(0, 1, N))
    era5_quantiles = era5_cfs.quantile(np.linspace(0, 1, N))

    def corrector(cf):
        # The bias correction function takes a capacity factor, finds
        # the quantile it is at in the ERA5 data, and returns the
        # capacity factor at that quantile in the NVE data.
        return production_quantiles.iloc[
            np.clip(
                era5_quantiles.searchsorted(cf, side="right"), a_min=None, a_max=N - 1
            )
        ]

    return corrector


def bias_correct_data(data, corrector):
    """Bias correct `data` using the (estimated) `corrector` function."""
    return data.apply(corrector)


if __name__ == "__main__":
    # Load the input data: cutout (both for corrector year and the one to
    # be corrected), shapefile, production data, and wind park locations.

    # Start with the shape file to extract Norway.
    europe = gpd.read_file(snakemake.input.europeshape)
    norway = europe.loc[europe.NUTS_ID == "NO"]
    # Remove svalbard by intersecting with box
    norway = norway.intersection(shapely.geometry.box(0, 0, 60, 73))
    # NB: We add a buffer because the resolution of the grid cells does
    # not capture the fractal nature of the coastline.
    norway = norway.buffer(0.4)

    # Load the cutouts.
    # NB: We need two cutouts, one for the year which is the base for the
    # bias correction, and one for the year which is to be corrected.
    cutout = atlite.Cutout(snakemake.input.era5)
    cutout_corrector = atlite.Cutout(snakemake.input.era5_corrector)
    turbine = "Vestas_V112_3MW"

    # Create a grid based on the cutout grid.
    norway_grid = cutout_corrector.grid.loc[
        cutout_corrector.grid.intersects(norway.unary_union)
    ]
    norway_grid.reset_index(inplace=True)

    # Generate the capacity factors.
    cfs_corrector = cutout_corrector.wind(
        turbine=turbine, shapes=norway_grid
    ).to_pandas()
    cfs_corrector.rename(columns=norway_grid["index"].to_dict(), inplace=True)

    cfs = cutout.wind(turbine=turbine, shapes=norway_grid).to_pandas()
    cfs.rename(columns=norway_grid["index"].to_dict(), inplace=True)

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
    # Columns are wind park names; format to align with the
    # "wind_parks" dataframe.
    new_cols = production_data.columns.str.split("_", n=1, expand=True)
    new_cols = new_cols.droplevel(1)
    production_data.rename(
        columns=dict(zip(production_data.columns, new_cols)), inplace=True
    )

    # Normalise the production data for capacity factors; note that the
    #  reported installed capacities seem unreliable.
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

    print("Finished loading data and generating capacity factors.")

    # Bias correct the data for the wind parks and their corresponding grid cells.
    bias_correction_factors = {}
    corrected_data = {}

    for wind_park in wind_parks.index:
        corrector_data = cfs_corrector[turbine_cells.loc[wind_park, "gridcell_code"]]
        ERA5_data = cfs[turbine_cells.loc[wind_park, "gridcell_code"]]
        # NB: The bias corrector is based on snakemake.params.corrector_year.
        corrector = bias_correct(NVE_cap_factors[wind_park], corrector_data)
        bias_correction_factors[wind_park] = corrector
        # We correct the data for snakemake.wildcards.year.
        corrected_data[wind_park] = bias_correct_data(ERA5_data, corrector)

    print("Finished generating bias correctors.")

    # Bias correct the data for the grid cells without wind parks.
    nearest_windpark = {}
    for _, row in norway_grid.iterrows():
        nearest_windpark[row["index"]] = row.geometry.distance(
            wind_parks.geometry
        ).idxmin()

    corrected_cap_factors = cfs.copy()  # NB: Based on snakemake.wildcards.year.
    for i, code in enumerate(corrected_cap_factors.columns):
        # Select the corrector for the nearest windpark.
        corrector = bias_correction_factors[nearest_windpark[code]]
        corrected_cap_factors[code] = bias_correct_data(
            corrected_cap_factors[code], corrector
        )
        if i % 100 == 0:
            print(
                f"Finished correcting {i} out of "
                f"{len(corrected_cap_factors.columns)} grid cells."
            )

    # Export the corrected (and uncorrected) capacity factors.
    xr.DataArray(corrected_cap_factors).to_netcdf(snakemake.output.corrected)
    xr.DataArray(cfs).to_netcdf(snakemake.output.uncorrected)

    # Output for highres

    def format_coords(x: float, y: float) -> str:
        """Format coordinates to grid cell names as expected by highres"""
        # Example: (-10.50, 60.0) -> "x-1050y6000"
        x = int(float(x) * 100)
        y = int(float(y) * 100)
        return f"x{x}y{y}"

    # Rename gridcells using the above formatting
    corrected_for_highres = corrected_cap_factors.rename(
        lambda code: format_coords(
            *tuple(
                norway_grid.loc[norway_grid["index"] == code][
                    ["x", "y"]
                ].values.flatten()
            )
        ),
        axis="columns",
    )

    # Mold the capacity factors into the format required by highres
    corrected_for_highres = corrected_for_highres.stack()
    corrected_for_highres.index.set_names(["time", "spatial"], inplace=True)
    corrected_for_highres = pd.DataFrame(corrected_for_highres)
    corrected_for_highres["technology"] = "Windonshore"
    corrected_for_highres = corrected_for_highres.reset_index().set_index(
        ["time", "technology", "spatial"]
    )
    corrected_for_highres.rename(columns={0: "0"}, inplace=True)
    corrected_for_highres.to_parquet(snakemake.output.corrected_for_highres)
