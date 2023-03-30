CORRECTOR_YEAR = 2019


rule bias_correct_wind:
    input:
        era5_corrector=f"data/input/europe-era5_{CORRECTOR_YEAR}.nc",
        era5="data/input/europe-era5_{year}.nc",
        wind_parks="data/input/nve_wind_parks.geojson",
        production="data/input/TimeSeries_UTC_kWh_2022_all.csv",
        europeshape="data/input/Onshore_EU-NUTS0_NO-NUTS3_UK-NG.geojson",
    output:
        uncorrected="data/output/uncorrected_wind_capacity_factors_{year}.nc",
        corrected="data/output/corrected_wind_capacity_factors_{year}.nc",
        corrected_for_highres="data/output/corrected_wind_capacity_factors_{year}.parquet",
    params:
        corrector_year=CORRECTOR_YEAR,
    conda:
        "envs/environment.yaml"
    script:
        "scripts/bias_correct_wind.py"


rule plot_bias_correction:
    input:
        era5=f"data/input/europe-era5_{CORRECTOR_YEAR}.nc",
        production="data/input/TimeSeries_UTC_kWh_2022_all.csv",
        europeshape="data/input/Onshore_EU-NUTS0_NO-NUTS3_UK-NG.geojson",
        wind_parks="data/input/nve_wind_parks.geojson",
        uncorrected_cap_factors=f"data/output/uncorrected_wind_capacity_factors_{CORRECTOR_YEAR}.nc",
        corrected_cap_factors=f"data/output/corrected_wind_capacity_factors_{CORRECTOR_YEAR}.nc",
    output:
        directory("data/output/plots"),
    params:
        corrector_year=CORRECTOR_YEAR,
    conda:
        "envs/environment.yaml"
    script:
        "scripts/plot_bias_correction.py"
