"""main.py

This file executes the full workflow to obtain global GHG/CO2 budgets and
trajectories, similar to the first cell in notebooks/Main.ipynb

It collects data from all input files, combines them into one big dataset, which is saved as xr_dataread.nc.
Also, some country-specific datareaders are executed.

Run with:
    python src/effortsharing/main.py notebooks/config.yml
"""

import logging

import numpy as np
import pandas as pd
import xarray as xr

from effortsharing.pathways.co2_trajectories import determine_global_co2_trajectories
from effortsharing.config import Config
from effortsharing.country_specific.netherlands import datareader_netherlands
from effortsharing.country_specific.norway import datareader_norway
from effortsharing.pathways.global_budgets import determine_global_budgets
from effortsharing.pathways.nonco2 import determine_global_nonco2_trajectories, nonco2variation
from effortsharing.save import save_rbw, save_rci, save_regions, save_total

logger = logging.getLogger(__name__)


def merge_data(
    xr_co2_budgets,
    all_projected_gases,
    emission_data,
    ndc_data,
    socioeconomic_data,
):
    return xr.merge(
        [
            xr_co2_budgets["Budget"],
            all_projected_gases[  # TODO: could merge whole dataarray at once, no need to list all vars explicitly. Did this to get overview of what variable comes from where.
                [
                    "GHG_globe",
                    "CO2_globe",
                    "CO2_neg_globe",
                    "NonCO2_globe",
                    "GHG_globe_excl",
                    "CO2_globe_excl",
                ]
            ],
            emission_data[  # TODO: already stored elsewhere. Remove?
                [
                    "GHG_hist",
                    "GHG_hist_excl",
                    "CO2_hist",
                    "CO2_hist_excl",
                    "CH4_hist",
                    "N2O_hist",
                    "CO2_base_excl",
                    "CO2_base_incl",
                    "GHG_base_excl",
                    "GHG_base_incl",
                    "GHG_excl_C",
                    "CO2_excl_C",
                    "CO2_neg_C",
                    "CO2_bunkers_C",
                ]
            ],
            ndc_data[  # TODO: already stored elsewhere. Remove?
                [
                    "GHG_ndc",
                    "GHG_ndc_red",
                    "GHG_ndc_inv",
                    "GHG_ndc_excl_red",
                    "GHG_ndc_excl",
                    "GHG_ndc_excl_inv",
                    "GHG_ndc_excl_CR",
                ]
            ],
            socioeconomic_data[  # TODO: already stored elsewhere. Remove?
                [
                    "GDP",
                    "HDIsh",
                    "Population",
                ]
            ],
        ]
    )


def add_country_groups(config: Config, regions, xr_total):
    logger.info("Add country groups")

    data_root = config.paths.input
    filename = "UNFCCC_Parties_Groups_noeu.xlsx"
    regions_name = list(regions.keys())
    regions_iso = list(regions.values())

    df = pd.read_excel(data_root / filename, sheet_name="Country groups")
    countries_iso = np.array(df["Country ISO Code"])
    list_of_regions = list(np.array(regions_iso).copy())
    reg_iso = regions_iso.copy()
    reg_name = regions_name.copy()
    new_total = xr_total.copy()
    for group_of_choice in [
        "G20",
        "EU",
        "G7",
        "SIDS",
        "LDC",
        "Northern America",
        "Australasia",
        "African Group",
        "Umbrella",
    ]:
        if group_of_choice != "EU":
            list_of_regions = list_of_regions + [group_of_choice]
        group_indices = countries_iso[np.array(df[group_of_choice]) == 1]
        country_to_eu = {}
        for cty in np.array(new_total.Region):
            if cty in group_indices:
                country_to_eu[cty] = [group_of_choice]
            else:
                country_to_eu[cty] = [""]
        group_coord = xr.DataArray(
            [
                group
                for country in np.array(new_total["Region"])
                for group in country_to_eu[country]
            ],
            dims=["Region"],
            coords={
                "Region": [
                    country
                    for country in np.array(new_total["Region"])
                    for group in country_to_eu[country]
                ]
            },
        )
        if group_of_choice == "EU":
            xr_eu = (
                new_total[
                    [
                        "Population",
                        "GDP",
                        "GHG_hist",
                        "GHG_base_incl",
                        "CO2_hist",
                        "CO2_base_incl",
                        "GHG_hist_excl",
                        "GHG_base_excl",
                        "CO2_hist_excl",
                        "CO2_base_excl",
                    ]
                ]
                .groupby(group_coord)
                .sum()
            )  # skipna=False)
        else:
            xr_eu = (
                new_total[
                    [
                        "Population",
                        "GDP",
                        "GHG_hist",
                        "GHG_base_incl",
                        "CO2_hist",
                        "CO2_base_incl",
                        "GHG_hist_excl",
                        "GHG_base_excl",
                        "CO2_hist_excl",
                        "CO2_base_excl",
                        "GHG_ndc",
                        "GHG_ndc_inv",
                        "GHG_ndc_excl",
                        "GHG_ndc_excl_inv",
                        "GHG_ndc_excl_CR",
                    ]
                ]
                .groupby(group_coord)
                .sum(skipna=False)
            )
        xr_eu2 = xr_eu.rename({"group": "Region"})
        dummy = new_total.reindex(Region=list_of_regions)

        new_total = xr.merge([dummy, xr_eu2])
        new_total = new_total.reindex(Region=list_of_regions)
        if group_of_choice not in ["EU", "EARTH"]:
            reg_iso.append(group_of_choice)
            reg_name.append(group_of_choice)

    new_total = new_total
    new_total["GHG_base_incl"][np.where(new_total.Region == "EU")[0], np.array([3, 4])] = (
        np.nan
    )  # SSP4, 5 are empty for Europe!
    new_total["CO2_base_incl"][np.where(new_total.Region == "EU")[0], np.array([3, 4])] = (
        np.nan
    )  # SSP4, 5 are empty for Europe!
    new_total["GHG_base_excl"][np.where(new_total.Region == "EU")[0], np.array([3, 4])] = (
        np.nan
    )  # SSP4, 5 are empty for Europe!
    new_total["CO2_base_excl"][np.where(new_total.Region == "EU")[0], np.array([3, 4])] = (
        np.nan
    )  # SSP4, 5 are empty for Europe!

    new_regions = dict(zip(reg_name, reg_iso))

    return new_total, new_regions


def main(config: Config):
    import effortsharing as es

    countries, regions = es.input.socioeconomics.read_general(config)

    # Read input data
    socioeconomic_data = es.input.socioeconomics.load_socioeconomics(config)
    modelscenarios = es.input.emissions.read_modelscenarios(config)
    emission_data = es.input.emissions.load_emissions(config)
    primap_data = es.input.emissions.read_primap(config)
    ndc_data = es.input.ndcs.load_ndcs(config)

    # Calculate global budgets and pathways
    xr_temperatures, xr_nonco2warming_wrt_start = nonco2variation(config)
    xr_traj_nonco2 = determine_global_nonco2_trajectories(config)
    xr_co2_budgets = determine_global_budgets(config)
    all_projected_gases = determine_global_co2_trajectories(config)

    # Merge all data into a single xrarray object
    xr_total = (
        merge_data(
            xr_co2_budgets,
            all_projected_gases,
            emission_data,  # TODO: already stored elsewhere. Skip?
            ndc_data,  # TODO: already stored elsewhere. Skip?
            socioeconomic_data,  # TODO: already stored elsewhere. Skip?
        )
        .reindex(Region=list(regions.values()))
        .reindex(Time=np.arange(1850, 2101))
        .interpolate_na(dim="Time", method="linear")
    )

    # Add country groups
    new_total, new_regions = add_country_groups(config, regions, xr_total)

    # Save the data
    save_temp = np.array(config.dimension_ranges.peak_temperature_saved).astype(float).round(2)
    xr_version = new_total.sel(Temperature=save_temp)
    save_regions(config, new_regions, countries)
    save_total(config, xr_version)
    save_rbw(config, xr_version, countries)
    save_rci(config, xr_version)

    # Country-specific data readers
    datareader_netherlands(config, new_total)
    datareader_norway(config, new_total, primap_data)


if __name__ == "__main__":
    import argparse

    from rich.logging import RichHandler

    # Set up logging
    logging.basicConfig(level="INFO", format="%(message)s", handlers=[RichHandler(show_time=False)])

    # Get the config file from command line arguments
    parser = argparse.ArgumentParser(description="Process all input data")
    parser.add_argument("config", help="Path to config file")
    args = parser.parse_args()

    config = Config.from_file(args.config)
    main(config)
