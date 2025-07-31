"""
The carbon budget explorer needs a bunch of files to work.
They are described at https://github.com/pbl-nl/website-carbon-budget-explorer/blob/main/README.md#data-requirements-and-configuration

This script generates those files.
"""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm

import effortsharing as es
from effortsharing.allocation import determine_allocations, save_allocations
from effortsharing.allocation.utils import LULUCF, Gas


def cabe_export(config_file: Path, gas: Gas = 'GHG', lulucf: LULUCF = 'incl', aggregatd_years: Iterable[int] = (2030,2040,2050) ) -> None:
    # From notebooks/Main.ipynb
    config = es.Config.from_file('config.yml')

    # Read input data
    countries, regions = es.input.socioeconomics.read_general(config)
    socioeconomic_data = es.input.socioeconomics.load_socioeconomics(config)
    modelscenarios = es.input.emissions.read_modelscenarios(config)
    emission_data = es.input.emissions.load_emissions(config)
    primap_data = es.input.emissions.read_primap(config)
    ndc_data = es.input.ndcs.load_ndcs(config, emission_data)

    # Calculate global budgets and pathways
    xr_temperatures, xr_nonco2warming_wrt_start = es.pathways.nonco2.nonco2variation(config)
    xr_traj_nonco2 = es.pathways.nonco2.determine_global_nonco2_trajectories(config)
    xr_co2_budgets = es.pathways.global_budgets.determine_global_budgets(config)
    all_projected_gases = es.pathways.co2_trajectories.determine_global_co2_trajectories(config)

    # Merge all data into a single xrarray object
    xr_total = (
        es.main.merge_data(
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
    new_total, new_regions = es.main.add_country_groups(config, regions, xr_total)

    # Save the data
    save_temp = np.array(config.dimension_ranges.peak_temperature_saved).astype(float).round(2)
    xr_version = new_total.sel(Temperature=save_temp)

    es.main.save_regions(config, new_regions, countries)
    es.main.save_total(config, xr_version)
    es.main.save_rbw(config, xr_version, countries)
    es.main.load_rci(config)

    # Country-specific data readers
    es.country_specific.netherlands.datareader_netherlands(config, new_total)
    es.country_specific.norway.datareader_norway(config, new_total, primap_data)

    # TODO get regions from somewhere else
    regions_iso = np.load(config.paths.output / "all_regions.npy", allow_pickle=True)
    # TODO allow to make selection of regions instead of defaulting to all regions
    for cty in tqdm(regions_iso):
        allocations = determine_allocations(config, cty, gas=gas, lulucf=lulucf)
        # TODO now saved as Allocations_{gas}_{lulucf}/xr_alloc_{cty}.nc
        # while cabe expects Allocations/{gas}_{lulucf}/xr_alloc_{cty}.nc
        save_allocations(config, cty, allocations)

    # From notebooks/Aggregator.ipynb
    for year in aggregatd_years:
        for cty_i, cty in tqdm(enumerate(regions_iso)):
            ds = (
                # TODO can we do region loop once instead of twice?
                xr.open_dataset(config.paths.output / "Allocations_GHG_incl" / f"xr_alloc_{cty}.nc")
                .sel(Time=year)
                .expand_dims(Region=[cty])
            )
            if cty_i == 0:
                xrt = ds.copy()
            else:
                xrt = xr.merge([xrt, ds])
            ds.close()
        # TODO save as {CABE_DATA_DIR} / {CABE_START_YEAR} / {CABE_ASSUMPTIONSET} / "Aggregated_files" / "xr_alloc_{YEAR}.nc"
        xrt.astype("float32").to_netcdf(
            config.paths.output / "Aggregated_files" / f"xr_alloc_{year}_GHG_incl.nc", format="NETCDF4"
        )