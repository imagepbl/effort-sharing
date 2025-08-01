"""Script to generate files for the Carbon Budget Explorer website.

The carbon budget explorer needs a bunch of files to work.
They are described at
https://github.com/pbl-nl/website-carbon-budget-explorer/blob/main/README.md#data-requirements-and-configuration

This script generates those files.
"""

import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

import effortsharing as es
from effortsharing.allocation import allocations_for_region, allocations_for_year, save_allocations
from effortsharing.allocation.utils import LULUCF, Gas
from effortsharing.cli import use_rich_logger
from effortsharing.input.policyscens import policy_scenarios
from effortsharing.pathways.global_pathways import global_pathways

# Set up logging
use_rich_logger(level="INFO")

# Configuration
config_file = Path("config.yml")
config = es.Config.from_file(config_file)
# TODO move below to config file? or move some from config to here?
gas: Gas = "GHG"
lulucf: LULUCF = "incl"
aggregated_years = (2030, 2040, 2050)

# Create {CABE_START_YEAR} / "xr_dataread.nc"
global_pathways(config)

# Create "xr_policyscen.nc"
policy_scenarios(config)

# Create {CABE_START_YEAR} / {CABE_ASSUMPTIONSET} / "Allocations" / "xr_alloc_{REGION}.nc"
# TODO get regions from somewhere else
regions_iso = np.load(config.paths.output / "all_regions.npy", allow_pickle=True)
# uncomment below for testing only run on a few regions
# regions_iso = list(regions_iso)[:4]
for cty in tqdm(regions_iso, desc="Allocations for region", unit="region"):
    dss = allocations_for_region(region=cty, config=config, gas=gas, lulucf=lulucf)
    save_allocations(dss=dss, region=cty, config=config, gas=gas, lulucf=lulucf)

# Create {CABE_START_YEAR} / {CABE_ASSUMPTIONSET} / "Aggregated_files" / "xr_alloc_{YEAR}.nc"
for year in aggregated_years:
    allocations_for_year(year=year, config=config, regions=regions_iso, gas=gas, lulucf=lulucf)
