# ======================================== #
# Class that does the budget allocation
# ======================================== #

# =========================================================== #
# PREAMBULE
# Put in packages that we need
# =========================================================== #

import logging
from collections.abc import Iterable

import numpy as np
import xarray as xr
from tqdm import tqdm

from effortsharing.allocation.ap import ap
from effortsharing.allocation.ecpc import ecpc
from effortsharing.allocation.gdr import gdr
from effortsharing.allocation.gf import gf
from effortsharing.allocation.pc import pc
from effortsharing.allocation.pcb import pcb
from effortsharing.allocation.pcc import pcc
from effortsharing.allocation.utils import LULUCF, Gas
from effortsharing.config import Config

logger = logging.getLogger(__name__)

# =========================================================== #
# allocation methods
# =========================================================== #


def allocations_for_year(config: Config, regions, gas: Gas, lulucf: LULUCF, year: int):
    """Extract allocations for a specific year from the regional allocations."""
    # TODO now expects xr_alloc_{REGION}.nc files to exist
    # and gives error if not
    # would be nice if they where generated here
    for cty_i, cty in tqdm(enumerate(regions), desc=f"Allocation for {year}", unit="region"):
        fn = (
            config.paths.output
            / f"startyear_{config.params.start_year_analysis}"
            / f"{gas}_{lulucf}"
            / "Allocations"
            / f"xr_alloc_{cty}.nc"
        )
        if not fn.exists():
            raise FileNotFoundError(
                f"Allocation file {fn} does not exist. First calculate allocations for each region."
            )
        ds = (
            # TODO can we do region loop once instead of here and above?
            xr.open_dataset(fn).sel(Time=year).expand_dims(Region=[cty])
        )
        if cty_i == 0:
            xrt = ds.copy()
        else:
            xrt = xr.merge([xrt, ds])
        ds.close()
    # TODO save as {CABE_DATA_DIR} / {CABE_START_YEAR} / {CABE_ASSUMPTIONSET} / "Aggregated_files" / "xr_alloc_{YEAR}.nc"
    # change here not in cabe
    root = (
        config.paths.output
        / f"startyear_{config.params.start_year_analysis}"
        / f"{gas}_{lulucf}"
        / "Aggregated_files"
    )
    root.mkdir(parents=True, exist_ok=True)
    xrt.astype("float32").to_netcdf(root / f"xr_alloc_{year}.nc", format="NETCDF4")


def allocations_for_region(
    config: Config, region, gas: Gas = "GHG", lulucf: LULUCF = "incl"
) -> list[xr.DataArray]:
    """
    Run all allocation methods and return list of xr.DataArray per method.
    """
    # TODO report progress with logger.info or tqdm
    gf_da = gf(config, region, gas, lulucf)
    pc_da = pc(config, region, gas, lulucf)
    pcc_da = pcc(config=config, region=region, gas=gas, lulucf=lulucf, gf_da=gf_da, pc_da=pc_da)
    pcb_da, pcb_lin_da = pcb(config, region, gas, lulucf)
    ecpc_da = ecpc(config, region, gas, lulucf)
    ap_da = ap(config, region, gas, lulucf)
    gdr_da = gdr(config=config, region=region, gas=gas, lulucf=lulucf, ap_da=ap_da)

    return [
        gf_da,
        pc_da,
        pcc_da,
        pcb_da,
        pcb_lin_da,
        ecpc_da,
        ap_da,
        gdr_da,
    ]


def save_allocations(
    config: Config,
    region: str,
    dss: Iterable[xr.DataArray],
    gas: Gas = "GHG",
    lulucf: LULUCF = "incl",
):
    """
    Combine data arrays returned by each allocation method into a NetCDF file
    """
    fn = f"xr_alloc_{region}.nc"
    root = (
        config.paths.output
        / f"startyear_{config.params.start_year_analysis}"
        / f"{gas}_{lulucf}"
        / "Allocations"
    )
    root.mkdir(parents=True, exist_ok=True)
    save_path = root / fn

    start_year_analysis = config.params.start_year_analysis
    end_year_analysis = 2101

    combined = (
        xr.merge(dss, compat="override")
        .sel(Time=np.arange(start_year_analysis, end_year_analysis))
        .astype("float32")
    )
    logger.info(f"Saving allocations to {save_path}")
    combined.to_netcdf(save_path, format="NETCDF4")
