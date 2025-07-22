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


def determine_allocations(
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
    # TODO refactor or remove?
    # if self.dataread_file != "xr_dataread.nc":
    #     savename = "xr_alloc_" + self.focus_region + "_adapt.nc"
    dir = config.paths.output / f"Allocations_{gas}_{lulucf}"
    dir.mkdir(parents=True, exist_ok=True)
    save_path = dir / fn

    start_year_analysis = config.params.start_year_analysis
    # TODO move to config.config.params?
    end_year_analysis = 2101

    combined = (
        xr.merge(dss, compat="override")
        .sel(Time=np.arange(start_year_analysis, end_year_analysis))
        .astype("float32")
    )
    logger.info(f"Saving allocations to {save_path}")
    combined.to_netcdf(save_path, format="NETCDF4")


if __name__ == "__main__":
    import argparse

    from rich.logging import RichHandler

    # Set up logging
    logging.basicConfig(level="INFO", format="%(message)s", handlers=[RichHandler(show_time=False)])

    # Get the config file from command line arguments
    parser = argparse.ArgumentParser(description="Process emission input data")
    parser.add_argument("--config", help="Path to config file", default="notebooks/config.yml")
    parser.add_argument("--region", help="Region to allocate emissions for", default="BRA")
    parser.add_argument(
        "--gas",
        choices=["CO2", "GHG"],
        default="GHG",
        help="Gas type to allocate emissions for (default: GHG)",
    )
    parser.add_argument(
        "--lulucf",
        choices=["incl", "excl"],
        default="incl",
        help="LULUCF treatment (default: incl)",
    )

    args = parser.parse_args()

    # Read config
    config = Config.from_file(args.config)
    region = args.region
    gas: Gas = args.gas
    lulucf: LULUCF = args.lulucf
    gf_da = gf(config, region, gas, lulucf)
    pc_da = pc(config, region, gas, lulucf)
    pcc_da = pcc(config, region, gas, lulucf)
    pcb_da, pcb_lin_da = pcb(config, region, gas, lulucf)
    ecpc_da = ecpc(config, region, gas, lulucf)
    ap_da = ap(config, region, gas, lulucf)
    gdr_da = gdr(config, region, gas, lulucf)
    save_allocations(
        config=config,
        region=region,
        gas=gas,
        lulucf=lulucf,
        dss=[
            gf_da,
            pc_da,
            pcc_da,
            pcb_da,
            ecpc_da,
            ap_da,
            gdr_da,
        ],
    )
