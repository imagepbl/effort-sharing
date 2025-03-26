# ======================================== #
# Class that does the variance decomposition
# ======================================== #

# =========================================================== #
# PREAMBULE
# Put in packages that we need
# =========================================================== #

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from SALib.analyze import sobol

# Sobol analysis
from SALib.sample import saltelli
from SALib.test_functions import Ishigami
from tqdm import tqdm

warnings.simplefilter(action="ignore")

# =========================================================== #
# CLASS OBJECT
# =========================================================== #


class vardecomposing(object):
    # =========================================================== #
    # =========================================================== #

    def __init__(self, startyear=2021, gas="GHG", lulucf="incl"):
        print("# ==================================== #")
        print("# Initializing vardecomposing class    #")
        print("# ==================================== #")

        self.current_dir = Path.cwd()

        # Read in Input YAML file
        with open(self.current_dir / "input.yml") as file:
            self.settings = yaml.load(file, Loader=yaml.FullLoader)

        self.startyear = startyear
        self.gas = gas
        self.lulucf = lulucf

        self.xr_total = xr.open_dataset(
            self.settings["paths"]["data"]["datadrive"]
            + "startyear_"
            + str(self.startyear)
            + "/xr_dataread.nc"
        )
        self.all_regions_iso = np.load(
            self.settings["paths"]["data"]["datadrive"] + "all_regions.npy"
        )
        self.all_regions_names = np.load(
            self.settings["paths"]["data"]["datadrive"] + "all_regions_names.npy"
        )
        self.all_countries_iso = np.load(
            self.settings["paths"]["data"]["datadrive"] + "all_countries.npy",
            allow_pickle=True,
        )
        self.all_countries_names = np.load(
            self.settings["paths"]["data"]["datadrive"] + "all_countries_names.npy",
            allow_pickle=True,
        )

    # =========================================================== #
    # =========================================================== #

    def prepare_global_sobol(self, year):
        # print("- Prepare Sobol decomposition and draw samples for the full globe in fixed year")
        self.xr_year = xr.open_dataset(
            self.settings["paths"]["data"]["datadrive"]
            + "startyear_"
            + str(self.startyear)
            + "/Aggregated_files/xr_alloc_"
            + str(year)
            + "_"
            + self.gas
            + "_"
            + self.lulucf
            + ".nc"
        )
        xr_globe = self.xr_year.bfill(dim="Timing")[["PCC", "ECPC", "AP"]].sel(
            Temperature=[1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
            Risk=[0.67, 0.5, 0.33],
            NonCO2red=[0.33, 0.5, 0.67],
            NegEmis=[0.33, 0.5, 0.67],
            Region=np.array(self.xr_year.Region),
            Scenario=["SSP1", "SSP2", "SSP3"],
            Convergence_year=[2040, 2050, 2080],
        )
        array_dims = np.array(xr_globe.sel(Region=xr_globe.Region[0]).to_array().dims)
        array_inputs = [["PCC", "ECPC", "AP"]]
        for dim_i, dim in enumerate(array_dims[1:]):
            array_inputs.append(list(np.array(xr_globe[dim])))
        problem = {
            "num_vars": len(array_dims),
            "names": array_dims,
            "bounds": [[0, len(ly)] for ly in array_inputs],
        }
        samples = np.floor(saltelli.sample(problem, 2**10)).astype(int)
        return (
            xr_globe,
            np.array(xr_globe.Region),
            array_dims,
            array_inputs,
            problem,
            samples,
        )

    # =========================================================== #
    # =========================================================== #

    def apply_decomposition(
        self, xdataset_, maindim_, dims_, inputs_, problem_, samples_
    ):
        # print("- Read functions and apply actual decomposition")
        def refine_sample(pars):
            new_pars = pars.astype(str)
            actual_values = []
            for var_i, var in enumerate(dims_):
                actual_val = np.array(inputs_[var_i])[pars[:, var_i]]
                actual_values.append(actual_val)
            actual_values = np.array(actual_values).T
            return actual_values

        def refine_sample_int(pars):
            return np.floor(pars).astype(int)

        def func2(pars, ar):
            vec = np.zeros(len(pars))
            for i in range(len(pars)):
                f = ar[
                    pars[i, 0],
                    pars[i, 1],
                    pars[i, 2],
                    pars[i, 3],
                    pars[i, 4],
                    pars[i, 5],
                    pars[i, 6],
                    pars[i, 7],
                    pars[i, 8],
                    pars[i, 9],
                ]
                vec[i] = f
            return vec

        Sis = np.zeros(shape=(len(maindim_), problem_["num_vars"]))
        ar_xrt = np.array(xdataset_.to_array())
        for reg_i, reg in tqdm(enumerate(maindim_)):
            xr_touse = ar_xrt[:, reg_i]
            Y = func2(refine_sample_int(samples_), xr_touse)
            Si = sobol.analyze(problem_, Y)
            Sis[reg_i, :] = Si["ST"]
        Si_norm = (Sis.T / Sis.sum(axis=1)).T
        for i in range(len(Si_norm)):
            m_i = np.nanmin(Si_norm[i])
            if m_i < 0:
                Si_norm[i] = (Si_norm[i] - m_i) / np.sum((Si_norm[i] - m_i))
        Si_norm[np.unique(np.where(np.isnan(Si_norm))[0])] = np.nan
        return Si_norm

    # =========================================================== #
    # =========================================================== #

    def save(self, dims_, times_):
        print("- Save global results")
        d = {}
        d["Time"] = times_
        d["Factor"] = dims_
        d["Region"] = np.array(
            xr.open_dataset(
                "K:/Data/Data_EffortSharing/DataUpdate_ongoing/startyear_2021/Aggregated_files/xr_alloc_2030_GHG_incl.nc"
            ).Region
        )  # Is independent of startyear, gas and lulucf

        xr_sobol = xr.Dataset(coords=d)

        sobol_data = {
            "Sobol_index": xr.DataArray(
                data=np.nan,
                coords=xr_sobol.coords,
                dims=xr_sobol.dims,
                attrs={"description": "Sobol indices"},
            )
        }

        for Time_i, Time in enumerate(times_):
            sobol_data["Sobol_index"][Time_i, :, :] = self.sobolindices[Time].T
        self.xr_sobol = xr_sobol.update(sobol_data)
        self.xr_sobol.to_netcdf(
            self.settings["paths"]["data"]["datadrive"]
            + "/startyear_"
            + str(self.startyear)
            + "/xr_sobol_"
            + self.gas
            + "_"
            + self.lulucf
            + ".nc",
            format="NETCDF4",
            engine="netcdf4",
        )
