# ======================================== #
# Class that does the variance decomposition
# ======================================== #

# =========================================================== #
# PREAMBULE
# Put in packages that we need
# =========================================================== #

from pathlib import Path
import yaml
import numpy as np
import pandas as pd
import xarray as xr
from SALib.sample import saltelli
from SALib.analyze import sobol
from tqdm import tqdm

# =========================================================== #
# CLASS OBJECT
# =========================================================== #

class vardecomposing(object):

    # =========================================================== #
    # =========================================================== #

    def __init__(self, xrtot, cty):
        print("# ==================================== #")
        print("# Initializing vardecomposing class        #")
        print("# ==================================== #")

        self.current_dir = Path.cwd()

        # Read in Input YAML file
        with open(self.current_dir / 'input.yml') as file:
            self.settings = yaml.load(file, Loader=yaml.FullLoader)
        self.xr_total = xrtot
        self.countries_iso = cty
        self.xr_unc = xr.open_dataset("K:/ECEMF/T5.2/xr_uncbudget_15.nc")
        self.all_regions_iso = np.load(self.settings['paths']['data']['internal'] + "all_regions.npy")
        self.all_regions_names = np.load(self.settings['paths']['data']['internal'] + "all_regions_names.npy")
        self.all_countries_iso = np.load(self.settings['paths']['data']['internal'] + "all_countries.npy", allow_pickle=True)
        self.all_countries_names = np.load(self.settings['paths']['data']['internal'] + "all_countries_names.npy", allow_pickle=True)

        # Define dimensions
        uni_es = np.array(['GF', 'PC', 'PCC', 'ECPC', 'AP', 'GDR'])
        uni_scen = np.array(self.xr_unc.Scenario)
        uni_nonco2 = np.array(self.xr_unc.NonCO2)
        uni_conv = np.array(self.xr_unc.Convergence_year)
        uni_risk = np.array(self.xr_unc.Risk)
        unis = [uni_es, uni_scen, uni_nonco2, uni_conv, uni_risk]

        # Set a few core parameters
        self.K = len(unis) # Total number of dimensions
        self.N = 1 # Redraws
        self.ss = 1 # Sample size per unique parameter setting
        self.total_ss = self.ss*len(uni_es)*len(uni_scen)*len(uni_nonco2)*len(uni_conv)*len(uni_risk) # Total sample size

    # =========================================================== #
    # =========================================================== #

    def construct_samples(self):
        print("- Create samples")


    # =========================================================== #
    # =========================================================== #

    def apply_decomposition(self):
        print("- Create samples")
        problem = {
            'num_vars': 4,
            'names': ['Temperature',
                      'Convergence_year',
                      'Scenario',
                      'EffortSharing',
                      'Risk_of_exceedance',
                      'Negative_emissions',
                      'Non_CO2_mitigation_potential'],
            'bounds': [[1.5, 3.0],
                        [np.min(self.xr_total.Convergence_year), np.max(self.xr_total.Convergence_year)],
                        [0, 1],
                        [0, 6],
                        [np.min(self.xr_total.Risk_of_exceedance), np.max(self.xr_total.Risk_of_exceedance)],
                        [np.min(self.xr_total.Negative_emissions), np.max(self.xr_total.Negative_emissions)],
                        [np.min(self.xr_total.Non_CO2_mitigation_potential), np.max(self.xr_total.Non_CO2_mitigation_potential)],
                        ]
        }
        param_values = saltelli.sample(problem, 64)
        
        def refine_sample(pars):
            v1 = pars.copy()[:, 0]
            v1s = v1.astype(str)
            v1s[v1 < 1.75] = '1.5 deg'
            v1s[(v1 < 2.25) & (v1 >= 1.85)] = '2.0 deg'
            v1s[(v1 < 2.75) & (v1 >= 2.25)] = '2.5 deg'
            v1s[(v1 >= 2.75)] = '3.0 deg'
            
            v2 = pars.copy()[:, 1]
            v2s = (np.round(v2 / 5) * 5).astype(int)
            
            v3 = pars.copy()[:, 2]
            v3s = v3.astype(str)
            v3s[v3 < 0.2] = 'SSP1'
            v3s[(v3 >= 0.2) & (v3 < 0.4)] = 'SSP2'
            v3s[(v3 >= 0.4) & (v3 < 0.6)] = 'SSP3'
            v3s[(v3 >= 0.6) & (v3 < 0.8)] = 'SSP4'
            v3s[(v3 >= 0.8) & (v3 < 1.0)] = 'SSP5'
            
            v4 = pars.copy()[:, 3]
            v4s = v4.astype(str)
            v4s[v4 < 1] = 'GF'
            v4s[(v4 >= 1) & (v4 < 2)] = 'PC'
            v4s[(v4 >= 2) & (v4 < 3)] = 'PCC'
            v4s[(v4 >= 3) & (v4 < 4)] = 'AP'
            v4s[(v4 >= 4) & (v4 < 5)] = 'GDR'
            v4s[(v4 >= 5) & (v4 < 6)] = 'ECPC'
            return np.array([v1s, v2s, v3s, v4s]).T

        def func(pars, reg, xrt):
            xrt2 = xrt.sel(Region=reg)
            vec = np.zeros(len(pars))
            for i in range(len(pars)):
                vec[i] = float(xrt2.sel(Temperature=pars[i, 0],
                                    Convergence_year=np.array(pars[i, 1]).astype(int),
                                    Scenario=pars[i, 2])[pars[i, 3]])
            return vec
        
        xrt = self.xr_total.sel(Time=np.arange(self.settings['params']['start_year_analysis'], 2101)).sum(dim='Time')

        print("- Apply decomposition")
        Sis = np.zeros(shape=(len(self.countries_iso), 4))
        for reg_i, reg in tqdm(enumerate(self.countries_iso)):
            Y = func(refine_sample(param_values), reg, xrt)
            Si = sobol.analyze(problem, Y)
            Sis[reg_i, :] = Si['ST']
        self.Si_norm = (Sis.T / Sis.sum(axis=1)).T

    # =========================================================== #
    # =========================================================== #

    def save(self):
        print("- Save results")
        self.Si_norm