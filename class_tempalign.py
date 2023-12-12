# ======================================== #
# Class that does the temperature alignment
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
from tqdm import tqdm
import json

# =========================================================== #
# CLASS OBJECT
# =========================================================== #

class tempaligning(object):

    # =========================================================== #
    # =========================================================== #

    def __init__(self):
        print("# ==================================== #")
        print("# Initializing tempaligning class        #")
        print("# ==================================== #")

        self.current_dir = Path.cwd()

        # Read in Input YAML file
        with open(self.current_dir / 'input.yml') as file:
            self.settings = yaml.load(file, Loader=yaml.FullLoader)
        self.countries_iso = np.load(self.settings['paths']['data']['datadrive'] + "all_countries.npy", allow_pickle=True)
        self.xr_total = xr.open_dataset(self.settings['paths']['data']['datadrive'] + "xr_dataread.nc").sel()

    # =========================================================== #
    # =========================================================== #

    def get_relation_2030emis_temp(self):
        print("- Determine relation between 2030-emissions and temperature outcome")
        df_ar6_2 = pd.read_csv(self.settings['paths']['data']['external']+"IPCC/AR6_Scenarios_Database_World_v1.1.csv")
        df_ar6_2 = df_ar6_2[df_ar6_2.Variable.isin(['Emissions|Kyoto Gases',
                                                    'AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|5.0th Percentile',
                                                    'AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile',
                                                    'AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|95.0th Percentile'])]
        mods = np.array(df_ar6_2.Model)
        scens = np.array(df_ar6_2.Scenario)
        modscens = np.array([mods[i]+'|'+scens[i] for i in range(len(scens))])
        df_ar6_2['ModelScenario'] = modscens
        df_ar6_2 = df_ar6_2.drop(['Model', 'Scenario', 'Region', 'Unit'], axis=1)
        dummy = df_ar6_2.melt(id_vars=["ModelScenario", "Variable"], var_name="Time", value_name="Value")
        dummy['Time'] = np.array(dummy['Time'].astype(int))
        dummy = dummy.set_index(["ModelScenario", "Variable", "Time"])
        xr_ar6_2 = xr.Dataset.from_dataframe(dummy)
        x_data = xr_ar6_2.sel(Time=2030, Variable='Emissions|Kyoto Gases').Value
        y_data = xr_ar6_2.sel(Time=2100, Variable='AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile').Value
        mask = ~np.isnan(y_data)
        x_fit = x_data[mask]
        y_fit = y_data[mask]
        mask = ~np.isnan(x_fit)
        x_fit = x_fit[mask]
        y_fit = y_fit[mask]
        self.coef_ghg_2030 = np.polyfit(x_fit, y_fit, self.settings['params']['polynomial_fit_2030relation'])

    # =========================================================== #
    # =========================================================== #

    def determine_tempoutcomes(self):
        print("- Determine temperature metric")
        xr_alloc_2030 = xr.open_dataset("K:/ECEMF/T5.2/xr_alloc_2030.nc")
        rules = ['GF', 'PC', 'PCC', 'ECPC', 'AP', 'GDR']
        percs = (xr_alloc_2030[rules] / xr_alloc_2030.sel(Region=self.countries_iso).GF.sum(dim='Region')).mean(dim='Temperature')
        condition = percs < 0
        percs = percs.where(~condition, 1e-9)
        ndc_globalversion_raw = self.xr_total.GHG_ndc.sel(Time=2030)/percs
        condition = ndc_globalversion_raw < 10000
        mod_data = ndc_globalversion_raw.where(~condition, 10000)
        condition = mod_data > 75000
        ndc_globalversion = mod_data.where(~condition, 75000)
        for n in tqdm(range(self.settings['params']['polynomial_fit_2030relation']+1)):#self.coef_ghg_2030[0]*x_fit**5+ self.coef_ghg_2030[1]*x_fit**4+ self.coef_ghg_2030[2]*x_fit**3 +self.coef_ghg_2030[3]*x_fit**2 +self.coef_ghg_2030[4]*x_fit**1 + self.coef_ghg_2030[5]
            if n == 0:
                xr_temps = ndc_globalversion**n*self.coef_ghg_2030[self.settings['params']['polynomial_fit_2030relation']-n]
            else:
                xr_temps +=  ndc_globalversion**n*self.coef_ghg_2030[self.settings['params']['polynomial_fit_2030relation']-n]
        self.xr_temps = xr_temps

    # =========================================================== #
    # =========================================================== #

    def save(self):
        print("- Save")
        self.xr_temps.to_netcdf(self.settings['paths']['data']['datadrive']+'xr_temps.nc')
