# ======================================== #
# Class that does the budget allocation
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
import json
from scipy.signal import savgol_filter

# =========================================================== #
# CLASS OBJECT
# =========================================================== #

class dataexportcl(object):

    # =========================================================== #
    # =========================================================== #

    def __init__(self):
        self.current_dir = Path.cwd()

        # Read in Input YAML file
        with open(self.current_dir / '../input.yml') as file:
            self.settings = yaml.load(file, Loader=yaml.FullLoader)
        self.countries_iso = np.load(self.settings['paths']['data']['datadrive'] + "all_countries.npy", allow_pickle=True)
        self.xr_dataread =  xr.open_dataset(self.settings['paths']['data']['datadrive'] + "xr_dataread.nc").load()

    # =========================================================== #
    # =========================================================== #

    def global_default(self):
        '''
        Export default 1.5(6) and 2.0 pathways that roughly match the IPCC pathways
        '''
        self.data_15 = self.xr_dataread.sel(Timing='Immediate', NegEmis=0.5, Risk=0.5, NonCO2red=0.5, Temperature=[1.6])[['GHG_globe', 'CO2_globe', 'NonCO2_globe']].drop_vars(['Timing', 'NegEmis', 'Risk', 'variable', 'NonCO2red']) # Using 1.6 as default temperature to match small overshoot in IPCC
        self.data_15['Temperature'] = [1.5]
        self.data_20 = self.xr_dataread.sel(Timing='Immediate', Risk=0.33, NegEmis=0.5, NonCO2red=0.5, Temperature=[2.0])[['GHG_globe', 'CO2_globe', 'NonCO2_globe']].drop_vars(['Timing', 'NegEmis', 'Risk', 'variable', 'NonCO2red'])
        self.data = xr.merge([self.data_15, self.data_20]).sel(Time=np.arange(self.settings['params']['start_year_analysis'], 2101))
        self.data.to_dataframe().to_csv(self.settings['paths']['data']['datadrive']+'../EffortSharingExports/paths_default.csv')

    # =========================================================== #
    # =========================================================== #

    def global_all(self):
        '''
        Export all pathways
        '''
        self.data = self.xr_dataread[['GHG_globe', 'CO2_globe', 'NonCO2_globe']].drop_vars(['variable']).sel(Time=np.arange(self.settings['params']['start_year_analysis'], 2101))
        self.data.to_dataframe().to_csv(self.settings['paths']['data']['datadrive']+'../EffortSharingExports/paths_all.csv')

    # =========================================================== #
    # =========================================================== #

    def allocations_temprisk(self, temp, risk):
        '''
        Export all allocations for a given temperature and risk level
        '''

    # =========================================================== #
    # =========================================================== #

    def allocations_year(self, year):
        '''
        Export all allocations for a given year
        '''


    # =========================================================== #
    # =========================================================== #

    def ndcdata(self):
        '''
        Export NDC data
        '''
        self.xr_dataread.GHG_ndc.drop_vars(['variable']).to_dataframe().to_csv(self.settings['paths']['data']['datadrive']+'../EffortSharingExports/dataext_ndc.csv')

    # =========================================================== #
    # =========================================================== #

    def sspdata(self):
        '''
        Export SSP data
        '''
        self.xr_dataread[['Population', 'GDP']].drop_vars(['variable']).to_dataframe().to_csv(self.settings['paths']['data']['datadrive']+'../EffortSharingExports/dataext_ssp.csv')

    # =========================================================== #
    # =========================================================== #

    def emisdata(self):
        '''
        Export historical emission data
        '''
        self.xr_dataread[['GHG_hist', 'GHG_hist_excl', 'CO2_hist', 'CH4_hist', 'N2O_hist']].drop_vars(['variable']).sel(Time=np.arange(1850, 1+self.settings['params']['start_year_analysis'])).to_dataframe().to_csv(self.settings['paths']['data']['datadrive']+'../EffortSharingExports/dataext_histemis.csv')

    # =========================================================== #
    # =========================================================== #

    def project_PBLreport(self):
        '''
        Export files for ELEVATE
        '''

    # =========================================================== #
    # =========================================================== #

    def project_ELEVATE(self):
        '''
        Export files for ELEVATE
        '''

    # =========================================================== #
    # =========================================================== #

    def project_COMMITTED(self):
        '''
        Export files for COMMITTED
        '''

    # =========================================================== #
    # =========================================================== #

    def project_DGCLIMA(self):
        '''
        Export files for DG-CLIMA
        '''

    # =========================================================== #
    # =========================================================== #

    def project_Yann(self):
        '''
        Export files for Yann
        '''

if __name__ == "__main__":
    region = input("Choose a focus country or region: ")
    dataexporter = dataexportcl(region)
