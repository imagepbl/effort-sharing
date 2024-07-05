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
from tqdm import tqdm
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
        self.data.to_dataframe().to_csv(self.settings['paths']['data']['export']+'../EffortSharingExports/emissionspathways_default.csv')

    # =========================================================== #
    # =========================================================== #

    def global_all(self):
        '''
        Export all pathways
        '''
        self.data = self.xr_dataread[['GHG_globe', 'CO2_globe', 'NonCO2_globe']].drop_vars(['variable']).sel(Time=np.arange(self.settings['params']['start_year_analysis'], 2101))
        self.data.to_dataframe().to_csv(self.settings['paths']['data']['export']+'../EffortSharingExports/emissionspathways_all.csv')

    def reduce_country_files(self):
        '''
        Export all pathways
        '''
        for cty_i, cty in tqdm(enumerate(np.array(self.xr_dataread.Region))):
            ds = xr.open_dataset(self.settings['paths']['data']['export']+ "Allocations/xr_alloc_"+cty+".nc").sel(Time=np.array([2021]+list(np.arange(2025, 2101,5)))).expand_dims(Region=[cty])
            ds = ds.drop_vars(['variable', 'PCB'])
            ds.to_netcdf(self.settings['paths']['data']['export']+"../EffortSharingExports/Allocations/allocations_"+cty+".nc",
                        encoding={
                            "GF": {"zlib": True, "complevel": 9},
                            "PC": {"zlib": True, "complevel": 9},
                            "PCC": {"zlib": True, "complevel": 9},
                            "PCB_lin": {"zlib": True, "complevel": 9},
                            "GDR": {"zlib": True, "complevel": 9},
                            "ECPC": {"zlib": True, "complevel": 9},
                            "AP": {"zlib": True, "complevel": 9},
                        }, format='NETCDF4', engine="netcdf4")
            ds.close()

    # =========================================================== #
    # =========================================================== #

    def allocations_default(self):
        '''
        Export default emission allocations and reductions
        '''
        for default_i in range(2):
            dss = []
            for cty in np.array(self.xr_dataread.Region):
                ds = xr.open_dataset(self.settings['paths']['data']['datadrive']+"/Allocations/xr_alloc_"+cty+".nc")
                ds2 = ds.sel(Discount_factor=0,
                Historical_startyear=1990,
                Capability_threshold='Th',
                RCI_weight='Half',
                Scenario='SSP2',
                Convergence_year=2050,
                Time=np.array([2021]+list(np.arange(2025, 2101,5))),
                Risk=[0.5, 0.33][default_i],
                NegEmis=0.5,
                Temperature=[1.6, 2.0][default_i],
                NonCO2red=0.5,
                Timing='Immediate').drop_vars(['Scenario', 'Convergence_year', 'Discount_factor', 'Historical_startyear', 'Capability_threshold', 'RCI_weight', 'NegEmis', 'Temperature', 'Risk', 'NonCO2red', 'Timing', 'PCB'])
                dss.append(ds2.expand_dims(Region=[cty]))
                ds.close()
            ds_total = xr.merge(dss)
            cur = self.xr_dataread.GHG_hist.sel(Time=2015)
            ds_total_red = -(cur-ds_total)/cur
            ds_total.drop_vars(['variable']).to_dataframe().to_csv(self.settings['paths']['data']['export']+"../EffortSharingExports/allocations_default_"+['15overshoot', '20'][default_i]+".csv")
            ds_total_red.drop_vars(['variable']).to_dataframe().to_csv(self.settings['paths']['data']['export']+"../EffortSharingExports/reductions_default_"+['15overshoot', '20'][default_i]+".csv")

    # =========================================================== #
    # =========================================================== #

    def ndcdata(self):
        '''
        Export NDC data
        '''
        self.xr_dataread.GHG_ndc.drop_vars(['variable']).to_dataframe().to_csv(self.settings['paths']['data']['export']+'../EffortSharingExports/inputdata_ndc.csv')

    # =========================================================== #
    # =========================================================== #

    def sspdata(self):
        '''
        Export SSP data
        '''
        self.xr_dataread[['Population', 'GDP']].drop_vars(['variable']).to_dataframe().to_csv(self.settings['paths']['data']['export']+'../EffortSharingExports/inputdata_ssp.csv')

    # =========================================================== #
    # =========================================================== #

    def emisdata(self):
        '''
        Export historical emission data
        '''
        self.xr_dataread[['GHG_hist', 'GHG_hist_excl', 'CO2_hist', 'CH4_hist', 'N2O_hist']].drop_vars(['variable']).sel(Time=np.arange(1850, 1+self.settings['params']['start_year_analysis'])).to_dataframe().to_csv(self.settings['paths']['data']['export']+'../EffortSharingExports/inputdata_histemis.csv')

    # =========================================================== #
    # =========================================================== #

    def project_PBLreport(self):
        '''
        Export files for ELEVATE
        '''

    # =========================================================== #
    # =========================================================== #

    def project_COMMITTED(self):
        '''
        Export files for COMMITTED
        '''
        df = pd.read_csv("K:/data/EffortSharingExports/allocations_default_15overshoot.csv")
        df = df[['Time', 'Region', 'PCC', 'ECPC', 'AP']]
        df['Temperature'] = ["1.5 deg at 50% with small overshoot"]*len(df)

        df2 = pd.read_csv("K:/data/EffortSharingExports/allocations_default_20.csv")
        df2 = df2[['Time', 'Region', 'PCC', 'ECPC', 'AP']]
        df2['Temperature'] = ["2.0 deg at 67%"]*len(df2)

        df3 = pd.concat([df, df2])
        df3.to_csv("K:/data/EffortSharingExports/allocations_COMMITTED.csv", index=False)

    # =========================================================== #
    # =========================================================== #

    def project_DGIS(self):
        '''
        Export files for DGIS
        '''
        df = pd.read_csv("K:/data/EffortSharingExports/allocations_default_15overshoot.csv")
        df = df[['Time', 'Region', 'PCC', 'ECPC', 'AP']]
        df['Temperature'] = ["1.5 deg at 50% with small overshoot"]*len(df)

        df2 = pd.read_csv("K:/data/EffortSharingExports/allocations_default_20.csv")
        df2 = df2[['Time', 'Region', 'PCC', 'ECPC', 'AP']]
        df2['Temperature'] = ["2.0 deg at 67%"]*len(df2)

        df3 = pd.concat([df, df2])
        df3.to_csv("K:/data/EffortSharingExports/allocations_DGIS.csv", index=False)

    # =========================================================== #
    # =========================================================== #

    def project_Yann(self):
        '''
        Export files for Yann
        '''

if __name__ == "__main__":
    region = input("Choose a focus country or region: ")
    dataexporter = dataexportcl(region)
