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
        with open(self.current_dir / 'input.yml') as file:
            self.settings = yaml.load(file, Loader=yaml.FullLoader)
        self.countries_iso = np.load(self.settings['paths']['data']['datadrive'] + "all_countries.npy", allow_pickle=True)
        self.xr_dataread =  xr.open_dataset(self.settings['paths']['data']['datadrive'] + "xr_dataread.nc").load()
        self.xr_dataread_sub = self.xr_dataread.sel(Temperature=[1.5, 1.6, 2.0], Risk=[0.5, 0.33, 0.17], NegEmis=0.5, NonCO2red=0.5, Timing='Immediate', Scenario='SSP2')

    # =========================================================== #
    # =========================================================== #

    def global_default(self):
        '''
        Export default 1.5(6) and 2.0 pathways that roughly match the IPCC pathways
        '''
        self.data_15 = self.xr_dataread.sel(Timing='Immediate', NegEmis=0.5, Risk=0.5, NonCO2red=0.5, Temperature=[1.6])[['GHG_globe', 'CO2_globe', 'NonCO2_globe']].drop_vars(['Timing', 'NegEmis', 'Risk', 'NonCO2red']) # Using 1.6 as default temperature to match small overshoot in IPCC
        #self.data_15['Temperature'] = [1.5] # Not change temperature manually - that leads to confusion
        self.data_20 = self.xr_dataread.sel(Timing='Immediate', Risk=0.33, NegEmis=0.5, NonCO2red=0.5, Temperature=[2.0])[['GHG_globe', 'CO2_globe', 'NonCO2_globe']].drop_vars(['Timing', 'NegEmis', 'Risk', 'NonCO2red'])
        self.data = xr.merge([self.data_15, self.data_20]).sel(Time=np.arange(self.settings['params']['start_year_analysis'], 2101))
        self.data.to_dataframe().to_csv(self.settings['paths']['data']['export']+'../EffortSharingExports/emissionspathways_default.csv')

    # =========================================================== #
    # =========================================================== #

    def global_all(self):
        '''
        Export all pathways
        '''
        self.data = self.xr_dataread[['GHG_globe', 'CO2_globe', 'NonCO2_globe']].sel(Time=np.arange(self.settings['params']['start_year_analysis'], 2101))
        self.data.to_dataframe().to_csv(self.settings['paths']['data']['export']+'../EffortSharingExports/emissionspathways_all.csv')

    def reduce_country_files(self):
        '''
        Export all pathways
        '''
        for cty_i, cty in tqdm(enumerate(np.array(self.xr_dataread.Region))):
            ds = xr.open_dataset(self.settings['paths']['data']['export']+ "Allocations/xr_alloc_"+cty+".nc").sel(Time=np.array([2021]+list(np.arange(2025, 2101,5)))).expand_dims(Region=[cty])
            ds = ds.drop_vars(['PCB'])
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
                ds = xr.open_dataset(self.settings['paths']['data']['datadrive']+"/Allocations_CO2_excl/xr_alloc_"+cty+".nc")
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
            ds_total.to_dataframe().to_csv(self.settings['paths']['data']['export']+"../EffortSharingExports/allocations_default_"+['15overshoot', '20'][default_i]+"_CO2_excl.csv")
            ds_total_red.to_dataframe().to_csv(self.settings['paths']['data']['export']+"../EffortSharingExports/reductions_default_"+['15overshoot', '20'][default_i]+"_CO2_excl.csv")

    # =========================================================== #
    # =========================================================== #

    def ndcdata(self):
        '''
        Export NDC data
        '''
        self.xr_dataread.GHG_ndc.to_dataframe().to_csv(self.settings['paths']['data']['export']+'../EffortSharingExports/inputdata_ndc.csv')

    # =========================================================== #
    # =========================================================== #

    def sspdata(self):
        '''
        Export SSP data
        '''
        self.xr_dataread[['Population', 'GDP']].to_dataframe().to_csv(self.settings['paths']['data']['export']+'../EffortSharingExports/inputdata_ssp.csv')

    # =========================================================== #
    # =========================================================== #

    def emisdata(self):
        '''
        Export historical emission data
        '''
        self.xr_dataread[['GHG_hist', 'GHG_hist_excl', 'CO2_hist', 'CH4_hist', 'N2O_hist']].sel(Time=np.arange(1850, 1+self.settings['params']['start_year_analysis'])).to_dataframe().to_csv(self.settings['paths']['data']['export']+'../EffortSharingExports/inputdata_histemis.csv')

    # =========================================================== #
    # =========================================================== #

    def co2_budgets_ap(self):
        '''
        CO2 budgets AP
        '''

        xr_rbw = xr.open_dataset(self.settings['paths']['data']['datadrive'] + "xr_rbw_co2.nc").load()
        xrt = self.xr_dataread_sub.sel(Time=np.arange(self.settings['params']['start_year_analysis'], 2101))
        GDP_sum_w = xrt.GDP.sel(Region='EARTH')
        pop_sum_w = xrt.Population.sel(Region='EARTH')
        r1_nom = GDP_sum_w / pop_sum_w

        base_worldsum = xrt.CO2_base.sel(Region='EARTH')
        rb_part1 = (xrt.GDP / xrt.Population / r1_nom)**(1/3.)
        rb_part2 = xrt.CO2_base*(base_worldsum - xrt.CO2_globe)/base_worldsum
        rb = rb_part1 * rb_part2

        # Step 2: Correction factor
        corr_factor = (1e-9+xr_rbw.__xarray_dataarray_variable__)/(base_worldsum - xrt.CO2_globe)

        # Step 3: Budget after correction factor
        ap = self.xr_dataread_sub.CO2_base - rb/corr_factor
        self.xr_ap = (ap.sel(Time=np.arange(self.settings['params']['start_year_analysis'], 2101)).sel(Scenario='SSP2', NegEmis=0.5, NonCO2red=0.5, Timing='Immediate')*xr.where(xrt.CO2_globe > 0, 1, 0)).to_dataset(name="AP").sum(dim='Time')

    # =========================================================== #
    # =========================================================== #

    def co2_budgets_pc(self):
        '''
        CO2 budgets PC
        '''

        pop_region = self.xr_dataread_sub.sel(Time=self.settings['params']['start_year_analysis']).Population
        pop_earth = self.xr_dataread_sub.sel(Region='EARTH',
                                            Time=self.settings['params']['start_year_analysis']).Population
        pop_fraction =  pop_region / pop_earth
        self.xr_pc = (pop_fraction*self.xr_dataread_sub.Budget*1e3).to_dataset(name="PC")

    # =========================================================== #
    # =========================================================== #

    def co2_budgets_ecpc(self):
        '''
        CO2 budgets ECPC
        '''
        xrs = []
        for focusregion in tqdm(np.array(self.xr_dataread_sub.Region)):
            compensation_form_sqrt = np.sqrt(np.arange(0, 2101-self.settings['params']['start_year_analysis'])) #make sqrt curve
            compensation_form_sqrt = compensation_form_sqrt / np.sum(compensation_form_sqrt) #sum of values has to be 1

            xr_comp = xr.DataArray(compensation_form_sqrt, dims=['Time'], coords={'Time': np.arange(self.settings['params']['start_year_analysis'], 2101)})

            # Defining the timeframes for historical and future emissions
            hist_emissions_startyears = [1990]
            for startyear_i, startyear in enumerate(hist_emissions_startyears):
                hist_emissions_timeframe = np.arange(startyear, 1 + self.settings['params']['start_year_analysis'])
                future_emissions_timeframe = np.arange(self.settings['params']['start_year_analysis']+1, 2101)

                # Summing all historical emissions over the hist_emissions_timeframe
                hist_emissions = self.xr_dataread_sub.CO2_hist.sel(Time = hist_emissions_timeframe)

                # Discounting -> We only do past discounting here
                for discount_i, discount in enumerate([0]):
                    past_timeline = np.arange(startyear, self.settings['params']['start_year_analysis']+1)
                    xr_dc = xr.DataArray((1-discount/100)**(self.settings['params']['start_year_analysis']-past_timeline), dims=['Time'],
                                            coords={'Time': past_timeline})
                    hist_emissions_dc = (hist_emissions*xr_dc).sum(dim='Time')
                    hist_emissions_w = float(hist_emissions_dc.sel(Region='EARTH'))
                    hist_emissions_r = float(hist_emissions_dc.sel(Region = focusregion))

                    # CO2 budget
                    future_emissions_w = self.xr_dataread_sub.Budget*1e3
                    total_emissions_w = hist_emissions_w + future_emissions_w

                    # Calculating the cumulative population shares for region and world
                    cum_pop = self.xr_dataread_sub.Population.sel(Time = np.arange(self.settings['params']['start_year_analysis'], 2101)).sum(dim='Time')
                    cum_pop_r = cum_pop.sel(Region=focusregion)
                    cum_pop_w = cum_pop.sel(Region='EARTH')
                    share_cum_pop = cum_pop_r / cum_pop_w
                    budget_rightful = total_emissions_w * share_cum_pop
                    budget_left = budget_rightful - hist_emissions_r
                    ecpc = budget_left.to_dataset(name='ECPC')
            xrs.append(ecpc.expand_dims(Region=[focusregion]))
        self.xr_ecpc = xr.merge(xrs)

    # =========================================================== #
    # =========================================================== #

    def concat_co2budgets(self):
        '''
        CO2 budgets ECPC, AP and PC
        '''
        self.xr_budgets = xr.merge([self.xr_pc, self.xr_ecpc, self.xr_ap])
        self.xr_budgets = xr.merge([xr.where(self.xr_budgets.sel(Region='EARTH').expand_dims(['Region']), self.xr_budgets.sel(Region=self.countries_iso).sum(dim='Region'), 0), self.xr_budgets.drop_sel(Region='EARTH')])
        self.xr_budgets.to_netcdf(self.settings['paths']['data']['datadrive']+"CO2budgets.nc",format="NETCDF4", engine="netcdf4")
        self.xr_budgets.drop_vars(['Scenario', 'Time', 'NonCO2red', 'NegEmis', 'Timing']).to_dataframe().to_csv("K:/data/EffortSharingExports/CO2budgets.csv")

    # =========================================================== #
    # =========================================================== #

    def project_COMMITTED(self):
        '''
        Export files for COMMITTED
        '''
        # Pathways
        df = pd.read_csv("K:/Data/Data_effortsharing/EffortSharingExports/allocations_default_15overshoot_CO2_excl.csv")
        df = df[['Time', 'Region', 'PC', 'PCC', 'ECPC', 'AP']]
        df['Temperature'] = ["1.5 deg at 50% with small overshoot"]*len(df)

        df2 = pd.read_csv("K:/Data/Data_effortsharing/EffortSharingExports/allocations_default_20_CO2_excl.csv")
        df2 = df2[['Time', 'Region', 'PC', 'PCC', 'ECPC', 'AP']]
        df2['Temperature'] = ["2.0 deg at 67%"]*len(df2)

        df3 = pd.concat([df, df2])
        df3.to_csv("K:/Data/Data_effortsharing/EffortSharingExports/allocations_CO2_excl_COMMITTED.csv", index=False)

        # # Budgets
        # xr_traj_16 = xr.open_dataset(self.settings['paths']['data']['datadrive']+"/xr_traj_t16_r50.nc")
        # xr_traj_20 = xr.open_dataset(self.settings['paths']['data']['datadrive']+"/xr_traj_t20_r67.nc")
        # self.ecpc = xr_traj_16.ECPC.sum(dim='Time')

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

if __name__ == "__main__":
    # region = input("Choose a focus country or region: ")

    dataexporter = dataexportcl()
    # dataexporter.allocations_default()
    dataexporter.project_COMMITTED()
