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

class allocation(object):

    # =========================================================== #
    # =========================================================== #

    def __init__(self, reg):
        self.current_dir = Path.cwd()

        # Read in Input YAML file
        with open(self.current_dir / 'input.yml') as file:
            self.settings = yaml.load(file, Loader=yaml.FullLoader)
        self.countries_iso = np.load(self.settings['paths']['data']['datadrive'] + "all_countries.npy", allow_pickle=True)
        self.xr_total = xr.open_dataset(self.settings['paths']['data']['datadrive'] + "xr_dataread.nc").load()
        
        # Region and Time variables
        self.FocusRegion = reg
        self.start_year_analysis = self.settings['params']['start_year_analysis']
        self.analysis_timeframe = np.arange(self.start_year_analysis, 2101)

    # =========================================================== #
    # =========================================================== #

    def gf(self):
        '''
        Grandfathering: Divide the global budget over the regions based on 
        their historical emissions
        '''
        # Calculating the current GHG fraction for region and world based on start_year_analysis
        current_GHG_region = self.xr_total.sel(Region=self.FocusRegion,
                                                 Time=self.start_year_analysis).GHG_hist
        
        current_GHG_earth = (1e-9+self.xr_total.sel(Region='EARTH', 
                                                      Time=self.start_year_analysis).GHG_hist)
    
        emis_fraction = current_GHG_region / current_GHG_earth
        
        # New emission time series from the start_year to 2101 by multiplying the global budget with the fraction
        xr_new = (emis_fraction*self.xr_total.GHG_globe).sel(Time=self.analysis_timeframe)
        
        # Adds the new emission time series to the xr_total dataset
        self.xr_total = self.xr_total.assign(GF = xr_new)

    # =========================================================== #
    # =========================================================== #

    def pc(self):
        '''
        Per Capita: Divide the global budget equally per capita
        '''
        
        pop_region = self.xr_total.sel(Region=self.FocusRegion, 
                                       Time=self.start_year_analysis).Population
        pop_earth = self.xr_total.sel(Region=self.countries_iso, 
                                      Time=self.start_year_analysis).Population.sum(dim=['Region'])
        pop_fraction =  pop_region / pop_earth
        
        # Multiplying the global budget with the population fraction to create 
        # new allocation time series from start_year to 2101
        xr_new = (pop_fraction*self.xr_total.GHG_globe).sel(Time=self.analysis_timeframe)
        self.xr_total = self.xr_total.assign(PC = xr_new)

    # =========================================================== #
    # =========================================================== #

    def pcc(self):
        '''
        Per Capita Convergence: Grandfathering converging into per capita
        '''
        # Define function to transform fractions until 2150
        # def transform_time(time, convyear):
        #     fractions = []
        #     for t in time:
        #         if t < self.start_year_analysis+1:
        #             fractions.append(1)
        #         elif t < convyear:
        #             fractions.append(1-(t-self.start_year_analysis) / (convyear-self.start_year_analysis))
        #         else:
        #             fractions.append(0)
        #     return fractions
        
        def transform_time(time, convyear):
            '''
            Function that calculates the convergence based on 
            the convergence time frame
            '''
            fractions = pd.DataFrame({'Year': time, 
                                      'Convergence': 0}, dtype=float)

            fractions.loc[fractions['Year'] < self.start_year_analysis + 1, 'Convergence'] = 1
            fractions.loc[(fractions['Year'] >= self.start_year_analysis + 1) & 
                          (fractions['Year'] < convyear), 'Convergence'] = \
                1 - (fractions['Year'] - self.start_year_analysis) / (convyear - self.start_year_analysis)
            fractions.loc[fractions['Year'] >= convyear, 'Convergence'] = 0

            return fractions['Convergence'].tolist()

        gfdeel = []
        pcdeel = []
        
        convergence_years = np.arange(self.settings['params']['pcc_conv_start'],
                                      self.settings['params']['pcc_conv_end']+1, 5)
        
        for year in convergence_years:
            ar = np.array([transform_time(self.xr_total.Time, year)]).T
            gfdeel.append(xr.DataArray(data=ar, 
                                       dims=['Time', 'Convergence_year'], 
                                       coords=dict(Time=self.xr_total.Time, 
                                                   Convergence_year=[year])).to_dataset(name="PCC"))
            pcdeel.append(xr.DataArray(data=1-ar, 
                                       dims=['Time', 'Convergence_year'], 
                                       coords=dict(Time=self.xr_total.Time, 
                                                   Convergence_year=[year])).to_dataset(name="PCC"))
        
        # Merging the list of DataArays into one Dataset
        gfdeel_single = xr.merge(gfdeel)
        pcdeel_single = xr.merge(pcdeel)
        
        # Creating new allocation time series by multiplying convergence fractions
        # with existing GF and PC allocations
        xr_new = (gfdeel_single*self.xr_total.GF + pcdeel_single*self.xr_total.PC)['PCC']
        self.xr_total = self.xr_total.assign(PCC = xr_new)
    # =========================================================== #
    # =========================================================== #

    def ecpc(self):
        '''
        Equal Cumulative per Capita: Uses historical emissions, discount factors and 
        population shares to allocate the global budget
        '''
        compensation_form = np.array(list(np.linspace(0, 1, len(np.arange(self.start_year_analysis, 2040))))+[1]*len(np.arange(2040, 2101)))
        compensation_form2 = np.convolve(compensation_form, np.ones(3)/3, mode='valid')
        compensation_form[1:-1] = compensation_form2
        compensation_form = compensation_form - compensation_form[0]
        compensation_form = compensation_form / np.sum(compensation_form)
        xr_comp = xr.DataArray(compensation_form, dims=['Time'], 
                                coords={'Time': self.analysis_timeframe})

        # Defining the timeframes for historical and future emissions
        xrs = []
        hist_emissions_startyears = [1850, 1950, 1990]
        for startyear_i, startyear in enumerate(hist_emissions_startyears):
            hist_emissions_timeframe = np.arange(startyear, 1 + self.start_year_analysis)
            future_emissions_timeframe = np.arange(self.start_year_analysis+1, 2101)

            # Summing all historical emissions over the hist_emissions_timeframe
            hist_emissions = self.xr_total.GHG_hist.sel(Time = hist_emissions_timeframe)

            # Discounting -> We only do past discounting here
            for discount_i, discount in enumerate([1.6, 2.0, 2.8]):
                past_timeline = np.arange(startyear, self.start_year_analysis+1)
                xr_dc = xr.DataArray((1-discount/100)**(self.start_year_analysis-past_timeline), dims=['Time'], 
                                        coords={'Time': past_timeline})
                hist_emissions_dc = (hist_emissions*xr_dc).sum(dim='Time')
                hist_emissions_w = float(hist_emissions_dc.sel(Region='EARTH'))
                hist_emissions_r = float(hist_emissions_dc.sel(Region = self.FocusRegion))

                # Summing all future emissions over the future_emissions_timeframe
                future_emissions_w = self.xr_total.GHG_globe.sel(Time = future_emissions_timeframe).sum(dim='Time')

                total_emissions_w = hist_emissions_w + future_emissions_w

                # Calculating the cumulative population shares for region and world
                cum_pop = self.xr_total.Population.sel(Time = self.analysis_timeframe).sum(dim='Time')
                cum_pop_r = cum_pop.sel(Region=self.FocusRegion)
                cum_pop_w = cum_pop.sel(Region='EARTH')
                share_cum_pop = cum_pop_r / cum_pop_w
                budget_rightful = total_emissions_w * share_cum_pop
                budget_left = budget_rightful - hist_emissions_r

                # Now temporal allocation
                #globalbudget = self.xr_total.GHG_globe.sel(Time=self.analysis_timeframe).sum(dim='Time')
                globalpath = self.xr_total.GHG_globe

                emis_2021_i = self.xr_total.GHG_hist.sel(Time=self.start_year_analysis, 
                                                            Region=self.FocusRegion)
                emis_2021_w = self.xr_total.GHG_hist.sel(Time=self.start_year_analysis, 
                                                            Region='EARTH')
                path_scaled_0 = emis_2021_i/emis_2021_w*globalpath
                budget_without_assumptions = path_scaled_0.sum(dim='Time')
                budget_surplus = budget_left - budget_without_assumptions

                def ecpc_factor(f):
                    return path_scaled_0+xr_comp*f

                ecpc = ecpc_factor(budget_surplus).expand_dims(Discount_factor=[discount], Historical_startyear=[startyear]).to_dataset(name='ECPC')
                xrs.append(ecpc)
        xr_ecpc = xr.merge(xrs)
        self.xr_total = self.xr_total.assign(ECPC = xr_ecpc.ECPC)
        
    # =========================================================== #
    # =========================================================== #

    def ap(self):
        '''
        Ability to Pay: Uses GDP per capita to allocate the global budget
        '''
        xr_rbw = xr.open_dataset(self.settings['paths']['data']['datadrive'] + "xr_rbw.nc").load()
        xrt = self.xr_total.sel(Time=self.analysis_timeframe)
        GDP_sum_w = xrt.GDP.sel(Region='EARTH')
        pop_sum_w = xrt.Population.sel(Region='EARTH')
        r1_nom = GDP_sum_w / pop_sum_w
        
        base_worldsum = xrt.GHG_base.sel(Region='EARTH')
        rb_part1 = (xrt.GDP.sel(Region=self.FocusRegion) / xrt.Population.sel(Region=self.FocusRegion) / r1_nom)**(1/3.)
        rb_part2 = xrt.GHG_base.sel(Region=self.FocusRegion)*(base_worldsum - xrt.GHG_globe)/base_worldsum
        rb = rb_part1*rb_part2
        
        ap = self.xr_total.GHG_base.sel(Region=self.FocusRegion) - rb/(1e-9+xr_rbw.__xarray_dataarray_variable__)*(base_worldsum - self.xr_total.GHG_globe)
        ap = ap.sel(Time=self.analysis_timeframe)
        self.xr_total = self.xr_total.assign(AP = ap)
        xr_rbw.close()

    # =========================================================== #
    # =========================================================== #

    def gdr(self):
        # Read RCI
        df_rci = pd.read_csv(self.settings['paths']['data']['external'] + "RCI/RCI.xls", 
                             delimiter='\t', 
                             skiprows=30)[:-2]
        df_rci = df_rci[['iso3', 'year', 'rci']]
        df_rci['year'] = df_rci['year'].astype(int)
        df_rci = df_rci.rename(columns={"iso3": 'Region', 'year': 'Time'})
        dfdummy = df_rci.set_index(['Region', 'Time'])
        xr_rci = xr.Dataset.from_dataframe(dfdummy)
        xr_rci = xr_rci.reindex({"Region": self.xr_total.Region})
        if self.FocusRegion != 'EU': rci_reg = xr_rci.rci.sel(Region=self.FocusRegion)
        else:
            df = pd.read_excel("X:/user/dekkerm/Data/UNFCCC_Parties_Groups_noeu.xlsx", sheet_name = "Country groups")
            countries_iso = np.array(df["Country ISO Code"])
            group_eu = countries_iso[np.array(df["EU"]) == 1]
            rci_reg = xr_rci.rci.sel(Region=group_eu).sum(dim='Region') # TODO check if this should be a mean instead of a sum

        # Compute GDR
        gdr = self.xr_total.GHG_base.sel(Region=self.FocusRegion) \
            - (self.xr_total.GHG_base.sel(Region='EARTH') \
                - self.xr_total.GHG_globe)*rci_reg
        yearfracs = xr.Dataset(data_vars={"Value": (['Time'], 
                                                    (self.analysis_timeframe - 2030) \
                                                        / (self.settings['params']['convergence_year_gdr'] - 2030))}, 
                               coords={"Time": self.analysis_timeframe})
        gdr = gdr.rename('Value')
        gdr_post2030 = ((self.xr_total.GHG_base.sel(Region=self.FocusRegion) \
            - (self.xr_total.GHG_base.sel(Region='EARTH', Time=self.analysis_timeframe) \
                - self.xr_total.GHG_globe.sel(Time=self.analysis_timeframe))*rci_reg.sel(Time=2030))*(1-yearfracs) \
                    + yearfracs*self.xr_total.AP.sel(Time=self.analysis_timeframe)).sel(Time=np.arange(2031, 2101))
        gdr_total = xr.merge([gdr, gdr_post2030])
        gdr_total = gdr_total.rename({'Value': 'GDR'})
        self.xr_total = self.xr_total.assign(GDR = gdr_total.GDR)
    # =========================================================== #
    # =========================================================== #

    def save(self):
        xr_total_onlyalloc = self.xr_total.drop_vars(['Population', "CO2_hist", "CO2_globe", "N2O_hist", "CH4_hist", 'GDP', 'GHG_hist', 'GHG_globe', "NonCO2_globe", "GHG_hist_all", 'GHG_base', 'GHG_ndc', 'Hot_air', 'Conditionality', 'Ambition', 'Budget']).sel(Region=self.FocusRegion, Time=np.arange(self.settings['params']['start_year_analysis'], 2101)).astype("float32")

        xr_total_onlyalloc.to_netcdf(self.settings['paths']['data']['datadrive']+'Allocations/xr_alloc_'+self.FocusRegion+'.nc',         
            # encoding={
            #     "Scenario": {"dtype": "str"},
            #     "Time": {"dtype": "int"},

            #     "Temperature": {"dtype": "float"},
            #     "NegEmis": {"dtype": "float"},
            #     "Risk": {"dtype": "float"},
            #     "TrajUnc": {"dtype": "str"},

            #     "Convergence_year": {"dtype": "int"},

            #     "GF": {"zlib": True, "complevel": 9},
            #     "PC": {"zlib": True, "complevel": 9},
            #     "PCC": {"zlib": True, "complevel": 9},
            #     "ECPC": {"zlib": True, "complevel": 9},
            #     "AP": {"zlib": True, "complevel": 9},
            #     "GDR": {"zlib": True, "complevel": 9},
            # },
            format='NETCDF4'
        )
        self.xr_alloc = xr_total_onlyalloc
        self.xr_total.close()
        
if __name__ == "__main__":
    region = input("Choose a focus country or region: ")
    allocator = allocation(region)
    allocator.gf()  
    allocator.pc()
    allocator.pcc()
    allocator.ecpc()
    allocator.ap()
    allocator.gdr()
    allocator.save()
    # print("allocator")
        