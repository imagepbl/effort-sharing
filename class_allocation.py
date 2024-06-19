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

    def __init__(self, reg, version = 'normal'): # Now modulating version (PBL or normal temperature increments) via the version argument
        self.current_dir = Path.cwd()
        self.version = version
        if version == 'PBL': self.version_path = 'PBL/'
        elif version == 'normal': self.version_path = ''

        # Read in Input YAML file
        with open(self.current_dir / 'input.yml') as file:
            self.settings = yaml.load(file, Loader=yaml.FullLoader)
        self.countries_iso = np.load(self.settings['paths']['data']['datadrive'] + "all_countries.npy", allow_pickle=True)
        self.xr_total = xr.open_dataset(self.settings['paths']['data']['datadrive'] + self.version_path+ "xr_dataread.nc").load()
        
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

    def pcb(self):
        '''
        Per capita on a budget basis
        '''
        # co2 part
        def budget_harm(nz):
            compensation_form = np.sqrt(np.arange(0, 2101-self.settings['params']['start_year_analysis']))
            xr_comp2 =  xr.DataArray(compensation_form, dims=['Time'], coords={'Time': np.arange(self.settings['params']['start_year_analysis'], 2101)})
            return xr_comp2 / ((nz-2021)**(3/2)*(2/3)) # TODO later: should be , but I now calibrated to 0.5. Not a problem because we have the while loop later.
        def pcb_new_factor(path, f):
            netzeros = 2021+path.where(path > 0, 0).where(path < 0 , 1).sum(dim='Time')
            netzeros = netzeros.where(netzeros < 2100, 2100)
            return path+budget_harm(netzeros)*f

        pop_region = self.xr_total.sel(Time=self.start_year_analysis).Population
        pop_earth = self.xr_total.sel(Region=self.countries_iso, 
                                    Time=self.start_year_analysis).Population.sum(dim=['Region'])
        pop_fraction =  (pop_region / pop_earth).mean(dim='Scenario')
        globalpath = self.xr_total.CO2_globe

        emis_2021_i = self.xr_total.CO2_hist.sel(Time=self.start_year_analysis)
        emis_2021_w = self.xr_total.CO2_hist.sel(Time=self.start_year_analysis, 
                                                    Region='EARTH')
        path_scaled_0 = (emis_2021_i/emis_2021_w*globalpath).sel(Time=np.arange(self.start_year_analysis, 2101)).sel(Region=self.FocusRegion)
        budget_left = (self.xr_total.CO2_globe.where(self.xr_total.CO2_globe > 0, 0).sel(Time=np.arange(self.start_year_analysis, 2101)).sum(dim='Time')*pop_fraction).sel(Region=self.FocusRegion)

        budget_without_assumptions_prepeak = path_scaled_0.where(path_scaled_0 > 0, 0).sum(dim='Time')
        budget_surplus = (budget_left - budget_without_assumptions_prepeak)
        pcb = pcb_new_factor(path_scaled_0, budget_surplus).to_dataset(name='PCB')

        # Optimize to bend the CO2 curves as close as possible to the CO2 budgets
        it=0
        while it < 3:
            pcb_pos = pcb.where(pcb > 0, 0).sum(dim='Time')
            budget_surplus = (budget_left - pcb_pos).PCB
            pcb = pcb_new_factor(pcb.PCB, budget_surplus).to_dataset(name='PCB')
            it+=1

        # CO2, but now linear
        nz = (budget_left*2/self.xr_total.CO2_hist.sel(Region=self.FocusRegion, Time=self.start_year_analysis)+self.start_year_analysis-1)
        coef = self.xr_total.CO2_hist.sel(Region=self.FocusRegion, Time=self.start_year_analysis)/(nz-self.start_year_analysis)
        linear_co2 = -coef*xr.DataArray(np.arange(0, 2101-self.start_year_analysis), dims=['Time'], coords={'Time': np.arange(self.settings['params']['start_year_analysis'], 2101)})+self.xr_total.CO2_hist.sel(Region=self.FocusRegion, Time=self.start_year_analysis)
        linear_co2_pos = linear_co2.where(linear_co2 > 0, 0).to_dataset(name='PCB_lin')

        # Non-co2 part
        nonco2_current = self.xr_total.GHG_hist.sel(Time=self.start_year_analysis) - self.xr_total.CO2_hist.sel(Time=self.start_year_analysis)
        nonco2_fraction = nonco2_current / nonco2_current.sel(Region='EARTH')
        nonco2_part_gf = nonco2_fraction*self.xr_total.NonCO2_globe

        pc_fraction = self.xr_total.Population.sel(Time=self.start_year_analysis) / self.xr_total.Population.sel(Time=self.start_year_analysis, Region='EARTH')
        nonco2_part_pc = pc_fraction*self.xr_total.NonCO2_globe

        compensation_form = np.array(list(np.linspace(0, 1, len(np.arange(self.settings['params']['start_year_analysis'], 2040))))+[1]*len(np.arange(2040, 2101)))
        xr_comp =  xr.DataArray(compensation_form, dims=['Time'], coords={'Time': np.arange(self.settings['params']['start_year_analysis'], 2101)})

        nonco2_part = nonco2_part_gf*(1-xr_comp) + nonco2_part_pc*(xr_comp)

        # together:
        self.ghg_pcb = pcb + nonco2_part.sel(Region=self.FocusRegion)
        self.ghg_pcb_lin = linear_co2_pos + nonco2_part.sel(Region=self.FocusRegion)
        self.xr_total = self.xr_total.assign(PCB = self.ghg_pcb.PCB)
        self.xr_total = self.xr_total.assign(PCB_lin = self.ghg_pcb_lin.PCB_lin)

        # Linear pathway down code (for now not used)
        # xr_comp =  xr.DataArray(np.arange(0, 2101-self.settings['params']['start_year_analysis']), dims=['Time'], coords={'Time': np.arange(self.settings['params']['start_year_analysis'], 2101)})
        # pop_region = self.xr_total.sel(Time=self.start_year_analysis).Population
        # pop_earth = self.xr_total.sel(Region=self.countries_iso, 
        #                             Time=self.start_year_analysis).Population.sum(dim=['Region'])
        # pop_fraction =  (pop_region / pop_earth).mean(dim='Scenario')

        # netzeros_delta = (2*self.xr_total.Budget*pop_fraction*1000)/self.xr_total.CO2_hist.sel(Time=2021)
        # co2_part = (self.xr_total.CO2_hist.sel(Time=2021)-(self.xr_total.CO2_hist.sel(Time=2021)/netzeros_delta)*xr_comp)
        # co2_part = co2_part.where(co2_part >= 0, 0)

        # nonco2_current = self.xr_total.GHG_hist.sel(Time=self.start_year_analysis) - self.xr_total.CO2_hist.sel(Time=self.start_year_analysis)
        # nonco2_fraction = nonco2_current / nonco2_current.sel(Region='EARTH')
        # nonco2_part = nonco2_fraction*self.xr_total.NonCO2_globe

    # =========================================================== #
    # =========================================================== #

    def ecpc(self):
        '''
        Equal Cumulative per Capita: Uses historical emissions, discount factors and 
        population shares to allocate the global budget
        '''
        compensation_form_sqrt = np.sqrt(np.arange(0, 2101-self.settings['params']['start_year_analysis'])) #make sqrt curve
        compensation_form_sqrt = compensation_form_sqrt / np.sum(compensation_form_sqrt) #sum of values has to be 1

        xr_comp = xr.DataArray(compensation_form_sqrt, dims=['Time'], coords={'Time': self.analysis_timeframe})

        # Defining the timeframes for historical and future emissions
        xrs = []
        hist_emissions_startyears = [1850, 1950, 1990]
        for startyear_i, startyear in enumerate(hist_emissions_startyears):
            hist_emissions_timeframe = np.arange(startyear, 1 + self.start_year_analysis)
            future_emissions_timeframe = np.arange(self.start_year_analysis+1, 2101)

            # Summing all historical emissions over the hist_emissions_timeframe
            hist_emissions = self.xr_total.GHG_hist.sel(Time = hist_emissions_timeframe)

            # Discounting -> We only do past discounting here
            for discount_i, discount in enumerate([0, 1.6, 2.0, 2.8]):
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
        Equation from van den Berg et al. (2020)
        '''
        # Step 1: Reductions before correction factor
        xr_rbw = xr.open_dataset(self.settings['paths']['data']['datadrive'] + self.version_path + "xr_rbw.nc").load()
        xrt = self.xr_total.sel(Time=self.analysis_timeframe)
        GDP_sum_w = xrt.GDP.sel(Region='EARTH')
        pop_sum_w = xrt.Population.sel(Region='EARTH')
        r1_nom = GDP_sum_w / pop_sum_w
        
        base_worldsum = xrt.GHG_base.sel(Region='EARTH')
        rb_part1 = (xrt.GDP.sel(Region=self.FocusRegion) / xrt.Population.sel(Region=self.FocusRegion) / r1_nom)**(1/3.)
        rb_part2 = xrt.GHG_base.sel(Region=self.FocusRegion)*(base_worldsum - xrt.GHG_globe)/base_worldsum
        rb = rb_part1 * rb_part2

        # Step 2: Correction factor
        corr_factor = (1e-9+xr_rbw.__xarray_dataarray_variable__)/(base_worldsum - xrt.GHG_globe)
        
        # Step 3: Budget after correction factor
        ap = self.xr_total.GHG_base.sel(Region=self.FocusRegion) - rb/corr_factor
        
        ap = ap.sel(Time=self.analysis_timeframe)
        self.xr_total = self.xr_total.assign(AP = ap)
        xr_rbw.close()

    # =========================================================== #
    # =========================================================== #

    def gdr(self):
        '''
        Greenhouse Development Rights: Uses the Responsibility-Capability Index
        (RCI) weighed at 50/50 to allocate the global budget
        Calculations from van den Berg et al. (2020)
        '''
        xr_rci = xr.open_dataset(self.settings['paths']['data']['datadrive'] + self.version_path + "xr_rci.nc").load()
        yearfracs = xr.Dataset(data_vars={"Value": (['Time'],
                                                    (self.analysis_timeframe - 2030) \
                                                        / (self.settings['params']['convergence_year_gdr'] - 2030))},
                                coords={"Time": self.analysis_timeframe})
        
        # Get the regional RCI values
        # If region is EU, we have to sum over the EU countries
        if self.FocusRegion != 'EU':
            rci_reg = xr_rci.rci.sel(Region=self.FocusRegion)
        else:
            df = pd.read_excel("X:/user/dekkerm/Data/UNFCCC_Parties_Groups_noeu.xlsx", sheet_name = "Country groups")
            countries_iso = np.array(df["Country ISO Code"])
            group_eu = countries_iso[np.array(df["EU"]) == 1]
            rci_reg = xr_rci.rci.sel(Region=group_eu).sum(dim='Region')

        # Compute GDR until 2030
        baseline = self.xr_total.GHG_base
        global_traject = self.xr_total.GHG_globe

        gdr = baseline.sel(Region=self.FocusRegion) - (baseline.sel(Region='EARTH') - global_traject) * rci_reg
        gdr = gdr.rename('Value')
        
        # GDR Post 2030
        gdr_post2030 = ((1-yearfracs) * (baseline.sel(Region=self.FocusRegion) - (baseline.sel(Region='EARTH', Time=self.analysis_timeframe) \
                - global_traject.sel(Time=self.analysis_timeframe)) * rci_reg.sel(Time=2030))  \
                    + yearfracs * self.xr_total.AP.sel(Time=self.analysis_timeframe)).sel(Time=np.arange(2031, 2101))
        
        gdr_total = xr.merge([gdr, gdr_post2030])
        gdr_total = gdr_total.rename({'Value': 'GDR'})
        self.xr_total = self.xr_total.assign(GDR = gdr_total.GDR)
        xr_rci.close()

    # =========================================================== #
    # =========================================================== #

    def save(self):
        savename = self.version_path + 'xr_alloc_'+self.FocusRegion+'.nc'

        xr_total_onlyalloc = (self.xr_total.drop_vars([
                'Population', 
                "CO2_hist", 
                "CO2_globe", 
                "N2O_hist", 
                "CH4_hist", 
                'GDP', 
                'GHG_hist', 
                'GHG_globe', 
                "NonCO2_globe", 
                'GHG_base', 
                'GHG_ndc', 
                'Conditionality', 
                'Ambition', 
                'Budget',
                'GHG_hist_excl',
            ])
            .sel(
                Region=self.FocusRegion, 
                Time=np.arange(self.settings['params']['start_year_analysis'], 2101)
            )
            .astype("float32")
        )
        xr_total_onlyalloc.to_netcdf(self.settings['paths']['data']['datadrive']+'Allocations/'+savename,         
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
