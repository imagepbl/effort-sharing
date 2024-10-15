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

# =========================================================== #
# CLASS OBJECT
# =========================================================== #

class allocation():
    '''
    Class that allocates the global CO2 budget to regions based on different allocation methods
    '''

    # =========================================================== #
    # =========================================================== #

    def __init__(self, reg, lulucf='incl', dataread_file="xr_dataread.nc", gas = 'GHG'):
        self.current_dir = Path.cwd()

        # Read in Input YAML file
        with open(self.current_dir / 'input.yml') as file:
            self.settings = yaml.load(file, Loader=yaml.FullLoader)
        self.countries_iso = np.load(self.settings['paths']['data']['datadrive'] + "all_countries.npy", allow_pickle=True)
        self.savepath = self.settings['paths']['data']['datadrive'] + "startyear_" + str(self.settings['params']['start_year_analysis']) + "/"
        self.xr_total = xr.open_dataset(self.savepath + dataread_file).load()
        self.dataread_file = dataread_file

        # Region and Time variables
        self.focus_region = reg
        self.start_year_analysis = self.settings['params']['start_year_analysis']
        self.analysis_timeframe = np.arange(self.start_year_analysis, 2101)

        # Historical emissions
        if lulucf == 'incl' and gas == 'CO2':
            self.emis_hist = self.xr_total.CO2_hist
            self.emis_fut = self.xr_total.CO2_globe
            self.emis_base = self.xr_total.CO2_base_incl
        elif lulucf == 'incl' and gas == 'GHG':
            self.emis_hist = self.xr_total.GHG_hist
            self.emis_fut = self.xr_total.GHG_globe
            self.emis_base = self.xr_total.GHG_base_incl
        elif lulucf == 'excl' and gas == 'CO2':
            self.emis_hist = self.xr_total.CO2_hist_excl
            self.emis_fut = self.xr_total.CO2_globe_excl
            self.emis_base = self.xr_total.CO2_base_excl
        elif lulucf == 'excl' and gas == 'GHG':
            self.emis_hist = self.xr_total.GHG_hist_excl
            self.emis_fut = self.xr_total.GHG_globe_excl
            self.emis_base = self.xr_total.GHG_base_excl
        self.rbw = xr.open_dataset(self.savepath + "xr_rbw_"+gas+"_"+lulucf+".nc").load()
        self.lulucf_indicator = lulucf
        self.gas_indicator = "_"+gas

    # =========================================================== #
    # =========================================================== #

    def gf(self):
        '''
        Grandfathering: Divide the global budget over the regions based on
        their historical CO2 emissions
        '''

        # Calculating the current CO2 fraction for region and world based on start_year_analysis
        current_co2_region = self.emis_hist.sel(Region=self.focus_region,
                                                    Time=self.start_year_analysis)

        current_co2_earth = (1e-9+self.emis_hist.sel(Region='EARTH',
                                                        Time=self.start_year_analysis))

        co2_fraction = current_co2_region / current_co2_earth

        # New CO2 time series from the start_year to 2101 by multiplying global budget with fraction
        xr_new_co2 = (co2_fraction*self.emis_fut).sel(Time=self.analysis_timeframe)

        # Adds the new CO2 time series to the xr_total dataset
        self.xr_total = self.xr_total.assign(GF = xr_new_co2)

    # =========================================================== #
    # =========================================================== #

    def pc(self):
        '''
        Per Capita: Divide the global budget equally per capita
        '''

        pop_region = self.xr_total.sel(Region=self.focus_region,
                                       Time=self.start_year_analysis).Population
        pop_earth = self.xr_total.sel(Region=self.countries_iso,
                                      Time=self.start_year_analysis).Population.sum(dim=['Region'])
        pop_fraction =  pop_region / pop_earth

        # Multiplying the global budget with the population fraction to create
        # new allocation time series from start_year to 2101
        xr_new = (pop_fraction * self.emis_fut).sel(Time=self.analysis_timeframe)
        self.xr_total = self.xr_total.assign(PC = xr_new)

    # =========================================================== #
    # =========================================================== #

    def pcc(self):
        '''
        Per Capita Convergence: Grandfathering converging into per capita
        '''

        def transform_time(time, convyear):
            '''
            Function that calculates the convergence based on the convergence time frame
            '''
            fractions = pd.DataFrame({'Year': time, 'Convergence': 0}, dtype=float)

            before_analysis_year = fractions['Year'] < self.start_year_analysis + 1
            fractions.loc[before_analysis_year, 'Convergence'] = 1.0

            start_conv = self.start_year_analysis + 1

            during_conv = (fractions['Year'] >= start_conv) & (fractions['Year'] < convyear)
            year_diff = fractions['Year'] - self.start_year_analysis
            conv_range = convyear - self.start_year_analysis
            fractions.loc[during_conv, 'Convergence'] = 1.0 - (year_diff / conv_range)

            return fractions['Convergence'].tolist()

        gfdeel = []
        pcdeel = []

        convergence_years = np.arange(self.settings['params']['pcc_conv_start'],
                                      self.settings['params']['pcc_conv_end'] + 1, 5)

        for year in convergence_years:
            ar = np.array([transform_time(self.xr_total.Time, year)]).T
            coords = {'Time': self.xr_total.Time, 'Convergence_year': [year]}
            dims = ['Time', 'Convergence_year']
            gfdeel.append(xr.DataArray(data=ar, dims=dims, coords=coords).to_dataset(name="PCC"))
            pcdeel.append(xr.DataArray(data=1-ar, dims=dims, coords=coords).to_dataset(name="PCC"))

        # Merging the list of DataArays into one Dataset
        gfdeel_single = xr.merge(gfdeel)
        pcdeel_single = xr.merge(pcdeel)

        # Creating new allocation time series by multiplying convergence fractions
        # with existing GF and PC allocations
        xr_new = (gfdeel_single * self.xr_total.GF + pcdeel_single * self.xr_total.PC)['PCC']
        self.xr_total = self.xr_total.assign(PCC = xr_new)

    # =========================================================== #
    # =========================================================== #

    def pcb(self):
        '''
        Per capita on a budget basis
        '''
        start_year = self.start_year_analysis
        focus_region = self.focus_region

        # co2 part
        def budget_harm(nz):
            end_year = 2101
            compensation_form = np.sqrt(np.arange(0, end_year - start_year))
            xr_comp2 = xr.DataArray(
                compensation_form,
                dims=['Time'],
                coords={'Time': np.arange(start_year, end_year)}
            )
            return xr_comp2 / ((nz - start_year) ** (3/2) * (2/3))
            # TODO later: should be , but I now calibrated to 0.5.
            # Not a problem because we have the while loop later.

        def pcb_new_factor(path, f):
            positive_path = path.where(path > 0, 0)
            negative_path = positive_path.where(path < 0, 1)

            netzeros = start_year + negative_path.sum(dim='Time')
            netzeros = netzeros.where(netzeros < 2100, 2100)

            return path + budget_harm(netzeros) * f

        pop_region = self.xr_total.sel(Time=start_year).Population
        pop_earth = self.xr_total.sel(Region='EARTH', Time=start_year).Population
        pop_fraction = (pop_region / pop_earth).mean(dim='Scenario')
        globalpath = self.emis_fut

        emis_start_i = self.emis_hist.sel(Time=start_year)
        emis_start_w = self.emis_hist.sel(Time=start_year, Region='EARTH')

        time_range = np.arange(start_year, 2101)
        path_scaled_0 = (
            (emis_start_i / emis_start_w * globalpath)
            .sel(Time=time_range)
            .sel(Region=focus_region)
        )

        budget_left = (
            self.emis_fut
            .where(self.emis_fut > 0, 0)
            .sel(Time=time_range)
            .sum(dim='Time') * pop_fraction
        ).sel(Region=focus_region)

        budget_without_assumptions_prepeak = (
            path_scaled_0.where(path_scaled_0 > 0, 0)
            .sum(dim='Time')
        )

        budget_surplus = (budget_left - budget_without_assumptions_prepeak)
        pcb = pcb_new_factor(path_scaled_0, budget_surplus).to_dataset(name='PCB')

        # Optimize to bend the CO2 curves as close as possible to the CO2 budgets
        iterations = 3

        for _ in range(iterations):
            # Calculate the positive part of the CO2 path
            pcb_pos = pcb.where(pcb > 0, 0).sum(dim='Time')

            # Calculate the budget surplus
            budget_surplus = (budget_left - pcb_pos).PCB

            # Adjust the CO2 path based on the budget surplus
            pcb = pcb_new_factor(pcb.PCB, budget_surplus).to_dataset(name='PCB')

        # CO2, but now linear
        co2_hist = self.emis_hist.sel(Region=focus_region, Time=start_year)
        time_range = np.arange(start_year, 2101)

        nz = (budget_left * 2 / co2_hist + start_year - 1)
        coef = co2_hist / (nz - start_year)

        linear_co2 = (
            -coef * xr.DataArray(
                np.arange(0, 2101 - start_year),
                dims=['Time'],
                coords={'Time': time_range}
            ) + co2_hist
        )

        linear_co2_pos = linear_co2.where(linear_co2 > 0, 0).to_dataset(name='PCB_lin')

        # TODO: Is it correct to just leave non-co2 part out?

        # # Non-co2 part
        # nonco2_current = (
        #     self.xr_total.GHG_hist.sel(Time=start_year) -
        #     self.xr_total.CO2_hist.sel(Time=start_year)
        # )

        # nonco2_fraction = nonco2_current / nonco2_current.sel(Region='EARTH')
        # nonco2_part_gf = nonco2_fraction * self.xr_total.NonCO2_globe

        # pc_fraction = (
        #     self.xr_total.Population.sel(Time=start_year) /
        #     self.xr_total.Population.sel(Time=start_year, Region='EARTH')
        # )
        # nonco2_part_pc = pc_fraction * self.xr_total.NonCO2_globe

        # # Create an array that transitions linearly from 0 to 1 from start_year to 2039,
        # # and then remains constant at 1 from 2040 to 2100.
        # compensation_form = np.concatenate([
        #     np.linspace(0, 1, len(np.arange(start_year, 2040))),
        #     np.ones(len(np.arange(2040, 2101)))
        # ])

        # xr_comp = xr.DataArray(
        #     compensation_form,
        #     dims=['Time'],
        #     coords={'Time': time_range}
        # )

        # nonco2_part = nonco2_part_gf * (1 - xr_comp) + nonco2_part_pc * xr_comp

        # # together:
        # nonco2_focus_region = nonco2_part.sel(Region=focus_region)
        # self.ghg_pcb = pcb + nonco2_focus_region
        # self.ghg_pcb_lin = linear_co2_pos + nonco2_focus_region

        # together:
        self.ghg_pcb = pcb
        self.ghg_pcb_lin = linear_co2_pos

        self.xr_total = self.xr_total.assign(
            PCB = self.ghg_pcb.PCB,
            PCB_lin = self.ghg_pcb_lin.PCB_lin
        )

    # =========================================================== #
    # =========================================================== #

    def ecpc(self):
        '''
        Equal Cumulative per Capita: Uses historical emissions, discount factors and
        population shares to allocate the global budget
        '''
        # make sqrt curve
        compensation_form_sqrt = np.sqrt(np.arange(0, 2101 - self.start_year_analysis))
        # sum of values has to be 1
        compensation_form_sqrt = compensation_form_sqrt / np.sum(compensation_form_sqrt)

        xr_comp = xr.DataArray(compensation_form_sqrt, dims=['Time'], coords={'Time': self.analysis_timeframe})

        # Defining the timeframes for historical and future emissions
        xrs = []
        hist_emissions_startyears = self.settings['params']['hist_emissions_startyears']
        discount_rates = self.settings['params']['discount_rates']

        for startyear in hist_emissions_startyears:
            hist_emissions_timeframe = np.arange(startyear, 1 + self.start_year_analysis)
            future_emissions_timeframe = np.arange(self.start_year_analysis + 1, 2101)

            # Summing all historical CO2 emissions over the hist_emissions_timeframe
            hist_emissions = self.emis_hist.sel(Time = hist_emissions_timeframe)

            # Discounting -> We only do past discounting here
            for discount in discount_rates:
                past_timeline = np.arange(startyear, self.start_year_analysis + 1)

                discount_factor = (1 - discount / 100)
                discount_period = self.start_year_analysis - past_timeline
                xr_discount = xr.DataArray(discount_factor ** discount_period, dims=['Time'],
                                        coords={'Time': past_timeline})
                hist_emissions_discounted = (hist_emissions * xr_discount).sum(dim='Time')
                hist_emissions_w = float(hist_emissions_discounted.sel(Region='EARTH'))
                hist_emissions_r = float(hist_emissions_discounted.sel(Region = self.focus_region))

                # Summing all future emissions over the future_emissions_timeframe
                future_emissions_w = self.emis_fut.sel(Time = future_emissions_timeframe).sum(dim='Time')

                total_emissions_w = hist_emissions_w + future_emissions_w

                # Calculating the cumulative population shares for region and world
                cum_pop = self.xr_total.Population.sel(Time = self.analysis_timeframe).sum(dim='Time')
                cum_pop_r = cum_pop.sel(Region=self.focus_region)
                cum_pop_w = cum_pop.sel(Region='EARTH')
                share_cum_pop = cum_pop_r / cum_pop_w
                budget_rightful = total_emissions_w * share_cum_pop
                budget_left = budget_rightful - hist_emissions_r

                # Now temporal allocation
                #globalbudget = self.xr_total.CO2_globe.sel(Time=self.analysis_timeframe).sum(dim='Time')
                globalpath = self.emis_fut

                emis_start_r = self.emis_hist.sel(Time=self.start_year_analysis,
                                                            Region=self.focus_region)
                emis_start_w = self.emis_hist.sel(Time=self.start_year_analysis,
                                                            Region='EARTH')
                emis_ratio = emis_start_r / emis_start_w
                path_scaled_0 = emis_ratio * globalpath
                budget_without_assumptions = path_scaled_0.sum(dim='Time')
                budget_surplus = budget_left - budget_without_assumptions

                def ecpc_factor(initial_path, f):
                    '''
                    Calculates a modified emissions path. Takes scaling factor f as input and
                    returns a new emissions path by scaling the compensation form (xr_comp) with f
                    and adding it to the initial emissions path (path_scaled_0).
                    '''
                    return initial_path + xr_comp * f

                ecpc = ecpc_factor(path_scaled_0, budget_surplus)
                ecpc_expanded = ecpc.expand_dims(Discount_factor=[discount],
                                                 Historical_startyear=[startyear]).to_dataset(name='ECPC')
                xrs.append(ecpc_expanded)

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
        xrt = self.xr_total.sel(Time=self.analysis_timeframe)
        GDP_sum_w = xrt.GDP.sel(Region='EARTH')
        pop_sum_w = xrt.Population.sel(Region='EARTH')
        # Global average GDP per capita
        r1_nom = GDP_sum_w / pop_sum_w

        base_worldsum = self.emis_base.sel(Time=self.analysis_timeframe).sel(Region='EARTH')
        rb_part1 = (xrt.GDP.sel(Region=self.focus_region) / xrt.Population.sel(Region=self.focus_region) / r1_nom)**(1/3.)
        rb_part2 = self.emis_base.sel(Time=self.analysis_timeframe).sel(Region=self.focus_region) * (base_worldsum - self.emis_fut.sel(Time=self.analysis_timeframe)) / base_worldsum
        rb = rb_part1 * rb_part2

        # Step 2: Correction factor
        corr_factor = (1e-9+self.rbw.__xarray_dataarray_variable__)/(base_worldsum - self.emis_fut.sel(Time=self.analysis_timeframe))

        # Step 3: Budget after correction factor
        ap = self.emis_base.sel(Region=self.focus_region) - rb/corr_factor

        ap = ap.sel(Time=self.analysis_timeframe)
        self.xr_total = self.xr_total.assign(AP = ap)
        self.rbw.close()

    # =========================================================== #
    # =========================================================== #

    def gdr(self):
        '''
        Greenhouse Development Rights: Uses the Responsibility-Capability Index
        (RCI) weighed at 50/50 to allocate the global budget
        Calculations from van den Berg et al. (2020)
        '''
        xr_rci = xr.open_dataset(self.settings['paths']['data']['datadrive'] + "xr_rci.nc").load()
        yearfracs = xr.Dataset(data_vars={"Value": (['Time'],
                                                    (self.analysis_timeframe - 2030) \
                                                        / (self.settings['params']['convergence_year_gdr'] - 2030))},
                                coords={"Time": self.analysis_timeframe})

        # Get the regional RCI values
        # If region is EU, we have to sum over the EU countries
        if self.focus_region != 'EU':
            rci_reg = xr_rci.rci.sel(Region=self.focus_region)
        else:
            df = pd.read_excel("X:/user/dekkerm/Data/UNFCCC_Parties_Groups_noeu.xlsx", sheet_name = "Country groups")
            countries_iso = np.array(df["Country ISO Code"])
            group_eu = countries_iso[np.array(df["EU"]) == 1]
            rci_reg = xr_rci.rci.sel(Region=group_eu).sum(dim='Region')

        # Compute GDR until 2030
        baseline = self.emis_base
        global_traject = self.emis_fut

        gdr = baseline.sel(Region=self.focus_region) - (baseline.sel(Region='EARTH') - global_traject) * rci_reg
        gdr = gdr.rename('Value')

        # GDR Post 2030
        # Calculate the baseline difference
        baseline_earth = baseline.sel(Region='EARTH', Time=self.analysis_timeframe)
        global_traject_time = global_traject.sel(Time=self.analysis_timeframe)
        baseline_diff = baseline_earth - global_traject_time

        rci_2030 = baseline_diff * rci_reg.sel(Time=2030)
        part1 = (1 - yearfracs) * (baseline.sel(Region=self.focus_region) - rci_2030)
        part2 = yearfracs * self.xr_total.AP.sel(Time=self.analysis_timeframe)
        gdr_post2030 = (part1 + part2).sel(Time=np.arange(2031, 2101))

        gdr_total = xr.merge([gdr, gdr_post2030])
        gdr_total = gdr_total.rename({'Value': 'GDR'})
        self.xr_total = self.xr_total.assign(GDR = gdr_total.GDR)
        xr_rci.close()

    # =========================================================== #
    # =========================================================== #

    def save(self):
        '''
        Extract variables from xr_total dataset and save allocation data to a NetCDF file
        '''
        foldername = 'Allocations'+self.gas_indicator+'_'+self.lulucf_indicator
        savename = 'xr_alloc_'+self.focus_region+'.nc'
        if self.dataread_file != 'xr_dataread.nc':
            savename = 'xr_alloc_'+self.focus_region+'_adapt.nc'
        savepath = self.savepath+foldername+'/'+savename

        xr_total_onlyalloc = (self.xr_total[['GF', 'PC', 'PCC', 'ECPC', 'AP', 'GDR', 'PCB', 'PCB_lin']]
            .sel(
                Time=np.arange(self.settings['params']['start_year_analysis'], 2101)
            )
            .astype("float32")
        )
        xr_total_onlyalloc.to_netcdf(savepath,
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
    allocator.pcb()
    allocator.ecpc()
    allocator.ap()
    allocator.gdr()
    allocator.save()
