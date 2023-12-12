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
        #print("# ==================================== #")
        #print("# Initializing allocation class        #")
        #print("# ==================================== #")

        self.current_dir = Path.cwd()

        # Read in Input YAML file
        with open(self.current_dir / 'input.yml') as file:
            self.settings = yaml.load(file, Loader=yaml.FullLoader)
        self.countries_iso = np.load(self.settings['paths']['data']['datadrive'] + "all_countries.npy", allow_pickle=True)
        self.xr_total = xr.open_dataset(self.settings['paths']['data']['datadrive'] + "xr_dataread.nc").sel()
        self.FocusRegion = reg
        self.xr_rbw = xr.open_dataset(self.settings['paths']['data']['datadrive'] + "xr_rbw.nc")

    # =========================================================== #
    # =========================================================== #

    def gf(self):
        #print('- Allocate by grandfathering')
        emis_fraction = self.xr_total.sel(Region=self.FocusRegion, Time=self.settings['params']['start_year_analysis']).GHG_hist / (1e-9+self.xr_total.sel(Region='EARTH', Time=self.settings['params']['start_year_analysis']).GHG_hist)
        xr_new = (emis_fraction*self.xr_total.GHG_globe).sel(Time=np.arange(self.settings['params']['start_year_analysis'], 2101))
        self.xr_total = self.xr_total.assign(GF = xr_new)

    # =========================================================== #
    # =========================================================== #

    def pc(self):
        #print('- Allocate by per capita')
        pop_fraction = self.xr_total.sel(Region=self.FocusRegion, Time=self.settings['params']['start_year_analysis']).Population / (self.xr_total.sel(Region=self.countries_iso, Time=self.settings['params']['start_year_analysis']).Population.sum(dim=['Region']))
        xr_new = (pop_fraction*self.xr_total.GHG_globe).sel(Time=np.arange(self.settings['params']['start_year_analysis'], 2101))
        self.xr_total = self.xr_total.assign(PC = xr_new)

    # =========================================================== #
    # =========================================================== #

    def pcc(self):
        #print('- Allocate by per capita convergence')
        def transform_time(time, convyear):
            fractions = []
            for t in time:
                if t < self.settings['params']['start_year_analysis']+1:
                    fractions.append(1)
                elif t < convyear:
                    fractions.append(1-(t-self.settings['params']['start_year_analysis']) / (convyear-self.settings['params']['start_year_analysis']))
                else:
                    fractions.append(0)
            return fractions
        gfdeel = []
        pcdeel = []
        for year_i, year in enumerate(np.arange(self.settings['params']['pcc_conv_start'], self.settings['params']['pcc_conv_end']+1, 5)):
            ar = np.array([transform_time(self.xr_total.Time, year)]).T
            gfdeel.append(xr.DataArray(data=ar, dims=['Time', 'Convergence_year'], coords=dict(Time=self.xr_total.Time, Convergence_year=[year])).to_dataset(name="PCC"))
            pcdeel.append(xr.DataArray(data=1-ar, dims=['Time', 'Convergence_year'], coords=dict(Time=self.xr_total.Time, Convergence_year=[year])).to_dataset(name="PCC"))
        gfdeel_single = xr.merge(gfdeel)
        pcdeel_single = xr.merge(pcdeel)
        self.xr_total = self.xr_total.assign(PCC = (gfdeel_single*self.xr_total.GF + pcdeel_single*self.xr_total.PC)['PCC'])

    # =========================================================== #
    # =========================================================== #

    def ecpc(self):
        #print('- Allocate by equal cumulative per capita')
        hist_emissions = self.xr_total.GHG_hist.sel( Time=np.arange(self.settings['params']['historical_emissions_startyear'], 1+self.settings['params']['start_year_analysis'])).sum(dim='Time')
        hist_emissions_w = float(hist_emissions.sel(Region='EARTH'))
        future_emissions_w = self.xr_total.GHG_globe.sel(Time=np.arange(self.settings['params']['start_year_analysis']+1, 2101)).sum(dim='Time')
        total_emissions_w = hist_emissions_w+future_emissions_w
        hist_emissions_r = float(hist_emissions.sel(Region=self.FocusRegion))
        cum_pop = self.xr_total.Population.sel(Time=np.arange(self.settings['params']['historical_emissions_startyear'], 2101)).sum(dim='Time')
        share_cum_pop = cum_pop.sel(Region=self.FocusRegion) / cum_pop.sel(Region=self.countries_iso).sum(dim='Region')
        budget_rightful = total_emissions_w*share_cum_pop
        budget_left = budget_rightful - hist_emissions_r

        # Now temporal allocation
        #globalbudget = self.xr_total.GHG_globe.sel(Time=np.arange(self.settings['params']['start_year_analysis'], 2101)).sum(dim='Time')
        globalpath = self.xr_total.GHG_globe

        emis_2021_i = self.xr_total.GHG_hist.sel(Time=self.settings['params']['start_year_analysis'], Region=self.FocusRegion)
        emis_2021_w = self.xr_total.GHG_hist.sel(Time=self.settings['params']['start_year_analysis'], Region='EARTH')
        path_scaled_0 = emis_2021_i/emis_2021_w*globalpath
        budget_without_assumptions = path_scaled_0.sum(dim='Time')
        budget_surplus = budget_left - budget_without_assumptions

        compensation_form = np.array(list(np.linspace(0, 1, len(np.arange(self.settings['params']['start_year_analysis'], 2040))))+[1]*len(np.arange(2040, 2101)))
        compensation_form2 = np.convolve(compensation_form, np.ones(3)/3, mode='valid')
        compensation_form[1:-1] = compensation_form2
        compensation_form = compensation_form - compensation_form[0]
        compensation_form = compensation_form / np.sum(compensation_form)
        xr_comp = xr.DataArray(compensation_form, dims=['Time'], coords={'Time': np.arange(self.settings['params']['start_year_analysis'], 2101)})

        def ecpc_factor(f):
            return path_scaled_0+xr_comp*f

        ecpc = ecpc_factor(budget_surplus)
        self.xr_total = self.xr_total.assign(ECPC = ecpc)

    # =========================================================== #
    # =========================================================== #

    def ap(self):
        #print('- Allocate by ability to pay')
        xrt = self.xr_total.sel(Time=np.arange(self.settings['params']['start_year_analysis'], 2101))
        r1_nom = (xrt.GDP.sel(Region=self.countries_iso).sum(dim='Region') / xrt.Population.sel(Region=self.countries_iso).sum(dim='Region'))
        base_worldsum = xrt.GHG_base.sel(Region=self.countries_iso).sum(dim='Region')
        rb_part1 = (xrt.GDP.sel(Region=self.FocusRegion) / xrt.Population.sel(Region=self.FocusRegion) / r1_nom)**(1/3.)
        rb_part2 = xrt.GHG_base.sel(Region=self.FocusRegion)*(base_worldsum - xrt.GHG_globe)/base_worldsum
        rb = rb_part1*rb_part2
        ap = self.xr_total.GHG_base.sel(Region=self.FocusRegion) - rb/self.xr_rbw.__xarray_dataarray_variable__*(base_worldsum - self.xr_total.GHG_globe)
        ap = ap.sel(Time=np.arange(self.settings['params']['start_year_analysis'], 2101))
        self.xr_total = self.xr_total.assign(AP = ap)

    # =========================================================== #
    # =========================================================== #

    def gdr(self):
        #print('- Allocate by greenhouse development rights')
        # Read RCI
        df_rci = pd.read_csv(self.settings['paths']['data']['external'] + "RCI/RCI.xls", delimiter='\t', skiprows=30)[:-2]
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
        gdr = self.xr_total.GHG_base.sel(Region=self.FocusRegion) - (self.xr_total.GHG_base.sel(Region=self.countries_iso).sum(dim='Region') - self.xr_total.GHG_globe)*rci_reg
        yearfracs = xr.Dataset(data_vars={"Value": (['Time'], (np.arange(self.settings['params']['start_year_analysis'], 2101) - 2030) / (self.settings['params']['convergence_year_gdr'] - 2030))}, coords={"Time": np.arange(self.settings['params']['start_year_analysis'], 2101)})
        gdr = gdr.rename('Value')
        gdr_post2030 = ((self.xr_total.GHG_base.sel(Region=self.FocusRegion) - (self.xr_total.GHG_base.sel(Region=self.countries_iso, Time=np.arange(self.settings['params']['start_year_analysis'], 2101)).sum(dim='Region') - self.xr_total.GHG_globe.sel(Time=np.arange(self.settings['params']['start_year_analysis'], 2101)))*rci_reg.sel(Time=2030))*(1-yearfracs) + yearfracs*self.xr_total.AP.sel(Time=np.arange(self.settings['params']['start_year_analysis'], 2101))).sel(Time=np.arange(2031, 2101))
        gdr_total = xr.merge([gdr, gdr_post2030])
        gdr_total = gdr_total.rename({'Value': 'GDR'})
        self.xr_total = self.xr_total.assign(GDR = gdr_total.GDR)

    # =========================================================== #
    # =========================================================== #

    def save(self):
        #print('- Save')
        xr_total_onlyalloc = self.xr_total.drop_vars(['Population', "CO2_hist", "CO2_globe", "N2O_hist", "CH4_hist", 'GDP', 'GHG_hist', 'GHG_globe', "CH4_globe", "N2O_globe", "GHG_hist_all", 'GHG_base', 'GHG_ndc', 'Hot_air', 'Conditionality', 'Ambition', 'Region', 'Budget']).sel(Time=np.arange(self.settings['params']['start_year_analysis'], 2101)).astype("float32")
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