# ======================================== #
# Class that does the data reading
# ======================================== #

# =========================================================== #
# PREAMBULE
# Put in packages that we need
# =========================================================== #

from pathlib import Path
import yaml
import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit
import pandas as pd
import xarray as xr
import json

# =========================================================== #
# CLASS OBJECT
# =========================================================== #

class datareading(object):

    # =========================================================== #
    # =========================================================== #

    def __init__(self):
        print("# ==================================== #")
        print("# Initializing datareading class     #")
        print("# ==================================== #")

        self.current_dir = Path.cwd()

        # Read in Input YAML file
        with open(self.current_dir / 'input.yml') as file:
            self.settings = yaml.load(file, Loader=yaml.FullLoader)
        
        # Lists of variable settings
        self.Tlist = np.array([1.5, 1.56, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4]).astype(float).round(2)           # At least between 1.3 and 2.4
        self.Plist = np.array([.17, 0.33, 0.50, 0.67, 0.83]).round(2)               # At least between 0.17 and 0.83
        #self.NClist = np.arange(0.05, 0.95+1e-9, 0.15).astype(float).round(2)
        self.Neglist = np.array([0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]).round(2) 
        self.NonCO2list = np.array([0.1, 0.33, 0.5, 0.67, 0.9]).round(2)#np.arange(0.2, 0.71, 0.1).round(2) # These are reductions in 2040 tov 2020
        #self.PathUnclist = np.array([0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80])
        self.Timinglist = ['Immediate', 'Delayed']

    # =========================================================== #
    # =========================================================== #

    def read_general(self):
        print('- Reading general data')
        df_gen = pd.read_excel(self.settings['paths']['data']['external']+"UNFCCC_Parties_Groups_noeu.xlsx", sheet_name = "Country groups")
        self.countries_iso = np.array(list(df_gen["Country ISO Code"]))
        self.countries_name = np.array(list(df_gen["Name"]))
        self.regions_iso = np.array(list(df_gen["Country ISO Code"]) + ['EU', 'EARTH'])
        self.regions_name = np.array(list(df_gen["Name"])+['Earth'])

    # =========================================================== #
    # =========================================================== #

    def read_ssps(self):
        print('- Reading GDP and population data from SSPs')
        df_ssp = pd.read_csv(self.settings['paths']['data']['external']+"SSPs/SspDb_country_data_2013-06-12.csv")
        df_ssp = df_ssp[(df_ssp.MODEL == 'OECD Env-Growth') & (df_ssp.SCENARIO.isin(['SSP1_v9_130325', 'SSP2_v9_130325', 'SSP3_v9_130325', 'SSP4_v9_130325', 'SSP5_v9_130325']))]
        Scenario = np.array(df_ssp['SCENARIO'])
        Scenario[Scenario == 'SSP1_v9_130325'] = 'SSP1'
        Scenario[Scenario == 'SSP2_v9_130325'] = 'SSP2'
        Scenario[Scenario == 'SSP3_v9_130325'] = 'SSP3'
        Scenario[Scenario == 'SSP4_v9_130325'] = 'SSP4'
        Scenario[Scenario == 'SSP5_v9_130325'] = 'SSP5'
        df_ssp['SCENARIO'] = Scenario
        Variable = np.array(df_ssp['VARIABLE'])
        Variable[Variable == 'GDP|PPP'] = 'GDP'
        df_ssp['VARIABLE'] = Variable
        df_ssp = df_ssp.drop(['MODEL', 'UNIT'], axis=1)
        df_ssp = df_ssp.rename(columns={'SCENARIO': "Scenario", 'REGION': 'Region', 'VARIABLE': 'Variable'})
        dummy = df_ssp.melt(id_vars=["Scenario", "Region", "Variable"], var_name="Time", value_name="Value")
        dummy['Time'] = np.array(dummy['Time'].astype(int))
        self.xr_ssp_old = xr.Dataset.from_dataframe(dummy.pivot(index=['Scenario', 'Region', 'Time'], columns='Variable', values='Value'))

        df_ssp = pd.read_excel(self.settings['paths']['data']['external']+"SSPs/SSPs_v2023.xlsx", sheet_name='data')
        df_ssp = df_ssp[(df_ssp.Model.isin(['OECD ENV-Growth 2023', 'IIASA-WiC POP 2023'])) & (df_ssp.Scenario.isin(['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']))]
        region_full = np.array(df_ssp.Region)
        region_iso = []
        for r_i, r in enumerate(region_full):
            wh = np.where(self.regions_name == r)[0]
            if len(wh) > 0:
                iso = self.countries_iso[wh[0]]
            elif r == 'Aruba':
                iso = 'ABW'
            elif r == 'Bahamas':
                iso = 'BHS'
            elif r == 'Democratic Republic of the Congo':
                iso = 'COD'
            elif r == 'Cabo Verde':
                iso = 'CPV'
            elif r == "C?te d'Ivoire":
                iso = 'CIV'
            elif r == 'Western Sahara':
                iso = 'ESH'
            elif r == 'Gambia':
                iso = 'GMB'
            elif r == 'Czechia':
                iso = 'CZE'
            elif r == 'French Guiana':
                iso = 'GUF'
            elif r == 'Guam':
                iso = 'GUM'
            elif r == 'Hong Kong':
                iso = 'HKG'
            elif r == 'Iran':
                iso = 'IRN'
            elif r == 'Macao':
                iso = 'MAC'
            elif r == 'Moldova':
                iso = 'MDA'
            elif r == 'Mayotte':
                iso = 'MYT'
            elif r == 'New Caledonia':
                iso = 'NCL'
            elif r == 'Puerto Rico':
                iso = 'PRI'
            elif r == 'French Polynesia':
                iso = 'PYF'
            elif r == 'Turkey':
                iso = 'TUR'
            elif r == 'Taiwan':
                iso = 'TWN'
            elif r == 'Tanzania':
                iso = 'TZA'
            elif r == 'United States':
                iso = 'USA'
            elif r == 'United States Virgin Islands':
                iso = 'VIR'
            elif r == 'Viet Nam':
                iso = 'VNM'
            elif r == 'Cura?ao':
                iso = 'CUW'
            elif r == 'Guadeloupe':
                iso = 'GLP'
            elif r == 'Martinique':
                iso = 'MTQ'
            elif r == 'Palestine':
                iso = 'PSE'
            elif r == 'R?union':
                iso = 'REU'
            elif r == 'Syria':
                iso = 'SYR'
            elif r == 'Venezuela':
                iso = 'VEN'
            elif r == 'World':
                iso = 'EARTH'
            else:
                print(r)
                iso = 'oeps'
            region_iso.append(iso)
        df_ssp['Region'] = region_iso
        Variable = np.array(df_ssp['Variable'])
        Variable[Variable == 'GDP|PPP'] = 'GDP'
        df_ssp['Variable'] = Variable
        df_ssp = df_ssp.drop(['Model', 'Unit'], axis=1)
        dummy = df_ssp.melt(id_vars=["Scenario", "Region", "Variable"], var_name="Time", value_name="Value")
        dummy['Time'] = np.array(dummy['Time'].astype(int))
        self.xr_ssp = xr.Dataset.from_dataframe(dummy.pivot(index=['Scenario', 'Region', 'Time'], columns='Variable', values='Value'))

    # =========================================================== #
    # =========================================================== #

    def read_undata(self):
        print('- Reading UN population data (for past population)')
        df_unp = pd.read_excel(self.settings['paths']['data']['external']+'/UN Population/WPP2022_GEN_F01_DEMOGRAPHIC_INDICATORS.xlsx',
                               sheet_name="Estimates", header=16)
        df_unp = df_unp[["Region, subregion, country or area *", "ISO3 Alpha-code", "Total Population, as of 1 January (thousands)", "Year"]]
        df_unp = df_unp.rename(columns={"Region, subregion, country or area *": "Region",
                                        "ISO3 Alpha-code": "ISO",
                                        "Total Population, as of 1 January (thousands)": "Population",
                                        "Year": "Time"})
        vals = np.array(df_unp.Population).astype(str)
        vals[vals == '...'] = 'nan'
        vals = vals.astype(float)
        vals = vals/1e3
        df_unp['Population'] = vals
        df_unp = df_unp.drop(['Region'], axis=1)
        df_unp = df_unp[df_unp.Time < 2000]
        df_unp = df_unp[df_unp.ISO.isin(self.regions_iso)]
        dummy = df_unp.rename(columns={'ISO': "Region"})
        dummy = dummy.set_index(['Region', 'Time'])
        self.xr_unp = xr.Dataset.from_dataframe(dummy)

    # =========================================================== #
    # =========================================================== #

    def read_historicalemis(self):
        print('- Reading historical emissions (primap)')
        xr_primap = xr.open_dataset("X:/user/dekkerm/Data/PRIMAP/Guetschow_et_al_2023b-PRIMAP-hist_v2.5_final_no_rounding_15-Oct-2023.nc")
        xr_primap = xr_primap.rename({"area (ISO3)": "Region", 'time': 'Time', "category (IPCC2006_PRIMAP)": 'Category', "scenario (PRIMAP-hist)": "Version"}).sel(source='PRIMAP-hist_v2.5_final_nr', Version='HISTTP', Category=["M.0.EL", "M.LULUCF"])[['KYOTOGHG (AR4GWP100)', 'CO2', 'N2O', 'CH4']].sum(dim='Category')
        xr_primap = xr_primap.rename({'KYOTOGHG (AR4GWP100)': "GHG_hist", "CO2": "CO2_hist", "CH4": "CH4_hist", "N2O": "N2O_hist"})
        xr_primap.coords['Time'] = np.array([str(i)[:4] for i in np.array(xr_primap.Time)]).astype(int)
        xr_primap = xr_primap.drop_vars(['source', 'Version']).sel(provenance='measured').drop_vars(['provenance'])
        xr_primap['CO2_hist'] = xr_primap['CO2_hist']/1e3
        xr_primap['GHG_hist_all'] = xr_primap['GHG_hist']/1e3
        xr_primap['GHG_hist'] = (xr_primap['CO2_hist']+xr_primap['CH4_hist']*self.settings['params']['gwp_ch4']/1e3+xr_primap['N2O_hist']*self.settings['params']['gwp_n2o']/1e3)
        xr_primap['CH4_hist'] = xr_primap['CH4_hist']/1e3
        xr_primap['N2O_hist'] = xr_primap['N2O_hist']/1e3
        xr_primap = xr_primap.sel(Time=np.arange(1750, self.settings['params']['start_year_analysis']+1))

        # Dutch emissions - harmonized with the KEV
        dutch_time = np.array([1990, 1995, 2000, 2005, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021])
        dutch_ghg = np.array([228.9, 238.0, 225.7, 220.9, 219.8, 206, 202, 201.2, 192.9, 199.8, 200.2, 196.5, 191.4, 185.6, 168.9, 172.0])
        dutch_time_interp = np.arange(1990, 2021+1)
        dutch_ghg_interp = np.interp(dutch_time_interp, dutch_time, dutch_ghg)
        fraction_1990 = dutch_ghg[0] / xr_primap['GHG_hist'].sel(Region='NLD', Time=1990)
        pre_1990_raw = np.array(xr_primap['GHG_hist'].sel(Region='NLD', Time=np.arange(1750, 1990)))*float(fraction_1990)
        total_time = np.arange(1750, 2021+1)
        total_ghg_nld = np.array(list(pre_1990_raw) + list(dutch_ghg_interp))
        fractions = np.array(xr_primap.GHG_hist.sel(Region='NLD') / total_ghg_nld)
        for t_i, t in enumerate(total_time):
            xr_primap.GHG_hist_all.loc[dict(Time=t, Region='NLD')] = total_ghg_nld[t_i]
            xr_primap.GHG_hist.loc[dict(Time=t, Region='NLD')] = total_ghg_nld[t_i]
        xr_primap.CO2_hist.loc[dict(Region='NLD')] = xr_primap.CO2_hist.sel(Region='NLD')/fractions
        xr_primap.CH4_hist.loc[dict(Region='NLD')] = xr_primap.CH4_hist.sel(Region='NLD')/fractions
        xr_primap.N2O_hist.loc[dict(Region='NLD')] = xr_primap.N2O_hist.sel(Region='NLD')/fractions

        self.xr_primap = xr_primap

        # TODO Excluding LULUCF is not harmonized with KEV !!
        xr_primap = xr.open_dataset("X:/user/dekkerm/Data/PRIMAP/Guetschow_et_al_2023b-PRIMAP-hist_v2.5_final_no_rounding_15-Oct-2023.nc")
        xr_primap = xr_primap.rename({"area (ISO3)": "Region", 'time': 'Time', "category (IPCC2006_PRIMAP)": 'Category', "scenario (PRIMAP-hist)": "Version"}).sel(source='PRIMAP-hist_v2.5_final_nr', Version='HISTTP', Category=["M.0.EL"])[['KYOTOGHG (AR4GWP100)', 'CO2', 'N2O', 'CH4']].sum(dim='Category')
        xr_primap = xr_primap.rename({'KYOTOGHG (AR4GWP100)': "GHG_hist", "CO2": "CO2_hist", "CH4": "CH4_hist", "N2O": "N2O_hist"})
        xr_primap.coords['Time'] = np.array([str(i)[:4] for i in np.array(xr_primap.Time)]).astype(int)
        xr_primap['CO2_hist'] = xr_primap['CO2_hist']/1e3
        xr_primap['GHG_hist_all'] = xr_primap['GHG_hist']/1e3
        xr_primap['GHG_hist_excl'] = (xr_primap['CO2_hist']+xr_primap['CH4_hist']*self.settings['params']['gwp_ch4']/1e3+xr_primap['N2O_hist']*self.settings['params']['gwp_n2o']/1e3)
        xr_primap['CH4_hist'] = xr_primap['CH4_hist']/1e3
        xr_primap['N2O_hist'] = xr_primap['N2O_hist']/1e3
        xr_primap = xr_primap.drop_vars(['CO2_hist', 'GHG_hist_all', 'CH4_hist', 'N2O_hist', 'GHG_hist'])
        self.xr_primap_excl = xr_primap.sel(Time=np.arange(1750, self.settings['params']['start_year_analysis']+1), provenance='measured')

    # =========================================================== #
    # =========================================================== #

    def read_ar6(self):
        print('- Read AR6 data')
        df_ar6raw = pd.read_csv(self.settings['paths']['data']['external']+"IPCC/AR6_Scenarios_Database_World_v1.1.csv")
        df_ar6 = df_ar6raw[df_ar6raw.Variable.isin(['Emissions|CO2',
                                                    'Emissions|CO2|AFOLU',
                                                    'Emissions|Kyoto Gases',
                                                    'Emissions|CO2|Energy and Industrial Processes',
                                                    'Emissions|CH4',
                                                    'Emissions|N2O',
                                                    'Carbon Sequestration|CCS',
                                                    'Carbon Sequestration|Land Use',
                                                    'Carbon Sequestration|Direct Air Capture',
                                                    'Carbon Sequestration|Enhanced Weathering',
                                                    'Carbon Sequestration|Other',
                                                    'Carbon Sequestration|Feedstocks',
                                                    'AR6 climate diagnostics|Exceedance Probability 1.5C|MAGICCv7.5.3',
                                                    'AR6 climate diagnostics|Exceedance Probability 2.0C|MAGICCv7.5.3',
                                                    'AR6 climate diagnostics|Exceedance Probability 2.5C|MAGICCv7.5.3',
                                                    'AR6 climate diagnostics|Exceedance Probability 3.0C|MAGICCv7.5.3',
                                                    'AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|5.0th Percentile',
                                                    'AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|33.0th Percentile',
                                                    'AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile',
                                                    'AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|67.0th Percentile',
                                                    'AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|95.0th Percentile'])]
        df_ar6 = df_ar6.reset_index(drop=True)
        idx = (df_ar6[(df_ar6.Variable == 'Emissions|CH4') & (df_ar6['2100'] > 1e5)]).index # Removing erroneous CH4 scenarios
        df_ar6 = df_ar6[~df_ar6.index.isin(idx)]
        df_ar6 = df_ar6.reset_index(drop=True)
        df_ar6_meta = pd.read_excel(self.settings['paths']['data']['external']+"IPCC/AR6_Scenarios_Database_metadata_indicators_v1.1.xlsx",
                                    sheet_name='meta_Ch3vetted_withclimate')
        mods = np.array(df_ar6_meta.Model)
        scens = np.array(df_ar6_meta.Scenario)
        modscens_meta = np.array([mods[i]+'|'+scens[i] for i in range(len(scens))])
        df_ar6_meta['ModelScenario'] = modscens_meta
        df_ar6_meta = df_ar6_meta[['ModelScenario', 'Category', 'Policy_category']]
        mods = np.array(df_ar6.Model)
        scens = np.array(df_ar6.Scenario)
        modscens = np.array([mods[i]+'|'+scens[i] for i in range(len(scens))])
        df_ar6['ModelScenario'] = modscens
        df_ar6 = df_ar6.drop(['Model', 'Scenario', 'Region', 'Unit'], axis=1)
        dummy = df_ar6.melt(id_vars=["ModelScenario", "Variable"], var_name="Time", value_name="Value")
        dummy['Time'] = np.array(dummy['Time'].astype(int))
        dummy = dummy.set_index(["ModelScenario", "Variable", "Time"])
        xr_scen2 = xr.Dataset.from_dataframe(dummy)
        xr_scen2 = xr_scen2.reindex(Time = np.arange(2000, 2101, 10))
        xr_scen2 = xr_scen2.reindex(Time = np.arange(2000, 2101))
        self.xr_ar6_prevet = xr_scen2.interpolate_na(dim="Time", method="linear")

        vetting_nans = np.array(self.xr_ar6_prevet.ModelScenario[~np.isnan(self.xr_ar6_prevet.Value.sel(Time=2100, Variable='Emissions|CO2'))])
        vetting_2020 = np.array(self.xr_ar6_prevet.ModelScenario[np.where(np.abs(self.xr_ar6_prevet.sel(Time=2020, Variable='Emissions|CO2').Value - self.xr_primap.sel(Region='EARTH', Time=2020).CO2_hist) < 1e4)[0]])
        vetting_total = np.intersect1d(vetting_nans, vetting_2020)
        self.xr_ar6 = self.xr_ar6_prevet.sel(ModelScenario=vetting_total)
        self.ms_immediate = np.array(df_ar6_meta[df_ar6_meta.Policy_category.isin(['P2', 'P2a', 'P2b', 'P2c'])].ModelScenario)
        self.ms_delayed = np.array(df_ar6_meta[df_ar6_meta.Policy_category.isin(['P3a', 'P3b', 'P3c'])].ModelScenario)

    # =========================================================== #
    # =========================================================== #

    def relation_budget_nonco2(self):
        print('- Get relationship between CO2 budgets and non-co2 reduction in 2050')
        df_nonco2 = pd.read_excel("X:/user/dekkerm/Data/Budgets_Rogelj/NonCO2.xlsx", sheet_name='Sheet1')
        df_nonco2 = df_nonco2[['Temperature', 'Reduction', 'Total']]
        dummy = df_nonco2.rename(columns={'Total': "EffectOnRCB"})
        dummy = dummy.rename(columns={'Reduction': "NonCO2red"})
        dummy = dummy.set_index(['Temperature', 'NonCO2red'])
        xr_nonco2effects = xr.Dataset.from_dataframe(dummy)
        xr_nonco2effects = xr_nonco2effects.reindex(NonCO2red = np.arange(0, 0.8001, 0.01).round(2))
        xr_nonco2effects = xr_nonco2effects.interpolate_na(dim="NonCO2red", method="linear")
        xr_nonco2effects = xr_nonco2effects.reindex(Temperature = list(self.Tlist)+[2.5])
        xr_nonco2effects = xr_nonco2effects.interpolate_na(dim="Temperature", method="linear")
        xr_nonco2effects = xr_nonco2effects.reindex(Temperature = list(self.Tlist))
        self.xr_nonco2effects = xr_nonco2effects

    # =========================================================== #
    # =========================================================== #

    def determine_global_nonco2_trajectories(self):
        print('- Computing global nonco2 trajectories')
        xr_ch4_raw = self.xr_ar6.sel(Variable='Emissions|CH4', Time=np.arange(2020, 2101))*self.settings['params']['gwp_ch4']/1e3
        xr_n2o_raw = self.xr_ar6.sel(Variable='Emissions|N2O', Time=np.arange(2020, 2101))*self.settings['params']['gwp_n2o']/1e6
        n2o_2021 = self.xr_primap.sel(Region='EARTH').sel(Time=2021).N2O_hist*self.settings['params']['gwp_n2o']/1e3
        ch4_2021 = self.xr_primap.sel(Region='EARTH').sel(Time=2021).CH4_hist*self.settings['params']['gwp_ch4']/1e3
        n2o_2020 = self.xr_primap.sel(Region='EARTH').sel(Time=2020).N2O_hist*self.settings['params']['gwp_n2o']/1e3
        ch4_2020 = self.xr_primap.sel(Region='EARTH').sel(Time=2020).CH4_hist*self.settings['params']['gwp_ch4']/1e3
        tot_2020 = n2o_2020+ch4_2020
        tot_2021 = n2o_2021+ch4_2021

        # Rescale CH4 and N2O trajectories
        compensation_form = np.array(list(np.linspace(0, 1, len(np.arange(self.settings['params']['start_year_analysis'], 2035))))+[1]*len(np.arange(2035, 2101)))
        xr_comp =  xr.DataArray(1-compensation_form, dims=['Time'], coords={'Time': np.arange(self.settings['params']['start_year_analysis'], 2101)})
        xr_nonco2_raw = xr_ch4_raw + xr_n2o_raw
        xr_nonco2_raw_2020 = xr_nonco2_raw.sel(Time=2021)
        xr_nonco2_raw = xr_nonco2_raw.sel(Time = np.arange(2021, 2101))

        def ms_temp(T):
            peaktemp = self.xr_ar6.sel(Variable='AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile').Value.max(dim='Time')
            return self.xr_ar6.ModelScenario[np.where((peaktemp < T+0.05) & (peaktemp > T-0.05))[0]]

        def check_monotomy(traj):
            vec = [traj[0]]
            for i in range(1, len(traj)):
                if traj[i]<=vec[i-1]:
                    vec.append(traj[i])
                else:
                    vec.append(vec[i-1])
            return np.array(vec)

        def rescale(traj):
            offset = traj.sel(Time = 2021) - tot_2021
            traj_scaled = (-xr_comp*offset+traj)
            return traj_scaled

        temps = []
        times = []
        nonco2 = []
        vals = []
        timings = []
        for timing_i, timing in enumerate(self.Timinglist):
            mslist = [self.ms_immediate, self.ms_delayed][timing_i]
            for temp_i, temp in enumerate(self.Tlist):
                ms_t = ms_temp(temp)
                ms = np.intersect1d(mslist, ms_t)
                if len(ms) == 0:
                    for n_i, n in enumerate(self.NonCO2list):
                        times = times + list(np.arange(2021, 2101))
                        timings = timings+[timing]*len(list(np.arange(2021, 2101)))
                        vals = vals+[np.nan]*len(list(np.arange(2021, 2101)))
                        nonco2 = nonco2+[n]*len(list(np.arange(2021, 2101)))
                        temps = temps + [temp]*len(list(np.arange(2021, 2101)))
                else:
                    reductions = (xr_nonco2_raw.sel(ModelScenario=ms, Time=2040)-xr_nonco2_raw_2020) / xr_nonco2_raw_2020
                    reds = reductions.Value.quantile(self.NonCO2list[::-1])
                    for n_i, n in enumerate(self.NonCO2list):
                        red = reds[n_i]
                        ms2 = reductions.ModelScenario[np.where(np.abs(reductions.Value - red) < 0.1)]
                        if len(ms2) == 0:
                            print(temp, n)
                        trajs = xr_nonco2_raw.sel(ModelScenario = ms2, Time=np.arange(2021, 2101))
                        trajectory_mean = rescale(trajs.Value.mean(dim='ModelScenario'))

                        # Harmonize reduction
                        red_traj = (trajectory_mean.sel(Time=2040) - tot_2020) / tot_2020
                        traj2 = -(1-xr_comp)*(red_traj-red*1.5)*xr_nonco2_raw_2020.mean().Value+trajectory_mean
                        trajectory_mean2 = check_monotomy(np.array(traj2))
                        times = times + list(np.arange(2021, 2101))
                        timings = timings+[timing]*len(list(np.arange(2021, 2101)))
                        vals = vals+list(trajectory_mean2)
                        nonco2 = nonco2+[n]*len(list(np.arange(2021, 2101)))
                        temps = temps + [temp]*len(list(np.arange(2021, 2101)))

        dict_nonco2 = {}
        dict_nonco2['Time'] = times
        dict_nonco2['NonCO2red'] = nonco2
        dict_nonco2['NonCO2_globe'] = vals
        dict_nonco2['Timing'] = timings
        dict_nonco2['Temperature'] = temps
        df_nonco2 = pd.DataFrame(dict_nonco2)
        dummy = df_nonco2.set_index(["NonCO2red", "Time", "Timing", 'Temperature'])
        self.xr_traj_nonco2 = xr.Dataset.from_dataframe(dummy)

        # Post-processing: making temperature dependence smooth
        self.xr_traj_nonco2_imm = self.xr_traj_nonco2.reindex({'Temperature': [1.5, 2.4]})
        self.xr_traj_nonco2_imm = self.xr_traj_nonco2_imm.reindex({'Temperature': self.Tlist})
        self.xr_traj_nonco2_imm = self.xr_traj_nonco2_imm.interpolate_na(dim='Temperature').sel(Timing='Immediate').expand_dims(Timing=["Immediate"])

        self.xr_traj_nonco2_del = self.xr_traj_nonco2.reindex({'Temperature': [1.56, 2.4]}) # Because 1.5 is empty for delayed
        self.xr_traj_nonco2_del = self.xr_traj_nonco2_del.reindex({'Temperature': self.Tlist})
        self.xr_traj_nonco2_del = self.xr_traj_nonco2_del.interpolate_na(dim='Temperature').sel(Timing='Delayed').expand_dims(Timing=["Delayed"])

        self.xr_traj_nonco2 = xr.merge([self.xr_traj_nonco2_imm, self.xr_traj_nonco2_del])

        # # Non-CO2 trajectories
        # xr_ch4_raw = self.xr_ar6.sel(Variable='Emissions|CH4', Time=np.arange(2020, 2101))*self.settings['params']['gwp_ch4']/1e3
        # xr_n2o_raw = self.xr_ar6.sel(Variable='Emissions|N2O', Time=np.arange(2020, 2101))*self.settings['params']['gwp_n2o']/1e6
        # n2o_2021 = self.xr_primap.sel(Region='EARTH').sel(Time=2021).N2O_hist*self.settings['params']['gwp_n2o']/1e3
        # ch4_2021 = self.xr_primap.sel(Region='EARTH').sel(Time=2021).CH4_hist*self.settings['params']['gwp_ch4']/1e3
        # n2o_2020 = self.xr_primap.sel(Region='EARTH').sel(Time=2020).N2O_hist*self.settings['params']['gwp_n2o']/1e3
        # ch4_2020 = self.xr_primap.sel(Region='EARTH').sel(Time=2020).CH4_hist*self.settings['params']['gwp_ch4']/1e3
        # tot_2020 = n2o_2020+ch4_2020
        # tot_2021 = n2o_2021+ch4_2021

        # # Rescale CH4 and N2O trajectories
        # compensation_form = np.array(list(np.linspace(0, 1, len(np.arange(self.settings['params']['start_year_analysis'], 2035))))+[1]*len(np.arange(2035, 2101)))
        # xr_comp =  xr.DataArray(1-compensation_form, dims=['Time'], coords={'Time': np.arange(self.settings['params']['start_year_analysis'], 2101)})
        # # offset = xr_ch4_raw.sel(Time = 2021) - ch4_2021
        # # xr_ch4 = (-xr_comp*offset+xr_ch4_raw)
        # # offset = xr_n2o_raw.sel(Time = 2021) - n2o_2021
        # # xr_n2o = (-xr_comp*offset+xr_n2o_raw)
        # # xr_nonco2 = xr_ch4 + xr_n2o
        # xr_nonco2_raw = xr_ch4_raw + xr_n2o_raw
        # xr_nonco2_raw_2020 = xr_nonco2_raw.sel(Time=2021)
        # xr_nonco2_raw = xr_nonco2_raw.sel(Time = np.arange(2021, 2101))
        # #xr_nonco2 = xr_nonco2.sel(Time = np.arange(2021, 2101))

        # # For each level of nonCO2 dimension, grab the trajectories
        # #reduction_2040_2020 = -(xr_nonco2.sel(Time=2040) - tot_2020) / tot_2020

        # def ms_temp(T):
        #     peaktemp = self.xr_ar6.sel(Variable='AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile').Value.max(dim='Time')
        #     return self.xr_ar6.ModelScenario[np.where((peaktemp < T+0.05) & (peaktemp > T-0.05))[0]]

        # def check_monotomy(traj):
        #     vec = [traj[0]]
        #     for i in range(1, len(traj)):
        #         if traj[i]<=vec[i-1]:
        #             vec.append(traj[i])
        #         else:
        #             vec.append(vec[i-1])
        #     return np.array(vec)

        # def rescale(traj):
        #     offset = traj.sel(Time = 2021) - tot_2021
        #     traj_scaled = (-xr_comp*offset+traj)
        #     return traj_scaled

        # temps = []
        # times = []
        # nonco2 = []
        # vals = []
        # timings = []
        # for temp_i, temp in enumerate(self.Tlist):
        #     ms_t = ms_temp(temp)
        #     reductions = (xr_nonco2_raw.sel(ModelScenario=ms_t, Time=2040)-xr_nonco2_raw_2020) / xr_nonco2_raw_2020
        #     reds = reductions.Value.quantile(self.NonCO2list[::-1])
        #     for n_i, n in enumerate(self.NonCO2list):
        #         red = reds[n_i]
        #         ms_2 = reductions.ModelScenario[np.where(np.abs(reductions.Value - red) < 0.1)]
        #         for timing_i, timing in enumerate(self.Timinglist):
        #             mslist = [self.ms_immediate, self.ms_delayed][timing_i]
        #             ms = np.intersect1d(ms_2, mslist)
        #             if len(ms) == 0: print(temp, n, timing)
        #             trajs = xr_nonco2_raw.sel(ModelScenario = ms, Time=np.arange(2021, 2101))
        #             trajectory_mean = rescale(trajs.Value.mean(dim='ModelScenario'))

        #             # Harmonize reduction
        #             red_traj = (trajectory_mean.sel(Time=2040) - tot_2020) / tot_2020
        #             traj2 = -(1-xr_comp)*(red_traj-red*1.5)*xr_nonco2_raw_2020.mean().Value+trajectory_mean
        #             # if n == 0.5 and timing == 'Immediate':
        #             #     trajectory_mean_imm = check_monotomy(trajectory_mean)
        #             # if n == 0.5 and timing == 'Delayed':
        #             #     trajectory_mean_del = check_monotomy(trajectory_mean)
        #             # if n == 0.7 or n == 0.8:
        #             #     dif = (trajectory_mean_del - trajectory_mean_imm)
        #             #     trajectory_mean2 = trajectory_mean2 + [dif, 0][timing_i]
        #             trajectory_mean2 = check_monotomy(np.array(traj2))
        #             times = times + list(np.arange(2021, 2101))
        #             timings = timings+[timing]*len(list(np.arange(2021, 2101)))
        #             vals = vals+list(trajectory_mean2)
        #             nonco2 = nonco2+[n]*len(list(np.arange(2021, 2101)))
        #             temps = temps + [temp]*len(list(np.arange(2021, 2101)))

        # dict_nonco2 = {}
        # dict_nonco2['Time'] = times
        # dict_nonco2['NonCO2red'] = nonco2
        # dict_nonco2['NonCO2_globe'] = vals
        # dict_nonco2['Timing'] = timings
        # dict_nonco2['Temperature'] = temps
        # df_nonco2 = pd.DataFrame(dict_nonco2)
        # dummy = df_nonco2.set_index(["NonCO2red", "Time", "Timing", 'Temperature'])
        # self.xr_traj_nonco2 = xr.Dataset.from_dataframe(dummy)

        # # Post-processing: making temperature dependence smooth
        # self.xr_traj_nonco2 = self.xr_traj_nonco2.reindex({'Temperature': [1.5, 2.4]})
        # self.xr_traj_nonco2 = self.xr_traj_nonco2.reindex({'Temperature': self.Tlist})
        # self.xr_traj_nonco2 = self.xr_traj_nonco2.interpolate_na(dim='Temperature')

    # =========================================================== #
    # =========================================================== #

    def determine_global_budgets(self):
        print('- Get global CO2 budgets')
        # CO2 budgets
        if self.settings['params']['toggle_co2_budgets'] == 'Forster':
            df_budgets = pd.read_csv("X:/user/dekkerm/Data/Budgets_Forster2023/ClimateIndicator-data-ed37002/data/carbon_budget/update_MAGICC_and_scenarios-budget.csv") # Now without the warming update in Forster, to link to IPCC AR6
            df_budgets = df_budgets[["dT_targets", "0.1", "0.17", "0.33", "0.5", "0.66", "0.83", '0.9']]
            dummy = df_budgets.melt(id_vars=["dT_targets"], var_name="Probability", value_name="Budget")
            ar = np.array(dummy['Probability'])
            ar = ar.astype(float).round(2)
            ar[ar == 0.66] = 0.67
            dummy['Probability'] = ar
            dummy['dT_targets'] = dummy['dT_targets'].astype(float).round(1)
            dummy = dummy.set_index(["dT_targets", "Probability"])
            #dummy['Budget'] = dummy['Budget'] + float(self.xr_primap.sel(Region='EARTH', Time=2021).CO2_hist)/1e3 + 40.9 # from Forster 2023 for the year 2022 -> 1 Jan 2021 as starting year! So not + float(self.xr_primap.sel(Region='EARTH', Time=2020).CO2_hist)/1e3 
            dummy['Budget'] = dummy['Budget'] - float(self.xr_primap.sel(Region='EARTH', Time=2020).CO2_hist)/1e3 # Forster without warming is from 2020 and on (so -2020 emissions)
            xr_bud_co2 = xr.Dataset.from_dataframe(dummy)
            xr_bud_co2 = xr_bud_co2.rename({'dT_targets': "Temperature"}).sel(Temperature = [1.5, 1.7, 2.0])
        elif self.settings['params']['toggle_co2_budgets'] == 'WG1':
            df_raw = pd.read_excel("X:/user/dekkerm/Data/IPCC/CarbonBudgets_IPCC.xlsx", sheet_name='Sheet1')
            df = df_raw.rename(columns={'Budget_17': 0.17, 'Budget_33': 0.33, 'Budget_50': 0.50, 'Budget_67': 0.67, 'Budget_83': 0.83})
            dummy = df.melt(id_vars=["Temperature"], var_name="Probability", value_name="Budget")
            dummy = dummy.set_index(["Temperature", "Probability"])
            dummy['Budget'] = dummy['Budget'] - float(self.xr_primap.sel(Region='EARTH', Time=2020).CO2_hist)/1e3
            xr_bud_co2 = xr.Dataset.from_dataframe(dummy)

        # # Fit nonco2 reduction in forster to different T levels
        # tot_2050 = np.zeros(3)
        # for T_i, T in enumerate([1.5, 1.7, 2.0]):
        #         tot_2050[T_i] = ch4_2020*(1-[0.48, 0.47, 0.35][T_i]) + n2o_2020*(1-[0.25, 0.15, 0.09][T_i])
        # forster_red = tot_2050/np.array(tot_2020)
        # f_a, f_b = np.polyfit(np.array([1.5, 1.7, 2.0]), forster_red, 1)

        def ms_temp(T):
            peaktemp = self.xr_ar6.sel(Variable='AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile').Value.max(dim='Time')
            return self.xr_ar6.ModelScenario[np.where((peaktemp < T+0.1) & (peaktemp > T-0.1))[0]]
            
        # Determine bunker emissions to subtract from global budget
        bunker_subtraction = []
        for t_i, t in enumerate(self.Tlist):
            ms = ms_temp(t)
            bunker_subtraction += [3.3/100] # Assuming bunker emissions keep constant (3.3% of global emissions) - https://www.pbl.nl/sites/default/files/downloads/pbl-2020-analysing-international-shipping-and-aviation-emissions-projections_4076.pdf

        # Interpolate
        n2o_2020 = self.xr_primap.sel(Region='EARTH').sel(Time=2020).N2O_hist*self.settings['params']['gwp_n2o']/1e3
        ch4_2020 = self.xr_primap.sel(Region='EARTH').sel(Time=2020).CH4_hist*self.settings['params']['gwp_ch4']/1e3
        tot_2020 = n2o_2020+ch4_2020
        Blist = np.zeros(shape=(len(self.Tlist), len(self.Plist), len(self.NonCO2list)))+np.nan
        for p_i, p in enumerate(self.Plist):
            a, b = np.polyfit(xr_bud_co2.Temperature, xr_bud_co2.sel(Probability = np.round(p, 2)).Budget, 1)
            for t_i, t in enumerate(self.Tlist):
                median_budget = (a*t+b)*(1-bunker_subtraction[t_i])
                #implied_leftover_median = f_a*t+f_b
                ar_50 = np.array((((self.xr_traj_nonco2.sel(NonCO2red=0.5, Time=2040, Temperature=t).mean(dim='Timing')-tot_2020) / tot_2020).round(2)).NonCO2_globe)
                ar_50 = ar_50[~np.isnan(ar_50)]
                ar_50[ar_50 > 0] =0
                ar_50[ar_50 < -0.8] = -0.8
                for n_i, n in enumerate(self.NonCO2list):
                        ar = np.array((((self.xr_traj_nonco2.sel(NonCO2red=n, Time=2040, Temperature=t).mean(dim='Timing')-tot_2020) / tot_2020).round(2)).NonCO2_globe)
                        ar = ar[~np.isnan(ar)]
                        ar[ar > 0] =0
                        ar[ar < -0.8] = -0.8
                        if self.settings['params']['toggle_co2_budgets']=='Forster': nonco2effect = self.xr_nonco2effects.sel(Temperature=t, NonCO2red=-ar.round(2)).EffectOnRCB # Effect on RCB based on how different non-CO2 paths are than they are in Forster
                        else: nonco2effect = float(self.xr_nonco2effects.sel(Temperature=t, NonCO2red=-ar.round(2)).EffectOnRCB) - float(self.xr_nonco2effects.sel(Temperature=t, NonCO2red=-ar_50.round(2)).EffectOnRCB)
                        Blist[t_i, p_i, n_i] = median_budget+nonco2effect#*(n-implied_leftover_median)
        data2 = xr.DataArray(Blist,
                            coords={'Temperature': self.Tlist,
                                    'Risk': (1-self.Plist).astype(float).round(2),
                                    'NonCO2red': self.NonCO2list},
                            dims=['Temperature', 'Risk', 'NonCO2red'])
        self.xr_co2_budgets = xr.Dataset({'Budget': data2})
        
    # =========================================================== #
    # =========================================================== #


    def determine_global_co2_trajectories(self):
        print('- Computing global co2 trajectories')
        # Initialize data arrays for co2
        startpoint = self.xr_primap.sel(Time=self.settings['params']['start_year_analysis'], Region="EARTH").CO2_hist
        #compensation_form = np.array(list(np.linspace(0, 1, len(np.arange(self.settings['params']['start_year_analysis'], 2101)))))#**1.1#+[1]*len(np.arange(2050, 2101)))
        
        compensation_form = np.array(list(np.linspace(0, 1, len(np.arange(self.settings['params']['start_year_analysis'], 2035))))+[1]*len(np.arange(2035, 2101)))
        xr_comp =  xr.DataArray(compensation_form, dims=['Time'], coords={'Time': np.arange(self.settings['params']['start_year_analysis'], 2101)})

        #compensation_form2 = np.array(list(np.linspace(0, 1, len(np.arange(self.settings['params']['start_year_analysis'], 2101)))))**0.5#+[1]*len(np.arange(2050, 2101)))
        def budget_harm(nz):
            compensation_form = np.array(list(np.linspace(0, 1, len(np.arange(self.settings['params']['start_year_analysis'], nz))))+[1]*len(np.arange(nz, 2101)))
            xr_comp2 =  xr.DataArray(compensation_form, dims=['Time'], coords={'Time': np.arange(self.settings['params']['start_year_analysis'], 2101)})
            return xr_comp2 / np.sum(np.linspace(0, 1, len(np.arange(self.settings['params']['start_year_analysis'], nz))))

        xr_traj_co2 = xr.Dataset(
            coords={
                'NegEmis': self.Neglist,
                'NonCO2red': self.NonCO2list,
                'Temperature': self.Tlist,
                'Risk': self.Plist,
                'Timing': self.Timinglist,
                'Time': np.arange(self.settings['params']['start_year_analysis'], 2101),
            }
        )

        pathways_data = {
            'CO2_globe': xr.DataArray(
                data=np.nan,
                coords=xr_traj_co2.coords,
                dims=('NegEmis', "NonCO2red", 'Temperature', 'Risk', 'Timing', 'Time'),
                attrs={'description': 'Pathway data'}
            )
        }

        xr_scen2_use = self.xr_ar6.sel(Variable='Emissions|CO2')
        xr_scen2_use = xr_scen2_use.reindex(Time = np.arange(2000, 2101, 10))
        xr_scen2_use = xr_scen2_use.reindex(Time = np.arange(2000, 2101))
        xr_scen2_use = xr_scen2_use.interpolate_na(dim="Time", method="linear")
        xr_scen2_use = xr_scen2_use.reindex(Time = np.arange(self.settings['params']['start_year_analysis'], 2101))

        co2_start = xr_scen2_use.sel(Time=self.settings['params']['start_year_analysis'])/1e3
        offsets = (startpoint/1e3-co2_start)
        emis_all = xr_scen2_use.sel(Time=np.arange(self.settings['params']['start_year_analysis'], 2101))/1e3 + offsets*(1-xr_comp)
        emis2100 = emis_all.sel(Time=2100)

        def remove_upward(ar):
            ar2 = np.zeros(len(ar))
            ar2[0:30] = ar[0:30]
            for i in range(30, len(ar)):
                if ar[i] > ar[i-1]:
                    ar2[i] = ar2[i-1]
                else:
                    ar2[i] = ar[i]
            return ar2

        def ms_temp(T):
            peaktemp = self.xr_ar6.sel(Variable='AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile').Value.max(dim='Time')
            return self.xr_ar6.ModelScenario[np.where((peaktemp < T+0.1) & (peaktemp > T-0.4))[0]] # 0.1 and 0.5 are chosen based on fitting with IPCC WGIII pathways from C1 and C3

        for temp_i, temp in enumerate(self.Tlist):
            ms1 = ms_temp(temp)
            # Shape impacted by timing of action
            for timing_i, timing in enumerate(self.Timinglist):
                if timing == 'Immediate': mslist = self.ms_immediate
                if timing == 'Delayed': mslist = self.ms_delayed
                ms2 = np.intersect1d(ms1, mslist)
                emis2100_i = emis2100.sel(ModelScenario=ms2)
                if len(ms2) == 0: # TODO have a look at this, the 1.5 scenarios do not have delayed action
                    3
                else:
                    # The 90-percentile of 2100 emissions
                    ms_90 = self.xr_ar6.sel(ModelScenario=ms2).ModelScenario[(emis2100_i >= emis2100_i.quantile(0.9-0.1)
                                                ).Value & (emis2100_i <= emis2100_i.quantile(0.9+0.1)).Value]
                
                    # The 10-percentile of 2100 emissions
                    ms_10 = self.xr_ar6.sel(ModelScenario=ms2).ModelScenario[(emis2100_i >= emis2100_i.quantile(0.1-0.1)
                                                ).Value & (emis2100_i <= emis2100_i.quantile(0.1+0.1)).Value]

                    # Difference and smoothen this
                    surplus_factor = emis_all.sel(ModelScenario = np.intersect1d(ms_90, ms2)).mean(dim='ModelScenario').Value - emis_all.sel(ModelScenario = np.intersect1d(ms_10, ms2)).mean(dim='ModelScenario').Value
                    surplus_factor2 = np.convolve(surplus_factor, np.ones(3)/3, mode='valid')
                    surplus_factor[1:-1] = surplus_factor2

                    for neg_i, neg in enumerate(self.Neglist):
                        xset = emis_all.sel(ModelScenario=ms2)-surplus_factor*(neg-0.5)
                        for risk_i, risk in enumerate(self.Plist):
                            for nonco2_i, nonco2 in enumerate(self.NonCO2list):
                                factor = (self.xr_co2_budgets.Budget.sel(Temperature=temp, Risk=risk, NonCO2red=nonco2) - xset.where(xset.Value > 0).sum(dim='Time')) / np.sum(compensation_form)
                                all_pathways = (1e3*(xset+factor*xr_comp)).Value/1e3
                                if len(all_pathways)>0:
                                    pathway = all_pathways.mean(dim='ModelScenario')
                                    pathway_sep = np.convolve(pathway, np.ones(3)/3, mode='valid') 
                                    pathway[1:-1] = pathway_sep
                                    offset = float(startpoint)/1e3 - pathway[0]
                                    pathway_final = np.array((pathway.T+offset)*1e3)

                                    # Remove upward emissions (harmonize later)
                                    pathway_final = remove_upward(np.array(pathway_final))

                                    # Harmonize by budget (iteration 3)
                                    try:
                                        nz = self.settings['params']['start_year_analysis']+np.where(pathway_final <= 0)[0][0]
                                    except:
                                        nz = 2100
                                    factor = (self.xr_co2_budgets.Budget.sel(Temperature=temp, Risk=risk, NonCO2red=nonco2)*1e3 - pathway_final[pathway_final > 0].sum())
                                    pathway_final2 = np.array((1e3*(pathway_final+factor*budget_harm(nz)))/1e3)

                                    try:
                                        nz = self.settings['params']['start_year_analysis']+np.where(pathway_final2 <= 0)[0][0]
                                    except:
                                        nz = 2100
                                    factor = (self.xr_co2_budgets.Budget.sel(Temperature=temp, Risk=risk, NonCO2red=nonco2)*1e3 - pathway_final2[pathway_final2 > 0].sum())
                                    pathway_final2 = (1e3*(pathway_final2+factor*budget_harm(nz)))/1e3
                                    
                                    pathways_data['CO2_globe'][neg_i, nonco2_i, temp_i, risk_i, timing_i, :] = pathway_final2
        self.xr_traj_co2 = xr_traj_co2.update(pathways_data)
        self.xr_traj_ghg_ds = (self.xr_traj_co2.CO2_globe+self.xr_traj_nonco2.NonCO2_globe*1e3)
        self.xr_traj_ghg = xr.merge([self.xr_traj_ghg_ds.to_dataset(name="GHG_globe"), self.xr_traj_co2.CO2_globe, self.xr_traj_nonco2.NonCO2_globe*1e3])

    # # =========================================================== #
    # # =========================================================== #

    def read_baseline(self):
        print('- Reading baseline emissions')
        df_ssps = []
        for i in range(5):
            df_base = pd.read_excel(self.settings['paths']['data']['baseline']+"SSP"+str(i+1)+".xlsx", sheet_name = 'Sheet1')
            df_base = df_base.drop(['Unnamed: 1'], axis=1)
            df_base = df_base.rename(columns={"COUNTRY": "Region"})
            df_base['Scenario'] = ['SSP'+str(i+1)]*len(df_base)
            df_base = pd.concat([df_base, pd.DataFrame(pd.Series(np.array(['EARTH']+[df_base[i].sum() for i in np.arange(2016, 2101)]+['SSP'+str(i+1)]), index = df_base.keys())).transpose()])
            df_ssps.append(df_base)
        df_base_all = pd.concat(df_ssps)
        df_base_all = df_base_all.reset_index(drop=True)
        dummy = df_base_all.melt(id_vars=["Region", "Scenario"], var_name="Time", value_name="CO2_base")
        dummy['Time'] = np.array(dummy['Time'].astype(int))
        dummy = dummy.set_index(["Region", "Scenario", "Time"])
        xr_base_raw = xr.Dataset.from_dataframe(dummy)
        xr_base_raw = xr_base_raw.reindex(Time = np.arange(self.settings['params']['start_year_analysis'], 2101))
        xr_base_raw = xr_base_raw.astype(float)

        # # Using a fraction, get total CO2 emissions
        fraction = (xr_base_raw.sel(Time=self.settings['params']['start_year_analysis']).CO2_base / self.xr_primap.sel(Time=self.settings['params']['start_year_analysis']).CO2_hist).mean(dim=['Scenario'])
        xr_base_harm_co2 = xr_base_raw / fraction

        # Assume nonCO2 following similar evolution as CO2 
        nonco2_2021 = self.xr_primap.sel(Time=self.settings['params']['start_year_analysis']).GHG_hist - self.xr_primap.sel(Time=self.settings['params']['start_year_analysis']).CO2_hist
        nonco2_base = (nonco2_2021*(xr_base_harm_co2/xr_base_harm_co2.sel(Time=self.settings['params']['start_year_analysis'])).to_array()).sel(variable='CO2_base')

        # Convert baseline emissions into GHG using this fraction (or offset)
        ghg_base = xr_base_harm_co2+nonco2_base
        ghg_base = ghg_base.rename({'CO2_base': "GHG_base"})
        ghg_base = ghg_base.reindex(Time = np.arange(self.settings['params']['start_year_analysis'], 2101))
        self.xr_base = ghg_base

    # =========================================================== #
    # =========================================================== #

    def read_ndc(self):
        print('- Reading NDC data')
        ghg_data = np.zeros(shape=(len(self.countries_iso), 3, 2, 2, len(np.arange(2010, 2051))))
        for cty_i, cty in enumerate(self.countries_iso):
            for cond_i, cond in enumerate(['conditional', 'range', 'unconditional']):
                for hot_i, hot in enumerate(['include', 'exclude']):
                    for amb_i, amb in enumerate(['low', 'high']):
                        params = self.settings['params']
                        path = f'X:/user/dekkerm/Data/NDC/ClimateResource_{params["version_ndcs"]}/{cond}/{hot}/{cty.lower()}_ndc_{params["version_ndcs"]}_CR_{cond}_{hot}.json'
                        try:
                            with open(path, 'r') as file:
                                json_data = json.load(file)
                            country_name = json_data['results']['country']['name']
                            series_items = json_data['results']['series']
                            for item in series_items:
                                columns = item['columns']
                                if columns['variable'] == "Emissions|Total GHG excl. LULUCF" and columns['category'] == "Updated NDC" and columns['ambition'] == amb:
                                    data = item['data']
                                    time_values = [int(year) for year in data.keys()]
                                    ghg_values = np.array(list(item['data'].values()))
                                    ghg_values[ghg_values == 'None'] = np.nan
                                    ghg_values = ghg_values.astype(float)
                                    ghg_values = ghg_values[np.array(time_values) >= 2010]
                                    ghg_data[cty_i, cond_i, hot_i, amb_i] = ghg_values
                                    #series.append([country_iso.upper(), country_name, "Emissions|Total GHG excl. LULUCF", conditionality, hot_air, ambition] + list(ghg_values))
                        except:
                            continue
        coords = {
            'Region': self.countries_iso,
            'Conditionality': ['conditional', 'range', 'unconditional'],
            'Hot_air': ['include', 'exclude'],
            'Ambition': ['low', 'high'],
            'Time': np.array(time_values)[np.array(time_values)>=2010],
        }
        data_vars = {
            'GHG_ndc': (['Region', 'Conditionality', 'Hot_air', 'Ambition', 'Time'], ghg_data),
        }
        xr_ndc = xr.Dataset(data_vars, coords=coords)

        #factors = xr_ndc.sel(Time=params['start_year_analysis']).GHG_ndc / self.xr_primap.sel(Time=params['start_year_analysis']).CO2_hist
        self.xr_ndc = xr_ndc#.assign(CO2_ndc = xr_ndc.GHG_ndc/factors)
        diff = self.xr_ndc.sel(Time=np.arange(2010, 2021)).mean(dim='Time') - self.xr_primap_excl.GHG_hist_excl.sel(Time=np.arange(2010, 2021)).mean(dim='Time')
        self.xr_primap['GHG_hist_ndc_corr'] = (self.xr_primap_excl.GHG_hist_excl + diff).GHG_ndc

    # =========================================================== #
    # =========================================================== #

    def merge_xr(self):
        print('- Merging xrarray object')
        xr_total = xr.merge([self.xr_ssp, self.xr_primap, self.xr_primap_excl, self.xr_unp, self.xr_co2_budgets, self.xr_traj_ghg, self.xr_base, self.xr_ndc])
        xr_total = xr_total.reindex(Region = self.regions_iso)
        xr_total = xr_total.reindex(Time = np.arange(1850, 2101))
        xr_total['GHG_globe'] = xr_total['GHG_globe'].astype(float)
        self.xr_total = xr_total.interpolate_na(dim="Time", method="linear")
        self.xr_total = self.xr_total.drop({'provenance'})

    # =========================================================== #
    # =========================================================== #

    def add_country_groups(self):
        print('- Add country groups')
        path_ctygroups = "X:/user/dekkerm/Data/" + "UNFCCC_Parties_Groups_noeu.xlsx"
        df = pd.read_excel(path_ctygroups, sheet_name = "Country groups")
        countries_iso = np.array(df["Country ISO Code"])
        list_of_regions = list(np.array(self.regions_iso).copy())
        for group_of_choice in ['G20', 'EU', 'G7', 'SIDS', 'LDC', 'Northern America', 'Australasia', 'African Group', 'Umbrella']:
            if group_of_choice != "EU":
                list_of_regions = list_of_regions + [group_of_choice]
            group_indices = countries_iso[np.array(df[group_of_choice]) == 1]
            country_to_eu = {}
            for cty in np.array(self.xr_total.Region):
                if cty in group_indices:
                    country_to_eu[cty] = [group_of_choice]
                else:
                    country_to_eu[cty] = ['']
            group_coord = xr.DataArray(
                [group for country in np.array(self.xr_total['Region']) for group in country_to_eu[country]],
                dims=['Region'],
                coords={'Region': [country for country in np.array(self.xr_total['Region']) for group in country_to_eu[country]]}
            )
            if group_of_choice == 'EU':
                xr_eu = self.xr_total[['Population', 'GDP', 'GHG_hist', "GHG_base", 'GHG_ndc']].groupby(group_coord).sum()#skipna=False)
            else:
                xr_eu = self.xr_total[['Population', 'GDP', 'GHG_hist', "GHG_base", 'GHG_ndc']].groupby(group_coord).sum(skipna=False)
            xr_eu2 = xr_eu.rename({'group': "Region"})
            dummy = self.xr_total.reindex(Region = list_of_regions)
            self.xr_total = xr.merge([dummy, xr_eu2])
            self.xr_total = self.xr_total.reindex(Region = list_of_regions)
        self.xr_total = self.xr_total
        #self.xr_total['GHG_base'][np.where(self.xr_total.Region=='EU')[0], np.array([2, 3, 4])] = np.nan # SSP3, 4, 5 are empty for Europe!

    # =========================================================== #
    # =========================================================== #

    def save(self): 
        print('- Save important files')
        np.save(self.settings['paths']['data']['datadrive'] + "all_regions.npy", self.regions_iso)
        np.save(self.settings['paths']['data']['datadrive'] + "all_regions_names.npy", self.regions_name)
        np.save(self.settings['paths']['data']['datadrive'] + "all_countries.npy", self.countries_iso)
        np.save(self.settings['paths']['data']['datadrive'] + "all_countries_names.npy", self.countries_name)
        self.xr_total.sel(Temperature=np.arange(1.5, 2.4+1e-9, 0.1).astype(float).round(2)).to_netcdf(self.settings['paths']['data']['datadrive']+'xr_dataread.nc',
            encoding={
                "Region": {"dtype": "str"},
                "Scenario": {"dtype": "str"},
                "Time": {"dtype": "int"},

                "Temperature": {"dtype": "float"},
                "NonCO2red": {"dtype": "float"},
                "NegEmis": {"dtype": "float"},
                "Risk": {"dtype": "float"},
                "Timing": {"dtype": "str"},

                "Conditionality": {"dtype": "str"},
                "Hot_air": {"dtype": "str"},
                "Ambition": {"dtype": "str"},

                "GDP": {"zlib": True, "complevel": 9},
                "Population": {"zlib": True, "complevel": 9},
                "GHG_hist": {"zlib": True, "complevel": 9},
                "GHG_globe": {"zlib": True, "complevel": 9},
                "GHG_base": {"zlib": True, "complevel": 9},
                "GHG_ndc": {"zlib": True, "complevel": 9},
                "GHG_hist_ndc_corr": {"zlib": True, "complevel": 9},
            },
            format="NETCDF4",
            engine="netcdf4",
        )
        xr_revised = self.xr_total.sel(Temperature=np.arange(1.5, 2.4+1e-9, 0.1).astype(float).round(2))
        xr_revised = xr_revised.assign_coords({"Timing": ("Timing", [0, 1])})
        xr_revised.drop_vars( ['source', 'Version']).to_netcdf(self.settings['paths']['data']['datadrive']+'xr_dataread_cabe.nc',
            encoding={
                "Region": {"dtype": "str"},
                "Scenario": {"dtype": "str"},
                "Time": {"dtype": "int"},

                "Temperature": {"dtype": "float"},
                "NonCO2red": {"dtype": "float"},
                "NegEmis": {"dtype": "float"},
                "Risk": {"dtype": "float"},

                "Conditionality": {"dtype": "str"},
                "Hot_air": {"dtype": "str"},
                "Ambition": {"dtype": "str"},

                "GDP": {"zlib": True, "complevel": 9},
                "Population": {"zlib": True, "complevel": 9},
                "GHG_hist": {"zlib": True, "complevel": 9},
                "GHG_globe": {"zlib": True, "complevel": 9},
                "GHG_base": {"zlib": True, "complevel": 9},
                "GHG_ndc": {"zlib": True, "complevel": 9},
                "GHG_hist_ndc_corr": {"zlib": True, "complevel": 9},
            },
            format="NETCDF4",
            engine="netcdf4",
        )

        self.xr_total.to_netcdf(self.settings['paths']['data']['datadrive']+'xr_dataread_pbl.nc',
            encoding={
                "Region": {"dtype": "str"},
                "Scenario": {"dtype": "str"},
                "Time": {"dtype": "int"},

                "Temperature": {"dtype": "float"},
                "NonCO2red": {"dtype": "float"},
                "NegEmis": {"dtype": "float"},
                "Risk": {"dtype": "float"},

                "Conditionality": {"dtype": "str"},
                "Hot_air": {"dtype": "str"},
                "Ambition": {"dtype": "str"},

                "GDP": {"zlib": True, "complevel": 9},
                "Population": {"zlib": True, "complevel": 9},
                "GHG_hist": {"zlib": True, "complevel": 9},
                "GHG_globe": {"zlib": True, "complevel": 9},
                "GHG_base": {"zlib": True, "complevel": 9},
                "GHG_ndc": {"zlib": True, "complevel": 9},
                "GHG_hist_ndc_corr": {"zlib": True, "complevel": 9},
            },
            format="NETCDF4",
            engine="netcdf4",
        )

        print('- Some pre-calculations for the AP allocation rule')
        xrt = self.xr_total.sel(Time=np.arange(self.settings['params']['start_year_analysis'], 2101))
        r1_nom = (xrt.GDP.sel(Region='EARTH') / xrt.Population.sel(Region='EARTH'))
        base_worldsum = xrt.GHG_base.sel(Region='EARTH')
        a=0
        for reg_i, reg in enumerate(self.countries_iso):
            rb_part1 = (xrt.GDP.sel(Region=reg) / xrt.Population.sel(Region=reg) / r1_nom)**(1/3.)
            rb_part2 = xrt.GHG_base.sel(Region=reg)*(base_worldsum - xrt.GHG_globe)/base_worldsum
            if a == 0:
                r = rb_part1*rb_part2
                if not np.isnan(np.max(r)):
                    rbw = r
                    a += 1
            else:
                r = rb_part1*rb_part2
                if not np.isnan(np.max(r)):
                    rbw += r
                    a += 1
        rbw.to_netcdf(self.settings['paths']['data']['datadrive']+'xr_rbw.nc')

        print('- Some pre-calculations for the GDR allocation rule')
        r=0
        hist_emissions_startyears = [1850, 1950, 1990]
        capability_thresholds = ['No', 'Th', 'PrTh']
        rci_weights = ['Resp', 'Half', 'Cap']
        for startyear_i, startyear in enumerate(hist_emissions_startyears):
            for th_i, th in enumerate(capability_thresholds):
                for weight_i, weight in enumerate(rci_weights):
                    # Read RCI
                    df_rci = pd.read_csv(self.settings['paths']['data']['external'] + "RCI/GDR_15_"+str(startyear)+"_"+th+"_"+weight+".xls", 
                                            delimiter='\t', 
                                            skiprows=30)[:-2]
                    df_rci = df_rci[['iso3', 'year', 'rci']]
                    iso3 = np.array(df_rci.iso3)
                    iso3[iso3 == 'CHK'] = 'CHN'
                    df_rci['iso3'] = iso3
                    df_rci['year'] = df_rci['year'].astype(int)
                    df_rci = df_rci.rename(columns={"iso3": 'Region', 'year': 'Time'})
                    df_rci['Historical_startyear'] = startyear
                    df_rci['Capability_threshold'] = th
                    df_rci['RCI_weight'] = weight
                    if r==0:
                        fulldf = df_rci
                        r+=1
                    else:
                        fulldf = pd.concat([fulldf, df_rci])
        dfdummy = fulldf.set_index(['Region', 'Time', 'Historical_startyear', 'Capability_threshold', 'RCI_weight'])
        xr_rci = xr.Dataset.from_dataframe(dfdummy)
        xr_rci = xr_rci.reindex({"Region": self.xr_total.Region})
        xr_rci.to_netcdf(self.settings['paths']['data']['datadrive']+'xr_rci.nc')
