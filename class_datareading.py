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
        print("# DATAREADING class                    #")

        self.current_dir = Path.cwd()

        # Read in Input YAML file
        with open(self.current_dir / 'input.yml') as file:
            self.settings = yaml.load(file, Loader=yaml.FullLoader)

        # Get dimension lists
        self.dim_temp = np.array(self.settings['dimension_ranges']['peak_temperature']).astype(float).round(2)
        self.dim_prob = np.array(self.settings['dimension_ranges']['risk_of_exceedance']).round(2)
        self.dim_negemis = np.array(self.settings['dimension_ranges']['negative_emissions']).round(2)
        self.dim_nonco2 = np.array(self.settings['dimension_ranges']['non_co2_reduction']).round(2)
        self.dim_timing = np.array(self.settings['dimension_ranges']['timing_of_mitigation_action'])

        self.time_future = np.arange(self.settings['params']['start_year_analysis'], 2101)
        self.time_past = np.arange(1850, self.settings['params']['start_year_analysis']+1)
        self.savepath = self.settings['paths']['data']['datadrive'] + "startyear_" + str(self.settings['params']['start_year_analysis']) + "/"
        self.recent_increment = int(np.floor(self.settings['params']['start_year_analysis']/5)*5)
        
        print("# startyear: ", self.settings['params']['start_year_analysis'])
        print("# ==================================== #")

    # =========================================================== #
    # =========================================================== #

    def read_general(self):
        print('- Reading general data')
        df_gen = pd.read_excel(self.settings['paths']['data']['external']+"UNFCCC_Parties_Groups_noeu.xlsx", sheet_name = "Country groups")
        self.countries_iso = np.array(list(df_gen["Country ISO Code"]))
        self.countries_name = np.array(list(df_gen["Name"]))
        self.regions_iso = np.array(list(df_gen["Country ISO Code"]) + ['EU', 'EARTH'])
        self.regions_name = np.array(list(df_gen["Name"])+['European Union', 'Earth'])

    # =========================================================== #
    # =========================================================== #

    def read_ssps(self):
        print('- Reading GDP and population data from SSPs')
        for i in range(6):
            df_ssp = pd.read_excel(self.settings['paths']['data']['external']+"SSPs/SSPs_v2023.xlsx", sheet_name='data')
            if i >= 1: df_ssp = df_ssp[(df_ssp.Model.isin(['OECD ENV-Growth 2023'])) & (df_ssp.Scenario == 'Historical Reference')]
            else: df_ssp = df_ssp[(df_ssp.Model.isin(['OECD ENV-Growth 2023', 'IIASA-WiC POP 2023'])) & (df_ssp.Scenario.isin(['SSP1', 'SSP2', 'SSP3', 'SSP4', 'SSP5']))]
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
            if i >= 1:
                dummy['Scenario'] = ['SSP'+str(i)]*len(dummy)
                xr_hist_gdp_i = xr.Dataset.from_dataframe(dummy.pivot(index=['Scenario', 'Region', 'Time'], columns='Variable', values='Value')).sel(Time=[1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015])
                self.xr_ssp = xr.merge([self.xr_ssp, xr_hist_gdp_i])
            else:
                self.xr_ssp = xr.Dataset.from_dataframe(dummy.pivot(index=['Scenario', 'Region', 'Time'], columns='Variable', values='Value')).reindex({'Time': np.arange(2020, 2101, 5)}).reindex({'Time': np.arange(1980, 2101, 5)})

    # =========================================================== #
    # =========================================================== #

    def read_undata(self):
        print('- Reading UN population data and gapminder, processed by OWID (for past population)')
        df_pop = pd.read_csv(self.settings['paths']['data']['external']+"UN Population/population_HYDE_UNP_Gapminder.csv")[['Code', 'Year', 'Population (historical)']].rename({'Code': "Region", 'Population (historical)': 'Population', 'Year': "Time"}, axis=1)
        reg = np.array(df_pop.Region)
        reg[reg == 'OWID_WRL'] = 'EARTH'
        df_pop.Region = reg
        self.xr_unp = xr.Dataset.from_dataframe(df_pop[df_pop.Region.isin(list(self.countries_iso)+['EARTH'])].set_index(['Region', 'Time'])).sel(Time = np.arange(1850, 2000))/1e6
        self.xr_unp_long = xr.Dataset.from_dataframe(df_pop[df_pop.Region.isin(list(self.countries_iso)+['EARTH'])].set_index(['Region', 'Time']))/1e6

    # =========================================================== #
    # =========================================================== #

    def read_hdi(self):
        print('- Read Human Development Index data')
        df_regions = pd.read_excel(self.settings['paths']['data']['external'] + "AR6_regionclasses.xlsx")
        df_regions = df_regions.sort_values(by=['name'])
        df_regions = df_regions.sort_index()

        self.df_hdi_raw = pd.read_excel(self.settings['paths']['data']['external'] + "HDI" + "/HDR21-22_Statistical_Annex_HDI_Table.xlsx", sheet_name='Rawdata')
        hdi_countries_raw = np.array(self.df_hdi_raw.Country)
        hdi_values_raw = np.array(self.df_hdi_raw.HDI).astype(str)
        hdi_values_raw[hdi_values_raw == ".."] = "nan"
        hdi_values_raw = hdi_values_raw.astype(float)
        hdi_av = np.nanmean(hdi_values_raw)

        # Construct new hdi object
        hdi_values = np.zeros(len(self.countries_iso))+np.nan
        hdi_sh_values = np.zeros(len(self.countries_iso))+np.nan
        for r_i, r in enumerate(self.countries_iso):
            reg = self.countries_name[r_i]
            wh = np.where(hdi_countries_raw == reg)[0]
            if len(wh) > 0:
                wh_i = wh[0]
                hdi_values[r_i] = hdi_values_raw[wh_i]
            elif r in ['ALA', 'ASM', "AIA", "ABW", "BMU", "ANT", "SCG", "BES", "BVT", "IOT", "VGB", "CYM", "CXR", "CCK", "COK",
                       'CUW', 'FLK', 'FRO', 'GUF', 'PYF', 'ATF', 'GMB', 'GIB', 'GRL', "GLP", 'GUM', "GGY", "HMD", "VAT", "IMN",
                       'JEY', "MAC", "MTQ", "MYT", "MSR", "NCL", "NIU", "NFK", "MNP", "PCN", "PRI", "REU", "BLM", "SHN", "SPM",
                       'SXM', 'SGS', "MAF", "SJM", "TKL", "TCA", "UMI", "VIR", "WLF", "ESH"]:
                hdi_values[r_i] = np.nan
            elif r == "USA":
                wh = np.where(hdi_countries_raw == "United States")[0][0]
                hdi_values[r_i] = hdi_values_raw[wh_i]
            elif r == "BHS":
                wh = np.where(hdi_countries_raw == "Bahamas")[0][0]
                hdi_values[r_i] = hdi_values_raw[wh_i]
            elif r == "GMB":
                wh = np.where(hdi_countries_raw == "Gambia")[0][0]
                hdi_values[r_i] = hdi_values_raw[wh_i]
            elif r == "CPV":
                wh = np.where(hdi_countries_raw == "Cabo Verde")[0][0]
                hdi_values[r_i] = hdi_values_raw[wh_i]
            elif r == "BOL":
                wh = np.where(hdi_countries_raw == "Bolivia (Plurinational State of)")[0][0]
                hdi_values[r_i] = hdi_values_raw[wh_i]
            elif r == "COD":
                wh = np.where(hdi_countries_raw == "Congo (Democratic Republic of the)")[0][0]
                hdi_values[r_i] = hdi_values_raw[wh_i]
            elif r == "COG":
                wh = np.where(hdi_countries_raw == "Congo")[0][0]
                hdi_values[r_i] = hdi_values_raw[wh_i]
            elif r == "CZE":
                wh = np.where(hdi_countries_raw == "Czechia")[0][0]
                hdi_values[r_i] = hdi_values_raw[wh_i]
            elif r == "EGY":
                wh = np.where(hdi_countries_raw == "Egypt")[0][0]
                hdi_values[r_i] = hdi_values_raw[wh_i]
            elif r == "HKG":
                wh = np.where(hdi_countries_raw == "Hong Kong, China (SAR)")[0][0]
                hdi_values[r_i] = hdi_values_raw[wh_i]
            elif r == "IRN":
                wh = np.where(hdi_countries_raw == "Iran (Islamic Republic of)")[0][0]
                hdi_values[r_i] = hdi_values_raw[wh_i]
            elif r == "PRK":
                wh = np.where(hdi_countries_raw == "Korea (Democratic People's Rep. of)")[0][0]
                hdi_values[r_i] = hdi_values_raw[wh_i]
            elif r == "KOR":
                wh = np.where(hdi_countries_raw == "Korea (Republic of)")[0][0]
                hdi_values[r_i] = hdi_values_raw[wh_i]
            elif r == "KGZ":
                wh = np.where(hdi_countries_raw == "Kyrgyzstan")[0][0]
                hdi_values[r_i] = hdi_values_raw[wh_i]
            elif r == "LAO":
                wh = np.where(hdi_countries_raw == "Lao People's Democratic Republic")[0][0]
                hdi_values[r_i] = hdi_values_raw[wh_i]
            elif r == "FSM":
                wh = np.where(hdi_countries_raw == "Micronesia (Federated States of)")[0][0]
                hdi_values[r_i] = hdi_values_raw[wh_i]
            elif r == "MDA":
                wh = np.where(hdi_countries_raw == "Moldova (Republic of)")[0][0]
                hdi_values[r_i] = hdi_values_raw[wh_i]
            elif r == "STP":
                wh = np.where(hdi_countries_raw == "Sao Tome and Principe")[0][0]
                hdi_values[r_i] = hdi_values_raw[wh_i]
            elif r == "SVK":
                wh = np.where(hdi_countries_raw == "Slovakia")[0][0]
                hdi_values[r_i] = hdi_values_raw[wh_i]
            elif r == "KNA":
                wh = np.where(hdi_countries_raw == "Saint Kitts and Nevis")[0][0]
                hdi_values[r_i] = hdi_values_raw[wh_i]
            elif r == "LCA":
                wh = np.where(hdi_countries_raw == "Saint Lucia")[0][0]
                hdi_values[r_i] = hdi_values_raw[wh_i]
            elif r == "VCT":
                wh = np.where(hdi_countries_raw == "Saint Vincent and the Grenadines")[0][0]
                hdi_values[r_i] = hdi_values_raw[wh_i]
            elif r == "SWZ":
                wh = np.where(hdi_countries_raw == "Eswatini (Kingdom of)")[0][0]
                hdi_values[r_i] = hdi_values_raw[wh_i]
            elif r == "TWN":
                wh = np.where(hdi_countries_raw == "China")[0][0]
                hdi_values[r_i] = hdi_values_raw[wh_i]
            elif r == "TZA":
                wh = np.where(hdi_countries_raw == "Tanzania (United Republic of)")[0][0]
                hdi_values[r_i] = hdi_values_raw[wh_i]
            elif r == "TUR":
                wh = np.where(hdi_countries_raw == "Türkiye")[0][0]
                hdi_values[r_i] = hdi_values_raw[wh_i]
            elif r == "VEN":
                wh = np.where(hdi_countries_raw == "Venezuela (Bolivarian Republic of)")[0][0]
                hdi_values[r_i] = hdi_values_raw[wh_i]
            elif r == "VNM":
                wh = np.where(hdi_countries_raw == "Viet Nam")[0][0]
                hdi_values[r_i] = hdi_values_raw[wh_i]
            elif r == "PSE":
                wh = np.where(hdi_countries_raw == "Palestine, State of")[0][0]
                hdi_values[r_i] = hdi_values_raw[wh_i]
            elif r == "YEM":
                wh = np.where(hdi_countries_raw == "Yemen")[0][0]
                hdi_values[r_i] = hdi_values_raw[wh_i]
            else:
                hdi_values[r_i] = np.nan
            try:
                pop = float(self.xr_unp_long.sel(Region=r, Time=2019).Population)
            except:
                pop = np.nan
            hdi_sh_values[r_i] = hdi_values[r_i]*pop
        hdi_sh_values = hdi_sh_values / np.nansum(hdi_sh_values)
        df_hdi = {}
        df_hdi['Region'] = self.countries_iso
        df_hdi["Name"] = self.countries_name
        df_hdi["HDI"] = hdi_values
        df_hdi = pd.DataFrame(df_hdi)
        df_hdi = df_hdi[["Region", 'HDI']]
        dfdummy = df_hdi.set_index(['Region'])
        self.xr_hdi = xr.Dataset.from_dataframe(dfdummy)

        df_hdi = {}
        df_hdi['Region'] = self.countries_iso
        df_hdi["Name"] = self.countries_name
        df_hdi["HDIsh"] = hdi_sh_values
        df_hdi = pd.DataFrame(df_hdi)
        df_hdi = df_hdi[["Region", 'HDIsh']]
        dfdummy = df_hdi.set_index(['Region'])
        self.xr_hdish = xr.Dataset.from_dataframe(dfdummy)

    # =========================================================== #
    # =========================================================== #

    def read_historicalemis_jones(self):
        print('- Reading historical emissions (jones)') # No harmonization with the KEV anymore, but it's also much closer now
        xr_primap2 = xr.open_dataset("X:/user/dekkerm/Data/PRIMAP/Guetschow_et_al_2024-PRIMAP-hist_v2.5.1_final_no_rounding_27-Feb-2024.nc")
        df_nwc = pd.read_csv('X:/user/dekkerm/Data/NationalWarningContributions/EMISSIONS_ANNUAL_1830-2022.csv')
        xr_nwc = xr.Dataset.from_dataframe(df_nwc.drop(columns=['CNTR_NAME', 'Unit']).set_index(['ISO3', 'Gas', 'Component', 'Year']))
        regs = np.array(xr_nwc.ISO3)
        regs[regs == 'GLOBAL'] = 'EARTH'
        xr_nwc['ISO3'] = regs
        xr_nwc_tot = (xr_nwc.sel(Gas='CH[4]')*self.settings['params']['gwp_ch4']/1e3+xr_nwc.sel(Gas='N[2]*O')*self.settings['params']['gwp_n2o']/1e3+xr_nwc.sel(Gas='CO[2]')*1).drop_vars(['Gas'])
        xr_nwc_co2 = xr_nwc.sel(Gas='CO[2]').drop_vars(['Gas'])
        xr_nwc_ch4 = xr_nwc.sel(Gas='CH[4]').drop_vars(['Gas'])*self.settings['params']['gwp_ch4']/1e3
        xr_nwc_n2o = xr_nwc.sel(Gas='N[2]*O').drop_vars(['Gas'])*self.settings['params']['gwp_n2o']/1e3
        xr_primap_agri = xr_primap2['KYOTOGHG (AR6GWP100)'].rename({'area (ISO3)': 'Region', 'scenario (PRIMAP-hist)': 'scen', 'category (IPCC2006_PRIMAP)': 'cat'}).sel(scen='HISTTP', provenance='derived', cat=['M.AG'], source='PRIMAP-hist_v2.5.1_final_nr').sum(dim='cat').drop_vars(['source', 'provenance', 'scen'])
        xr_primap_agri['time'] = np.arange(1750, 2023)
        xr_primap_agri = xr_primap_agri.rename({'time': 'Time'})
        xr_primap_agri_co2 = xr_primap2['CO2'].rename({'area (ISO3)': 'Region', 'scenario (PRIMAP-hist)': 'scen', 'category (IPCC2006_PRIMAP)': 'cat'}).sel(scen='HISTTP', provenance='derived', cat=['M.AG'], source='PRIMAP-hist_v2.5.1_final_nr').sum(dim='cat').drop_vars(['source', 'provenance', 'scen'])
        xr_primap_agri_co2['time'] = np.arange(1750, 2023)
        xr_primap_agri_co2 = xr_primap_agri_co2.rename({'time': 'Time'})

        xr_ghghist = (xr_nwc_tot.rename({'ISO3': 'Region', 'Year': 'Time', 'Data': 'GHG_hist'}).sel(Component='Total').drop_vars('Component'))
        xr_co2hist = (xr_nwc_co2.rename({'ISO3': 'Region', 'Year': 'Time', 'Data': 'CO2_hist'}).sel(Component='Total').drop_vars('Component'))
        xr_ch4hist = (xr_nwc_ch4.rename({'ISO3': 'Region', 'Year': 'Time', 'Data': 'CH4_hist'}).sel(Component='Total').drop_vars('Component'))
        xr_n2ohist = (xr_nwc_n2o.rename({'ISO3': 'Region', 'Year': 'Time', 'Data': 'N2O_hist'}).sel(Component='Total').drop_vars('Component'))
        xr_ghgexcl = (xr_nwc_tot.rename({'ISO3': 'Region', 'Year': 'Time'}).sel(Component='Total').drop_vars('Component') - xr_nwc_tot.rename({'ISO3': 'Region', 'Year': 'Time'}).sel(Component='LULUCF').drop_vars('Component') + xr_primap_agri/1e6).rename({'Data': 'GHG_hist_excl'})
        xr_co2excl = (xr_nwc_co2.rename({'ISO3': 'Region', 'Year': 'Time'}).sel(Component='Total').drop_vars('Component') - xr_nwc_co2.rename({'ISO3': 'Region', 'Year': 'Time'}).sel(Component='LULUCF').drop_vars('Component') + xr_primap_agri_co2/1e6).rename({'Data': 'CO2_hist_excl'})
        self.xr_hist = xr.merge([xr_ghghist, xr_ghgexcl, xr_co2hist, xr_co2excl,xr_ch4hist, xr_n2ohist])*1e3
        self.xr_hist = self.xr_hist.reindex({'Region': self.regions_iso})

        self.xr_ghg_afolu = xr_nwc_tot.rename({'ISO3': 'Region', 'Year': 'Time'}).sel(Component='LULUCF').drop_vars('Component')
        self.xr_ghg_agri = xr_primap_agri/1e6

        # Also read EDGAR for purposes of using CR data (note that this is GHG excl LULUCF)
        df_edgar = pd.read_excel('X:/user/dekkerm/Data/EDGAR/EDGARv8.0_FT2022_GHG_booklet_2023.xlsx', sheet_name='GHG_totals_by_country').drop(['Country'], axis=1).set_index('EDGAR Country Code')
        df_edgar.columns = df_edgar.columns.astype(int)

        # drop second-to-last row
        df_edgar = df_edgar.drop(df_edgar.index[-2])

        # Rename index column
        df_edgar.index.name = 'Region'

        # Melt time columns into a time index
        df_edgar = df_edgar.reset_index().melt(id_vars='Region', var_name='Time', value_name='Emissions').set_index(['Region', 'Time'])

        # Convert to xarray
        self.xr_edgar = df_edgar.to_xarray()

        self.xr_primap = xr_primap2.rename({'area (ISO3)': 'Region', 'scenario (PRIMAP-hist)': 'Scenario', 'category (IPCC2006_PRIMAP)': 'Category'}).sel(provenance='derived', source='PRIMAP-hist_v2.5.1_final_nr')
        self.xr_primap = self.xr_primap.assign_coords(time=self.xr_primap.time.dt.year)

        # Add EU (this is required for the NDC data reading)
        df = pd.read_excel("X:/user/dekkerm/Data/UNFCCC_Parties_Groups_noeu.xlsx", sheet_name = "Country groups")
        countries_iso = np.array(df["Country ISO Code"])
        group_eu = countries_iso[np.array(df["EU"]) == 1]
        self.xr_hist.GHG_hist.loc[dict(Region='EU')] = self.xr_hist.GHG_hist.sel(Region=group_eu).sum("Region")

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
                                                    'Emissions|CO2|AFOLU|Land',
                                                    'Emissions|CH4|AFOLU|Land',
                                                    'Emissions|N2O|AFOLU|Land',
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
        vetting_recentyear = np.array(self.xr_ar6_prevet.ModelScenario[np.where(np.abs(self.xr_ar6_prevet.sel(Time=self.recent_increment, Variable='Emissions|CO2').Value - self.xr_hist.sel(Region='EARTH', Time=self.recent_increment).CO2_hist) < 1e4)[0]])
        vetting_total = np.intersect1d(vetting_nans, vetting_recentyear)
        self.xr_ar6 = self.xr_ar6_prevet.sel(ModelScenario=vetting_total)
        self.ms_immediate = np.array(df_ar6_meta[df_ar6_meta.Policy_category.isin(['P2', 'P2a', 'P2b', 'P2c'])].ModelScenario)
        self.ms_delayed = np.array(df_ar6_meta[df_ar6_meta.Policy_category.isin(['P3a', 'P3b', 'P3c'])].ModelScenario)
        self.xr_ar6_landuse = (self.xr_ar6.sel(Variable='Emissions|CO2|AFOLU|Land')*1 + 
                               self.xr_ar6.sel(Variable='Emissions|CH4|AFOLU|Land')*self.settings['params']['gwp_ch4'] + 
                               self.xr_ar6.sel(Variable='Emissions|N2O|AFOLU|Land')*self.settings['params']['gwp_n2o']/1000)
        self.xr_ar6_landuse = self.xr_ar6_landuse.rename({'Value': 'GHG_LULUCF'})
        self.xr_ar6_landuse = self.xr_ar6_landuse.assign(CO2_LULUCF = self.xr_ar6.sel(Variable='Emissions|CO2|AFOLU|Land').Value)

        # Take averages of GHG excluding land use for the C-categories (useful for the Robiou paper)
        xr_both = xr.merge([self.xr_ar6, self.xr_ar6_landuse])
        xr_ar6_nozeros = xr_both.where(xr_both > -1e9, np.nan).where(xr_both != 0, np.nan)
        xr_averages = []
        for i in range(6):
            C = [['C1'], ['C1', 'C2'], ['C2'], ['C3'], ['C6'], ['C7']][i]
            Cname = ['C1', 'C1+C2', 'C2', 'C3', 'C6', 'C7'][i]
            C_cat = np.intersect1d(np.array(xr_ar6_nozeros.ModelScenario),
                                   np.array(df_ar6_meta[df_ar6_meta.Category.isin(C)].ModelScenario))
            xr_averages.append(xr_ar6_nozeros.sel(ModelScenario=C_cat).mean(dim='ModelScenario').expand_dims(Category=[Cname]))
        xr_av = xr.merge(xr_averages)
        self.xr_ar6_C = xr.merge([(xr_av.Value.sel(Variable='Emissions|Kyoto Gases') - xr_av.GHG_LULUCF).to_dataset(name="GHG_excl_C").drop_vars('Variable'),
                                  (xr_av.Value.sel(Variable='Emissions|CO2') - xr_av.CO2_LULUCF).to_dataset(name="CO2_excl_C").drop_vars('Variable'),
                                  (xr_av.Value.sel(Variable=['Carbon Sequestration|CCS', 'Carbon Sequestration|Direct Air Capture']).sum(dim='Variable', skipna=False)).to_dataset(name="CO2_neg_C")
        ])
        self.xr_ar6_C = self.xr_ar6_C.reindex(Time = np.arange(2000, 2101, 10))
        self.xr_ar6_C = self.xr_ar6_C.reindex(Time = np.arange(2000, 2101))

        # Bunker subtraction
        df_elevate_bunkers = pd.read_csv("X:/user/dekkerm/Data/Bunkers/elevate-internal_snapshot_1739887620.csv")[:-1]
        mods = np.array(df_elevate_bunkers.Model)
        scens = np.array(df_elevate_bunkers.Scenario)
        modscens = np.array([mods[i]+'|'+scens[i] for i in range(len(scens))])
        df_elevate_bunkers['ModelScenario'] = modscens
        df_elevate_bunkers = df_elevate_bunkers.drop(['Model', 'Scenario', 'Region', 'Unit'], axis=1)
        dummy = df_elevate_bunkers.melt(id_vars=["ModelScenario", "Variable"], var_name="Time", value_name="Value")
        dummy['Time'] = np.array(dummy['Time'].astype(int))
        dummy = dummy.set_index(["ModelScenario", "Variable", "Time"])
        xr_elevate_bunkers = xr.Dataset.from_dataframe(dummy)
        xr_elevate_bunkers = xr_elevate_bunkers.reindex({'Time': np.arange(2010, 2101, 10)})

        modscens = np.array(xr_elevate_bunkers.ModelScenario)
        categories = []
        for ms in modscens:
            if xr_elevate_bunkers.sel(ModelScenario = ms, Variable = 'AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile').max().Value < 1.5:
                categories.append('C1')
            elif xr_elevate_bunkers.sel(ModelScenario = ms, Variable = 'AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile').max().Value < 1.7:
                categories.append('C2')
            elif xr_elevate_bunkers.sel(ModelScenario = ms, Variable = 'AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile').max().Value < 2.0:
                categories.append('C3')
            elif xr_elevate_bunkers.sel(ModelScenario = ms, Variable = 'AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile').max().Value < 3.0:
                categories.append('C6')
            elif xr_elevate_bunkers.sel(ModelScenario = ms, Variable = 'AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile').max().Value < 4.0:
                categories.append('C7')
            elif xr_elevate_bunkers.sel(ModelScenario = ms, Variable = 'AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile').max().Value > 4.0:
                categories.append('C8')
            else:
                categories.append('C9') # Fictuous category to show that for these mds, there is no temperature assessment
        categories = np.array(categories)

        xrs = []
        for temp in [1.7, 2.0, 3.0]:
            if temp == 1.7: cat = 'C2'
            elif temp == 2.0: cat = 'C3'
            elif temp == 3.0: cat = 'C6'
            xrs.append((xr_elevate_bunkers.sel(Variable = 'Emissions|CO2|Energy|Demand|Bunkers', ModelScenario=modscens[categories==cat]).median(dim='ModelScenario')).expand_dims(Temperature=[temp]))
        xr_all = xr.concat(xrs, dim='Temperature')
        xr_all = xr_all.reindex(Temperature=[1.5, 1.6, 1.7, 2.0, 3.0, 4.0, 4.5])

        # Extrapolation
        vals = xr_all.loc[dict(Temperature=3.0)]-xr_all.loc[dict(Temperature=1.7)]
        vals = vals.where(vals>0, 0)
        xr_all.loc[dict(Temperature=1.5)] = xr_all.loc[dict(Temperature=1.7)]-vals*(0.2/1.3)
        xr_all.loc[dict(Temperature=1.6)] = xr_all.loc[dict(Temperature=1.7)]-vals*(0.1/1.3)
        xr_all.loc[dict(Temperature=4.0)] = xr_all.loc[dict(Temperature=3.0)]+vals*(1.0/1.3)
        xr_all.loc[dict(Temperature=4.5)] = xr_all.loc[dict(Temperature=3.0)]+vals*(1.5/1.3)

        xr_all = xr_all.rename({'Temperature': "Category"}).drop_vars('Variable').rename({"Value": "CO2_bunkers_C"})

        # Rename ticks of temperature
        xr_all = xr_all.assign_coords(Category=['C1', 'C1+C2', 'C2', 'C3', 'C6', 'C7', 'C8'])
        self.xr_ar6_C_bunkers = xr_all

    # =========================================================== #
    # =========================================================== #

    def nonco2variation(self):
        path_file = self.settings['paths']['data']['lambollrepo'] + "InputData/MAGICCAR6emWG3scen/" + "job-20211019-ar6-nonco2_Raw-GSAT-Non-CO2.csv"
        df = pd.read_csv(path_file)
        df = df[['model',
                'scenario',
                'Category',
                'variable',
                'permafrost',
                'median peak warming (MAGICCv7.5.3)',
                'p33 peak warming (MAGICCv7.5.3)',
                'p67 peak warming (MAGICCv7.5.3)',
                'median year of peak warming (MAGICCv7.5.3)',
                'p33 year of peak warming (MAGICCv7.5.3)',
                'p67 year of peak warming (MAGICCv7.5.3)',
                ] + list(df.keys()[28:])]

        df.columns = df.columns.str.replace(r"(\d{4})-01-01 00:00:00", r"\1", regex=True)
        df.rename(columns={'variable': 'NonCO2WarmingQuantile',
                        'permafrost' : 'Permafrost',
                        "median peak warming (MAGICCv7.5.3)": "T(0.5)",
                        "p33 peak warming (MAGICCv7.5.3)": "T(0.33)",
                        "p67 peak warming (MAGICCv7.5.3)": "T(0.67)",
                        "median year of peak warming (MAGICCv7.5.3)": "Y(0.50)",
                        "p33 year of peak warming (MAGICCv7.5.3)": "Y(0.33)",
                        "p67 year of peak warming (MAGICCv7.5.3)": "Y(0.67)"}, inplace=True)

        # ModelScenario
        modscen = []
        df['ModelScenario'] = df['model'] + '|' + df['scenario']
        df = df.drop(columns=['model', 'scenario'])

        # Rename warming quantiles
        quantiles_map = {
            f'AR6 climate diagnostics|Raw Surface Temperature (GSAT)|Non-CO2|MAGICCv7.5.3|{i}th Percentile': float(i) / 100
            for i in ['10.0', '16.7', '33.0', '5.0', '50.0', '67.0', '83.3', '90.0', '95.0']
            }
        df['NonCO2WarmingQuantile'] = df['NonCO2WarmingQuantile'].replace(quantiles_map).astype(float).round(2)

        # Only consider excluding permafrost
        df = df[df.Permafrost == False]
        df = df.drop(columns=['Permafrost'])

        # Xarray for time-varying data
        df_dummy = df[['ModelScenario', 'NonCO2WarmingQuantile']+list(np.arange(1995, 2101).astype(str))].melt(id_vars=['ModelScenario', 'NonCO2WarmingQuantile'], var_name='Time', value_name='NonCO2warming')
        df_dummy['Time'] = df_dummy['Time'].astype(int)
        df_dummy = df_dummy.set_index(['ModelScenario', 'NonCO2WarmingQuantile', 'Time'])
        xr_lamboll = xr.Dataset.from_dataframe(df_dummy)

        # Xarray for peak warming years
        df_peakyears = df[['ModelScenario', 'NonCO2WarmingQuantile', 'Y(0.50)', 'Y(0.33)', 'Y(0.67)']]
        df_peakyears = df_peakyears.rename(columns={'Y(0.50)': 0.5,
                                                    'Y(0.33)': 0.33,
                                                    'Y(0.67)': 0.67})
        df_peakyears = df_peakyears.melt(id_vars=['ModelScenario', 'NonCO2WarmingQuantile'], var_name='TCRE', value_name='PeakYear')
        df_dummy = df_peakyears.set_index(['ModelScenario', 'NonCO2WarmingQuantile', 'TCRE'])
        xr_peakyears = xr.Dataset.from_dataframe(df_dummy)

        # Xarray for full peak warming
        # Also extrapolate for 17 and 83 percentiles (based on normal distribution assumption)
        df_peaktemps = df[['ModelScenario', 'T(0.5)', 'T(0.33)', 'T(0.67)']].drop_duplicates()
        df_peaktemps = df_peaktemps.rename(columns={'T(0.5)': 0.5,
                                                    'T(0.33)': 0.33,
                                                    'T(0.67)': 0.67})
        df_peaktemps = df_peaktemps.melt(id_vars=['ModelScenario'], var_name='TCRE', value_name='Temperature')
        df_dummy = df_peaktemps.set_index(['ModelScenario', 'TCRE'])
        xr_temperatures = xr.Dataset.from_dataframe(df_dummy)
        xr_temperatures17 = (xr_temperatures.sel(TCRE=0.33) - 1*(xr_temperatures.sel(TCRE=0.67) - xr_temperatures.sel(TCRE=0.33))).drop_vars("TCRE").expand_dims({'TCRE': [0.17]})
        xr_temperatures83 = (xr_temperatures.sel(TCRE=0.67) + 1*(xr_temperatures.sel(TCRE=0.67) - xr_temperatures.sel(TCRE=0.33))).drop_vars("TCRE").expand_dims({'TCRE': [0.83]})
        xr_temperatures = xr.merge([xr_temperatures, xr_temperatures17, xr_temperatures83])

        # Peak warming -> at the peak year.
        xr_peaknonco2warming_all = xr_lamboll.sel(Time = xr_peakyears.PeakYear).rename({'NonCO2warming': 'PeakWarming'})

        # Now we assume that nonco2 warming quantiles are the same as the peak warming quantiles
        # That is: climate sensitivity for the full picture (TCRE) is directly related to climate sensitivity to only non-CO2
        # Also extrapolate for 17 and 83 percentiles (based on normal distribution assumption)
        # relation nonco2 peak warming to TCRE is not trivial, because the peakyears are also dependent on TCRE!
        # However, as it turns out, higher TCRE implies in practically all cases a higher nonCO2 warming at the peak year
        xr_peaknonco2warming_all = xr_peaknonco2warming_all.drop_vars('Time')
        peak50 = xr_peaknonco2warming_all.sel(NonCO2WarmingQuantile=0.5, TCRE=[0.5]).drop_vars("NonCO2WarmingQuantile")
        peak33 = xr_peaknonco2warming_all.sel(NonCO2WarmingQuantile=0.33, TCRE=[0.33]).drop_vars("NonCO2WarmingQuantile")
        peak67 = xr_peaknonco2warming_all.sel(NonCO2WarmingQuantile=0.67, TCRE=[0.67]).drop_vars("NonCO2WarmingQuantile")
        peak17 = (xr_peaknonco2warming_all.sel(NonCO2WarmingQuantile=0.33, TCRE=0.33)-1*(xr_peaknonco2warming_all.sel(NonCO2WarmingQuantile=0.67, TCRE=0.67)-xr_peaknonco2warming_all.sel(NonCO2WarmingQuantile=0.33, TCRE=0.33))).drop_vars("NonCO2WarmingQuantile").drop_vars("TCRE").expand_dims({'TCRE': [0.17]})
        peak83 = (xr_peaknonco2warming_all.sel(NonCO2WarmingQuantile=0.67, TCRE=0.67)+1*(xr_peaknonco2warming_all.sel(NonCO2WarmingQuantile=0.67, TCRE=0.67)-xr_peaknonco2warming_all.sel(NonCO2WarmingQuantile=0.33, TCRE=0.33))).drop_vars("NonCO2WarmingQuantile").drop_vars("TCRE").expand_dims({'TCRE': [0.83]})
        xr_peaknonco2warming = xr.merge([peak50, peak33, peak67, peak17, peak83])

        # Invert axis for Risk coordinate
        xr_peaknonco2warming = xr_peaknonco2warming.assign_coords(TCRE=[0.83, 0.67, 0.5, 0.33, 0.17]).rename({'TCRE': "Risk"})
        xr_temperatures = xr_temperatures.assign_coords(TCRE=[0.83, 0.67, 0.5, 0.33, 0.17]).rename({'TCRE': "Risk"})

        # Save for later use
        self.xr_temperatures = xr_temperatures
        self.xr_nonco2warmings = xr_peaknonco2warming
        self.xr_nonco2warming_wrt_start = xr_peaknonco2warming - xr_lamboll.rename({'NonCO2WarmingQuantile': 'Risk'}).sel(Time=self.settings['params']['start_year_analysis']).NonCO2warming

    # =========================================================== #
    # =========================================================== #

    def determine_global_nonco2_trajectories(self):
        print('- Computing global nonco2 trajectories')
        # Relationship between non-co2 reduction and budget is based on Rogelj et al and requires the year 2020 (even though startyear may be different) - not a problem
        xr_ch4_raw = self.xr_ar6.sel(Variable='Emissions|CH4')*self.settings['params']['gwp_ch4']
        xr_n2o_raw = self.xr_ar6.sel(Variable='Emissions|N2O')*self.settings['params']['gwp_n2o']/1e3
        n2o_start = self.xr_hist.sel(Region='EARTH').sel(Time=self.settings['params']['start_year_analysis']).N2O_hist
        ch4_start = self.xr_hist.sel(Region='EARTH').sel(Time=self.settings['params']['start_year_analysis']).CH4_hist
        n2o_2020 = self.xr_hist.sel(Region='EARTH').sel(Time=2020).N2O_hist
        ch4_2020 = self.xr_hist.sel(Region='EARTH').sel(Time=2020).CH4_hist
        tot_2020 = n2o_2020+ch4_2020
        tot_start = n2o_start+ch4_start

        # Rescale CH4 and N2O trajectories
        compensation_form = np.array(list(np.linspace(0, 1, len(np.arange(self.settings['params']['start_year_analysis'], self.settings['params']['harmonization_year']))))+[1]*len(np.arange(self.settings['params']['harmonization_year'], 2101)))
        xr_comp =  xr.DataArray(1-compensation_form, dims=['Time'], coords={'Time': np.arange(self.settings['params']['start_year_analysis'], 2101)})
        xr_nonco2_raw = xr_ch4_raw + xr_n2o_raw
        xr_nonco2_raw_start = xr_nonco2_raw.sel(Time=self.settings['params']['start_year_analysis'])
        xr_nonco2_raw = xr_nonco2_raw.sel(Time = np.arange(self.settings['params']['start_year_analysis'], 2101))

        def ms_temp(temp, risk):
            return self.xr_temperatures.ModelScenario[np.where(np.abs(temp - self.xr_temperatures.Temperature.sel(Risk=risk)) < 0.2)[0]].values

        def check_monotomy(traj):
            vec = [traj[0]]
            for i in range(1, len(traj)):
                if traj[i]<=vec[i-1]:
                    vec.append(traj[i])
                else:
                    vec.append(vec[i-1])
            return np.array(vec)

        def rescale(traj):
            offset = traj.sel(Time = self.settings['params']['start_year_analysis']) - tot_start
            traj_scaled = (-xr_comp*offset+traj)
            return traj_scaled

        xr_reductions = (xr_nonco2_raw.sel(Time=2040)-xr_nonco2_raw_start) / xr_nonco2_raw_start

        temps = []
        risks = []
        times = []
        nonco2 = []
        vals = []
        timings = []
        for temp_i, temp in enumerate(self.dim_temp):
            for p_i, p in enumerate(self.dim_prob):
                ms1 = ms_temp(temp, p)
                for timing_i, timing in enumerate(self.dim_timing):
                    if timing == 'Immediate' or temp in [1.5, 1.56, 1.6] and timing == 'Delayed':
                        mslist = self.ms_immediate
                    else:
                        mslist = self.ms_delayed
                    ms2 = np.intersect1d(ms1, mslist)
                    if len(ms2) == 0:
                        for n_i, n in enumerate(self.dim_nonco2):
                            times = times + list(np.arange(self.settings['params']['start_year_analysis'], 2101))
                            vals = vals+[np.nan]*len(list(np.arange(self.settings['params']['start_year_analysis'], 2101)))
                            nonco2 = nonco2+[n]*len(list(np.arange(self.settings['params']['start_year_analysis'], 2101)))
                            temps = temps + [temp]*len(list(np.arange(self.settings['params']['start_year_analysis'], 2101)))
                            risks = risks + [p]*len(list(np.arange(self.settings['params']['start_year_analysis'], 2101)))
                            timings = timings + [timing]*len(list(np.arange(self.settings['params']['start_year_analysis'], 2101)))
                    else:
                        reductions = xr_reductions.sel(ModelScenario=np.intersect1d(xr_reductions.ModelScenario, ms2))
                        reds = reductions.Value.quantile(self.dim_nonco2[::-1])
                        for n_i, n in enumerate(self.dim_nonco2):
                            red = reds[n_i]
                            ms2 = reductions.ModelScenario[np.where(np.abs(reductions.Value - red) < 0.1)]
                            trajs = xr_nonco2_raw.sel(ModelScenario = ms2, Time=np.arange(self.settings['params']['start_year_analysis'], 2101))
                            trajectory_mean = rescale(trajs.Value.mean(dim='ModelScenario'))

                            # Harmonize reduction
                            red_traj = (trajectory_mean.sel(Time=2040) - tot_2020) / tot_2020
                            traj2 = -(1-xr_comp)*(red_traj-red)*xr_nonco2_raw_start.mean().Value+trajectory_mean # 1.5*red has been removed -> check effect
                            trajectory_mean2 = check_monotomy(np.array(traj2))
                            times = times + list(np.arange(self.settings['params']['start_year_analysis'], 2101))
                            vals = vals+list(trajectory_mean2)
                            nonco2 = nonco2+[n]*len(list(np.arange(self.settings['params']['start_year_analysis'], 2101)))
                            temps = temps + [temp]*len(list(np.arange(self.settings['params']['start_year_analysis'], 2101)))
                            risks = risks + [p]*len(list(np.arange(self.settings['params']['start_year_analysis'], 2101)))
                            timings = timings + [timing]*len(list(np.arange(self.settings['params']['start_year_analysis'], 2101)))

        dict_nonco2 = {}
        dict_nonco2['Time'] = times
        dict_nonco2['NonCO2red'] = nonco2
        dict_nonco2['NonCO2_globe'] = vals
        dict_nonco2['Temperature'] = temps
        dict_nonco2['Risk'] = risks
        dict_nonco2['Timing'] = timings
        df_nonco2 = pd.DataFrame(dict_nonco2)
        dummy = df_nonco2.set_index(["NonCO2red", "Time", 'Temperature', 'Risk', 'Timing'])
        self.xr_traj_nonco2 = xr.Dataset.from_dataframe(dummy)

        # Post-processing: making temperature dependence smooth
        self.xr_traj_nonco2 = self.xr_traj_nonco2.reindex({'Temperature': [1.5, 1.8, 2.1, 2.4]})
        self.xr_traj_nonco2 = self.xr_traj_nonco2.reindex({'Temperature': self.dim_temp})
        self.xr_traj_nonco2 = self.xr_traj_nonco2.interpolate_na(dim='Temperature')
        self.xr_traj_nonco2_2 = self.xr_traj_nonco2.copy()

        # change time coordinate in self.xr_traj_nonco2 if needed (different starting year than 2021)
        difyears = 2020+1-self.settings['params']['start_year_analysis']
        if difyears > 0:
            self.xr_traj_nonco2_adapt = self.xr_traj_nonco2.assign_coords({'Time': self.xr_traj_nonco2.Time-(difyears-1)}).reindex({'Time': np.arange(self.settings['params']['start_year_analysis'], 2101)})
            for t in np.arange(0, difyears):
                self.xr_traj_nonco2_adapt.NonCO2_globe.loc[{'Time': 2101 - difyears + t}] = (self.xr_traj_nonco2.sel(Time=2101 - difyears + t).NonCO2_globe - self.xr_traj_nonco2.NonCO2_globe.sel(Time=2101 - difyears + t - 1)) + self.xr_traj_nonco2_adapt.NonCO2_globe.sel(Time=2101 - difyears + t - 1)
            fr = (self.xr_traj_nonco2.NonCO2_globe.sum(dim='Time') - self.xr_traj_nonco2_adapt.NonCO2_globe.sum(dim='Time')) * (1-xr_comp)/np.sum(1-xr_comp)
            self.xr_traj_nonco2 = self.xr_traj_nonco2_adapt + fr

    # =========================================================== #
    # =========================================================== #

    def determine_global_budgets(self):
        print('- Get global CO2 budgets')
        # CO2 budgets from Forster
        df_budgets = pd.read_csv("X:/user/dekkerm/Data/Budgets_Forster2023/ClimateIndicator-data-ed37002/data/carbon_budget/update_MAGICC_and_scenarios-budget.csv") # Now without the warming update in Forster, to link to IPCC AR6 
        df_budgets = df_budgets[["dT_targets", "0.1", "0.17", "0.33", "0.5", "0.66", "0.83", '0.9']]
        dummy = df_budgets.melt(id_vars=["dT_targets"], var_name="Probability", value_name="Budget")
        ar = np.array(dummy['Probability'])
        ar = ar.astype(float).round(2)
        ar[ar == 0.66] = 0.67
        dummy['Probability'] = ar
        dummy['dT_targets'] = dummy['dT_targets'].astype(float).round(1)
        dummy = dummy.set_index(["dT_targets", "Probability"])

        # Correct budgets based on startyear (Forster is from Jan 2020 and on)
        if self.settings['params']['start_year_analysis'] == 2020:
            budgets = dummy['Budget']
        elif self.settings['params']['start_year_analysis'] > 2020:
            budgets = dummy['Budget']
            for year in np.arange(2020, self.settings['params']['start_year_analysis']):
                budgets -= float(self.xr_hist.sel(Region='EARTH', Time=year).CO2_hist)/1e3
        elif self.settings['params']['start_year_analysis'] < 2020:
            budgets = dummy['Budget']
            for year in np.arange(self.settings['params']['start_year_analysis'], 2020):
                budgets += float(self.xr_hist.sel(Region='EARTH', Time=year).CO2_hist)/1e3
        dummy['Budget'] = budgets

        xr_bud_co2 = xr.Dataset.from_dataframe(dummy)
        xr_bud_co2 = xr_bud_co2.rename({'dT_targets': "Temperature"})#.sel(Temperature = [1.5, 1.7, 2.0])
        self.xr_bud_co2 = xr_bud_co2

        # Determine bunker emissions to subtract from global budget
        bunker_subtraction = []
        for t_i, t in enumerate(self.dim_temp):
            bunker_subtraction += [3.3/100] # Assuming bunker emissions have a constant fraction of global emissions (3.3%) - https://www.pbl.nl/sites/default/files/downloads/pbl-2020-analysing-international-shipping-and-aviation-emissions-projections_4076.pdf

        Blist = np.zeros(shape=(len(self.dim_temp), len(self.dim_prob), len(self.dim_nonco2)))+np.nan

        def ms_temp(T, R): # 0.2 is quite wide, but useful for studying nonCO2 variation among scenarios (is a relatively metric anyway)
            return self.xr_temperatures.ModelScenario[np.where(np.abs(T-self.xr_temperatures.Temperature.sel(Risk=R)) < 0.2)[0]].values

        for p_i, p in enumerate(self.dim_prob):
            a, b = np.polyfit(xr_bud_co2.Temperature, xr_bud_co2.sel(Probability = np.round(p, 2)).Budget, 1)
            for t_i, t in enumerate(self.dim_temp):
                ms = ms_temp(t, round(1-p, 2))

                # This assumes that the budget from Forster implicitly includes the median nonCO2 warming among scenarios that meet that Temperature target
                # Hence, only deviation (dT) from this median is interesting to assess here
                dT = self.xr_nonco2warming_wrt_start.sel(ModelScenario=ms,
                                                         Risk=round(1-p,2)) - self.xr_nonco2warming_wrt_start.sel(ModelScenario=ms,
                                                                                                                  Risk=round(1-p,2)).median(dim='ModelScenario')
                median_budget = (a*t+b)*(1-bunker_subtraction[t_i])
                for n_i, n in enumerate(self.dim_nonco2):
                    dT_quantile = dT.quantile(n, dim='ModelScenario').PeakWarming # Assuming relation between T and B also holds around the T-value
                    dB_quantile = a*dT_quantile
                    Blist[t_i, p_i, n_i] = median_budget+dB_quantile
        data2 = xr.DataArray(Blist,
                            coords={'Temperature': self.dim_temp,
                                    'Risk': (1-self.dim_prob).astype(float).round(2),
                                    'NonCO2red': self.dim_nonco2},
                            dims=['Temperature', 'Risk', 'NonCO2red'])
        self.xr_co2_budgets = xr.Dataset({'Budget': data2})

    # =========================================================== #
    # =========================================================== #

    def determine_global_co2_trajectories(self):
        print('- Computing global co2 trajectories')        
        # Initialize data arrays for co2
        startpoint = self.xr_hist.sel(Time=self.settings['params']['start_year_analysis'], Region="EARTH").CO2_hist
        #compensation_form = np.array(list(np.linspace(0, 1, len(np.arange(self.settings['params']['start_year_analysis'], 2101)))))#**1.1#+[1]*len(np.arange(2050, 2101)))

        hy = self.settings['params']['harmonization_year']
        if self.settings['params']['start_year_analysis'] >= 2020: 
            compensation_form = np.array(list(np.linspace(0, 1, len(np.arange(self.settings['params']['start_year_analysis'], hy))))+[1]*len(np.arange(hy, 2101)))
            xr_comp =  xr.DataArray(compensation_form, dims=['Time'], coords={'Time': np.arange(self.settings['params']['start_year_analysis'], 2101)})
        if self.settings['params']['start_year_analysis'] < 2020:
            compensation_form = (np.arange(0, 2101-self.settings['params']['start_year_analysis']))**0.5
            #hy = 2100
            #compensation_form = np.array(list(np.linspace(0, 1, len(np.arange(self.settings['params']['start_year_analysis'], hy))))+[1]*len(np.arange(hy, 2101)))
            xr_comp =  xr.DataArray(compensation_form/np.sum(compensation_form), dims=['Time'], coords={'Time': np.arange(self.settings['params']['start_year_analysis'], 2101)})

        def budget_harm(nz):
            return xr_comp / np.sum(xr_comp.sel(Time=np.arange(self.settings['params']['start_year_analysis'], nz)))

        #compensation_form2 = np.array(list(np.linspace(0, 1, len(np.arange(self.settings['params']['start_year_analysis'], 2101)))))**0.5#+[1]*len(np.arange(2050, 2101)))
        xr_traj_co2 = xr.Dataset(
            coords={
                'NegEmis': self.dim_negemis,
                'NonCO2red': self.dim_nonco2,
                'Temperature': self.dim_temp,
                'Risk': self.dim_prob,
                'Timing': self.dim_timing,
                'Time': np.arange(self.settings['params']['start_year_analysis'], 2101),
            }
        )

        xr_traj_co2_neg = xr.Dataset(
            coords={
                'NegEmis': self.dim_negemis,
                'Temperature': self.dim_temp,
                'Time': np.arange(self.settings['params']['start_year_analysis'], 2101),
            }
        )

        pathways_data = {
            'CO2_globe': xr.DataArray(
                data=np.nan,
                coords=xr_traj_co2.coords,
                dims=('NegEmis', "NonCO2red", 'Temperature', 'Risk', 'Timing', 'Time'),
                attrs={'description': 'Pathway data'}
            ),
            'CO2_neg_globe': xr.DataArray(
                data=np.nan,
                coords=xr_traj_co2_neg.coords,
                dims=('NegEmis', 'Temperature', 'Time'),
                attrs={'description': 'Pathway data'}
            )
        }
        # CO2 emissions from AR6
        xr_scen2_use = self.xr_ar6.sel(Variable='Emissions|CO2')
        xr_scen2_use = xr_scen2_use.reindex(Time = np.arange(2000, 2101, 10))
        xr_scen2_use = xr_scen2_use.reindex(Time = np.arange(2000, 2101))
        xr_scen2_use = xr_scen2_use.interpolate_na(dim="Time", method="linear")
        xr_scen2_use = xr_scen2_use.reindex(Time = np.arange(self.settings['params']['start_year_analysis'], 2101))

        co2_start = xr_scen2_use.sel(Time=self.settings['params']['start_year_analysis'])/1e3
        offsets = (startpoint/1e3-co2_start)
        emis_all = xr_scen2_use.sel(Time=np.arange(self.settings['params']['start_year_analysis'], 2101))/1e3 + offsets*(1-xr_comp)
        emis2100 = emis_all.sel(Time=2100)

        # Bend IAM curves to start in the correct starting year (only shape is relevant)
        difyears = 2020+1-self.settings['params']['start_year_analysis']
        if difyears > 0:
            emis_all_adapt = emis_all.assign_coords({'Time': emis_all.Time-(difyears-1)}).reindex({'Time': np.arange(self.settings['params']['start_year_analysis'], 2101)})
            for t in np.arange(0, difyears):
                dv = (emis_all.sel(Time=2101 - difyears + t).Value - emis_all.Value.sel(Time=2101 - difyears + t - 1))
                dv = dv.where(dv < 0, 0)
                emis_all_adapt.Value.loc[{'Time': 2101 - difyears + t}] = dv + emis_all_adapt.Value.sel(Time=2101 - difyears + t - 1)

            fr = (emis_all.Value.sum(dim='Time') - emis_all_adapt.Value.sum(dim='Time')) * (xr_comp)/np.sum(xr_comp)
            emis_all = emis_all_adapt + fr

        # Negative emissions from AR6 (CCS + DAC)
        xr_neg = self.xr_ar6.sel(Variable=['Carbon Sequestration|CCS', 'Carbon Sequestration|Direct Air Capture']).sum(dim='Variable', skipna=False)
        xr_neg = xr_neg.reindex(Time = np.arange(2000, 2101, 10))
        xr_neg = xr_neg.reindex(Time = np.arange(2000, 2101))
        xr_neg = xr_neg.interpolate_na(dim="Time", method="linear")
        xr_neg = xr_neg.reindex(Time = np.arange(self.settings['params']['start_year_analysis'], 2101))

        def remove_upward(ar):
            # Small function to ensure no late-century increase in emissions due to sparse scenario spaces
            ar2 = np.copy(ar)
            ar2[29:] = np.minimum.accumulate(ar[29:])
            return ar2

        # Correction on temperature calibration when using IAM shapes starting at earlier years
        difyear = 2021 - self.settings['params']['start_year_analysis']
        dt = difyear/6*0.1

        def ms_temp_shape(T, R): # Different temperature domain because this is purely for the shape, not for the nonCO2 variation or so
            return self.xr_temperatures.ModelScenario[np.where((self.xr_temperatures.Temperature.sel(Risk=R) < dt+T+0.) & (self.xr_temperatures.Temperature.sel(Risk=R) > dt+T-0.3))[0]].values

        for temp_i, temp in enumerate(self.dim_temp):
            ms1 = ms_temp_shape(temp, 0.5)
            # Shape impacted by timing of action
            for timing_i, timing in enumerate(self.dim_timing):
                if timing == 'Immediate' or temp in [1.5, 1.56, 1.6] and timing == 'Delayed':
                    mslist = self.ms_immediate
                else:
                    mslist = self.ms_delayed
                ms2 = np.intersect1d(ms1, mslist)
                emis2100_i = emis2100.sel(ModelScenario=ms2)

                # The 90-percentile of 2100 emissions
                ms_90 = self.xr_ar6.sel(ModelScenario=ms2).ModelScenario[(emis2100_i >= emis2100_i.quantile(0.9-0.1)
                                            ).Value & (emis2100_i <= emis2100_i.quantile(0.9+0.1)).Value]
            
                # The 50-percentile of 2100 emissions
                ms_10 = self.xr_ar6.sel(ModelScenario=ms2).ModelScenario[(emis2100_i >= emis2100_i.quantile(0.1-0.1)
                                            ).Value & (emis2100_i <= emis2100_i.quantile(0.1+0.1)).Value]

                # Difference and smoothen this
                surplus_factor = emis_all.sel(ModelScenario = np.intersect1d(ms_90, ms2)).mean(dim='ModelScenario').Value - emis_all.sel(ModelScenario = np.intersect1d(ms_10, ms2)).mean(dim='ModelScenario').Value
                surplus_factor2 = np.convolve(surplus_factor, np.ones(3)/3, mode='valid')
                surplus_factor[1:-1] = surplus_factor2

                for neg_i, neg in enumerate(self.dim_negemis):
                    xset = emis_all.sel(ModelScenario=ms2)-surplus_factor*(neg-0.5)
                    pathways_neg = xr_neg.sel(ModelScenario=ms1).quantile(neg, dim='ModelScenario')
                    pathways_data['CO2_neg_globe'][neg_i, temp_i, :] = np.array(pathways_neg.Value)
                    for risk_i, risk in enumerate(self.dim_prob):
                        for nonco2_i, nonco2 in enumerate(self.dim_nonco2):
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

                                try:
                                    nz = self.settings['params']['start_year_analysis']+np.where(pathway_final2 <= 0)[0][0]
                                except:
                                    nz = 2100
                                factor = (self.xr_co2_budgets.Budget.sel(Temperature=temp, Risk=risk, NonCO2red=nonco2)*1e3 - pathway_final2[pathway_final2 > 0].sum())
                                pathway_final2 = (1e3*(pathway_final2+factor*budget_harm(nz)))/1e3
                                
                                pathways_data['CO2_globe'][neg_i, nonco2_i, temp_i, risk_i, timing_i, :] = pathway_final2
        self.xr_traj_co2 = xr_traj_co2.update(pathways_data)
        self.xr_traj_ghg = (self.xr_traj_co2.CO2_globe+self.xr_traj_nonco2.NonCO2_globe).to_dataset(name="GHG_globe")
        # self.xr_traj_ghg = xr.merge([self.xr_traj_ghg_ds.to_dataset(name="GHG_globe"), self.xr_traj_co2.CO2_globe, self.xr_traj_co2.CO2_neg_globe, self.xr_traj_nonco2.NonCO2_globe])
        # x = (self.xr_ar6_landuse / self.xr_ar6.sel(Variable='Emissions|Kyoto Gases')).mean(dim='ModelScenario').Value
        # zero = np.arange(self.settings['params']['start_year_analysis'],2101)[np.where(x.sel(Time=np.arange(self.settings['params']['start_year_analysis'],2101))<0)[0][0]]
        # x0 = x*np.array(list(np.ones(zero-2000))+list(np.zeros(2101-zero)))
        # self.xr_traj_ghg_excl = (self.xr_traj_ghg.GHG_globe*(1-x0)).to_dataset(name='GHG_globe_excl')

        # projected land use emissions
        landuse_ghg = self.xr_ar6_landuse.mean(dim='ModelScenario').GHG_LULUCF
        landuse_co2 = self.xr_ar6_landuse.mean(dim='ModelScenario').CO2_LULUCF

        # historical land use emissions
        landuse_ghg_hist = self.xr_hist.sel(Region='EARTH').GHG_hist - self.xr_hist.sel(Region='EARTH').GHG_hist_excl
        landuse_co2_hist = self.xr_hist.sel(Region='EARTH').CO2_hist - self.xr_hist.sel(Region='EARTH').CO2_hist_excl

        # Harmonize on startyear
        diff_ghg = -landuse_ghg.sel(Time=self.settings['params']['start_year_analysis']) + landuse_ghg_hist.sel(Time=self.settings['params']['start_year_analysis'])
        diff_co2 = -landuse_co2.sel(Time=self.settings['params']['start_year_analysis']) + landuse_co2_hist.sel(Time=self.settings['params']['start_year_analysis'])

        # Corrected
        self.landuse_ghg_corr = landuse_ghg + diff_ghg
        self.landuse_co2_corr = landuse_co2 + diff_co2

        self.xr_traj_ghg_excl = (self.xr_traj_ghg.GHG_globe - self.landuse_ghg_corr).to_dataset(name='GHG_globe_excl')
        self.xr_traj_co2_excl = (self.xr_traj_co2.CO2_globe - self.landuse_co2_corr).to_dataset(name='CO2_globe_excl')
        self.all_projected_gases = xr.merge([self.xr_traj_ghg, self.xr_traj_co2.CO2_globe, self.xr_traj_co2.CO2_neg_globe, self.xr_traj_nonco2.NonCO2_globe, self.xr_traj_ghg_excl.GHG_globe_excl, self.xr_traj_co2_excl.CO2_globe_excl])

    # =========================================================== #
    # =========================================================== #

    def read_baseline(self):
        print('- Reading baseline emissions')
        xr_bases = []
        for i in range(3): # In the up-to-date baselines, only SSP1, 2 and 3 are included. Will be updated at some point.
            df_base = pd.read_excel(self.settings['paths']['data']['baseline']+"SSP"+str(i+1)+".xlsx", sheet_name = 'Sheet1')
            df_base = df_base[df_base['Unnamed: 1'] == 'Emissions|CO2|Energy']
            df_base = df_base.drop(['Unnamed: 1'], axis=1)
            df_base = df_base.rename(columns={"COUNTRY": "Region"})
            df_base['Scenario'] = ['SSP'+str(i+1)]*len(df_base)

            # Melt time index
            df_base = df_base.melt(id_vars=["Region", "Scenario"], var_name="Time", value_name="CO2_base_excl")
            df_base['Time'] = np.array(df_base['Time'].astype(int))

            # Convert to xarray
            dummy = df_base.set_index(["Region", "Scenario", "Time"])
            dummy = dummy.astype(float)
            xr_bases.append(xr.Dataset.from_dataframe(dummy))
        xr_base = xr.merge(xr_bases).reindex({'Region': self.countries_iso})

        # Assign 2020 values in Time index
        xr_base = xr_base.reindex(Time = np.arange(self.settings['params']['start_year_analysis'], 2101))
        for year in np.arange(self.settings['params']['start_year_analysis'], 2021):
            xr_base.CO2_base_excl.loc[dict(Time=year, Region=self.countries_iso)] = self.xr_hist.sel(Time=year, Region=self.countries_iso).CO2_hist_excl
        
        # Harmonize emissions from historical values to baseline emissions
        diffrac = self.xr_hist.CO2_hist_excl.sel(Time=self.settings['params']['start_year_analysis']) / xr_base.CO2_base_excl.sel(Time=self.settings['params']['start_year_analysis'])
        diffrac = diffrac.where(diffrac < 1e9)
        diffrac = diffrac.where(diffrac > -1e9)
        xr_base = xr_base.assign(CO2_base_excl = xr_base.CO2_base_excl * diffrac)

        # Using a fraction, get other emissions variables
        fraction_startyear_co2_incl = self.xr_hist.sel(Time=self.settings['params']['start_year_analysis']).CO2_hist / self.xr_hist.sel(Time=self.settings['params']['start_year_analysis']).CO2_hist_excl
        fraction_startyear_co2_incl = fraction_startyear_co2_incl.where(fraction_startyear_co2_incl < 1e9)
        fraction_startyear_co2_incl = fraction_startyear_co2_incl.where(fraction_startyear_co2_incl > -1e9)
        xr_base = xr_base.assign(CO2_base_incl = xr_base.CO2_base_excl * fraction_startyear_co2_incl)

        fraction_startyear_ghg_excl = self.xr_hist.sel(Time=self.settings['params']['start_year_analysis']).GHG_hist_excl / self.xr_hist.sel(Time=self.settings['params']['start_year_analysis']).CO2_hist_excl
        fraction_startyear_ghg_excl = fraction_startyear_ghg_excl.where(fraction_startyear_ghg_excl < 1e9)
        fraction_startyear_ghg_excl = fraction_startyear_ghg_excl.where(fraction_startyear_ghg_excl > -1e9)
        xr_base = xr_base.assign(GHG_base_excl = xr_base.CO2_base_excl * fraction_startyear_ghg_excl)

        fraction_startyear_ghg_incl = self.xr_hist.sel(Time=self.settings['params']['start_year_analysis']).GHG_hist / self.xr_hist.sel(Time=self.settings['params']['start_year_analysis']).CO2_hist_excl
        fraction_startyear_ghg_incl = fraction_startyear_ghg_incl.where(fraction_startyear_ghg_incl < 1e9)
        fraction_startyear_ghg_incl = fraction_startyear_ghg_incl.where(fraction_startyear_ghg_incl > -1e9)
        xr_base = xr_base.assign(GHG_base_incl = xr_base.CO2_base_excl * fraction_startyear_ghg_incl)

        # Assign 2020 values in Time index
        xr_base = xr_base.reindex(Time = np.arange(self.settings['params']['start_year_analysis'], 2101))
        for year in np.arange(self.settings['params']['start_year_analysis'], 2021):
            xr_base.GHG_base_excl.loc[dict(Time=year, Region=self.countries_iso)] = self.xr_hist.sel(Time=year, Region=self.countries_iso).GHG_hist_excl
            xr_base.CO2_base_incl.loc[dict(Time=year, Region=self.countries_iso)] = self.xr_hist.sel(Time=year, Region=self.countries_iso).CO2_hist
            xr_base.GHG_base_incl.loc[dict(Time=year, Region=self.countries_iso)] = self.xr_hist.sel(Time=year, Region=self.countries_iso).GHG_hist

        # Harmonize global baseline emissions with sum of all countries (this is important for consistency of AP, etc.)
        base_onlyc = xr_base.reindex(Region=self.countries_iso)
        base_w = base_onlyc.sum(dim='Region').expand_dims({'Region': ['EARTH']})
        self.xr_base = xr.merge([base_w,base_onlyc])

    # =========================================================== #
    # =========================================================== #

    def read_ndc_climateresource(self):
        print('- Reading NDC data from Climate resource')        
        ghg_data = np.zeros(shape=(len(self.countries_iso)+1, 3, 2, 2, len(np.arange(2010, 2051))))
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
        # Now also for EU
        for cond_i, cond in enumerate(['conditional', 'range', 'unconditional']):
            for hot_i, hot in enumerate(['include', 'exclude']):
                for amb_i, amb in enumerate(['low', 'high']):
                    params = self.settings['params']
                    path = f'X:/user/dekkerm/Data/NDC/ClimateResource_{params["version_ndcs"]}/{cond}/{hot}/regions/groupeu27_ndc_{params["version_ndcs"]}_CR_{cond}_{hot}.json'
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
                                ghg_data[cty_i+1, cond_i, hot_i, amb_i] = ghg_values
                                #series.append([country_iso.upper(), country_name, "Emissions|Total GHG excl. LULUCF", conditionality, hot_air, ambition] + list(ghg_values))
                    except:
                        continue
        coords = {
            'Region': list(self.countries_iso)+['EU'],
            'Conditionality': ['conditional', 'range', 'unconditional'],
            'Hot_air': ['include', 'exclude'],
            'Ambition': ['min', 'max'],
            'Time': np.array(time_values)[np.array(time_values)>=2010],
        }
        data_vars = {
            'GHG_ndc_excl_CR': (['Region', 'Conditionality', 'Hot_air', 'Ambition', 'Time'], ghg_data),
        }
        xr_ndc = xr.Dataset(data_vars, coords=coords)
        self.xr_ndc_CR = xr_ndc.sel(Time=2030)

    # =========================================================== #
    # =========================================================== #

    def read_ndc(self):
        print('- Reading NDC data')
        df_ndc_raw = pd.read_excel(self.settings['paths']['data']['external']+ "NDC/Infographics PBL NDC Tool 4Oct2024_for CarbonBudgetExplorer.xlsx",
                                   sheet_name='Reduction All_GHG_incl', header=[0, 1])
        regs = df_ndc_raw['(Mt CO2 equivalent)']['Country name']
        regs_iso = []
        for r in regs:
            wh = np.where(self.countries_name == r)[0]
            if len(wh) == 0:
                if r == 'United States':
                    regs_iso.append('USA')
                elif r == 'EU27':
                    regs_iso.append('EU')
                elif r == 'Turkey':
                    regs_iso.append('TUR')
                else:
                    regs_iso.append(np.nan)
            else:
                regs_iso.append(self.countries_iso[wh[0]])
        regs_iso = np.array(regs_iso)
        df_ndc_raw['ISO'] = regs_iso

        df_regs = []
        df_amb = []
        df_con = []
        df_emis = []
        df_lulucf = []
        df_red = []
        df_abs = []
        df_inv = []
        histemis = self.xr_hist.GHG_hist.sel(Time=2015)
        for r in list(self.countries_iso) + ['EU']:
            histemis_r = float(histemis.sel(Region=r))
            df_ndc_raw_sub = df_ndc_raw[df_ndc_raw['ISO'] == r]
            if len(df_ndc_raw_sub) > 0:
                val_2015 = float(df_ndc_raw_sub['(Mt CO2 equivalent)'][2015])
                for lulucf in ['incl']: # Maybe add excl later?
                    for emis_i, emis in enumerate(['NDC']):# , 'CP']): 
                        key = ['2030 NDCs', 'Domestic actions 2030'][emis_i]
                        for cond_i, cond in enumerate(['unconditional', 'conditional']):
                            condkey = ['Unconditional NDCs', 'Conditional NDCs'][cond_i]
                            for ambition_i, ambition in enumerate(['min', 'max']):
                                add = ['', '.1'][ambition_i]
                                val = float(df_ndc_raw_sub[key][condkey+add])
                                red = 1- val/val_2015
                                abs_jones = histemis_r*(1-red)
                                df_regs.append(r)
                                df_amb.append(ambition)
                                df_con.append(cond)
                                df_emis.append(emis)
                                df_lulucf.append(lulucf)
                                df_red.append(red)
                                df_abs.append(abs_jones)
                                df_inv.append(val)

        dict_ndc = {"Region": df_regs,
                    "Ambition": df_amb,
                    "Conditionality": df_con,
                    "GHG_ndc_red": df_red,
                    "GHG_ndc": df_abs,
                    "GHG_ndc_inv": df_inv}
        df_ndc = pd.DataFrame(dict_ndc)
        self.xr_ndc = xr.Dataset.from_dataframe(df_ndc.set_index(["Region", "Ambition", "Conditionality"]))

        # Now for GHG excluding LULUCF
        df_ndc_raw = pd.read_excel(self.settings['paths']['data']['external']+ "NDC/Infographics PBL NDC Tool 4Oct2024_for CarbonBudgetExplorer.xlsx", sheet_name='Reduction All_GHG_excl', header=[0, 1])
        regs = df_ndc_raw['(Mt CO2 equivalent)']['Country name']
        regs_iso = []
        for r in regs:
            wh = np.where(self.countries_name == r)[0]
            if len(wh) == 0:
                if r == 'United States':
                    regs_iso.append('USA')
                elif r == 'EU27':
                    regs_iso.append('EU')
                elif r == 'Turkey':
                    regs_iso.append('TUR')
                else:
                    regs_iso.append(np.nan)
            else:
                regs_iso.append(self.countries_iso[wh[0]])
        regs_iso = np.array(regs_iso)
        df_ndc_raw['ISO'] = regs_iso

        df_regs = []
        df_amb = []
        df_con = []
        df_emis = []
        df_lulucf = []
        df_red = []
        df_abs = []
        df_inv = []
        histemis = self.xr_hist.GHG_hist.sel(Time=2015)
        for r in list(self.countries_iso) + ['EU']:
            histemis_r = float(histemis.sel(Region=r))
            df_ndc_raw_sub = df_ndc_raw[df_ndc_raw['ISO'] == r]
            if len(df_ndc_raw_sub) > 0:
                val_2015 = float(df_ndc_raw_sub['(Mt CO2 equivalent)'][2015])
                for lulucf in ['incl']: # Maybe add excl later?
                    for emis_i, emis in enumerate(['NDC']):# , 'CP']): 
                        key = ['2030 NDCs', 'Domestic actions 2030'][emis_i]
                        for cond_i, cond in enumerate(['unconditional', 'conditional']):
                            condkey = ['Unconditional NDCs', 'Conditional NDCs'][cond_i]
                            for ambition_i, ambition in enumerate(['min', 'max']):
                                add = ['', '.1'][ambition_i]
                                val = float(df_ndc_raw_sub[key][condkey+add])
                                red = 1- val/val_2015
                                abs_jones = histemis_r*(1-red)
                                df_regs.append(r)
                                df_amb.append(ambition)
                                df_con.append(cond)
                                df_emis.append(emis)
                                df_lulucf.append(lulucf)
                                df_red.append(red)
                                df_abs.append(abs_jones)
                                df_inv.append(val)

        dict_ndc = {"Region": df_regs,
                    "Ambition": df_amb,
                    "Conditionality": df_con,
                    "GHG_ndc_excl_red": df_red,
                    "GHG_ndc_excl": df_abs,
                    "GHG_ndc_excl_inv": df_inv}
        df_ndc = pd.DataFrame(dict_ndc)
        self.xr_ndc_excl = xr.Dataset.from_dataframe(df_ndc.set_index(["Region", "Ambition", "Conditionality"]))

    # =========================================================== #
    # =========================================================== #

    def merge_xr(self):
        print('- Merging xrarray object')
        xr_total = xr.merge([self.xr_ssp,
                             self.xr_hist,
                             self.xr_unp,
                             self.xr_hdish,
                             self.xr_co2_budgets,
                             self.all_projected_gases,
                             self.xr_base,
                             self.xr_ndc,
                             self.xr_ndc_excl,
                             self.xr_ndc_CR,
                             self.xr_ar6_C,
                             self.xr_ar6_C_bunkers])
        xr_total = xr_total.reindex(Region = self.regions_iso)
        xr_total = xr_total.reindex(Time = np.arange(1850, 2101))
        xr_total['GHG_globe'] = xr_total['GHG_globe'].astype(float)
        self.xr_total = xr_total.interpolate_na(dim="Time", method="linear").drop_vars(['Variable'])

    # =========================================================== #
    # =========================================================== #

    def add_country_groups(self):
        print('- Add country groups')
        path_ctygroups = "X:/user/dekkerm/Data/" + "UNFCCC_Parties_Groups_noeu.xlsx"
        df = pd.read_excel(path_ctygroups, sheet_name = "Country groups")
        countries_iso = np.array(df["Country ISO Code"])
        list_of_regions = list(np.array(self.regions_iso).copy())
        reg_iso = list(self.regions_iso)
        reg_name = list(self.regions_name)
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
                xr_eu = self.xr_total[['Population', 'GDP', 'GHG_hist', "GHG_base_incl", "CO2_hist", "CO2_base_incl", "GHG_hist_excl", "GHG_base_excl", "CO2_hist_excl", "CO2_base_excl"]].groupby(group_coord).sum()#skipna=False)
            else:
                xr_eu = self.xr_total[['Population', 'GDP', 'GHG_hist', "GHG_base_incl", "CO2_hist", "CO2_base_incl", "GHG_hist_excl", "GHG_base_excl", "CO2_hist_excl", "CO2_base_excl", "GHG_ndc", "GHG_ndc_inv", "GHG_ndc_excl", "GHG_ndc_excl_inv", "GHG_ndc_excl_CR"]].groupby(group_coord).sum(skipna=False)
            xr_eu2 = xr_eu.rename({'group': "Region"})
            dummy = self.xr_total.reindex(Region = list_of_regions)
            self.xr_total = xr.merge([dummy, xr_eu2])
            self.xr_total = self.xr_total.reindex(Region = list_of_regions)
            if group_of_choice not in ['EU', 'EARTH']:
                reg_iso.append(group_of_choice)
                reg_name.append(group_of_choice)
        self.xr_total = self.xr_total
        self.xr_total['GHG_base_incl'][np.where(self.xr_total.Region=='EU')[0], np.array([3, 4])] = np.nan # SSP4, 5 are empty for Europe!
        self.xr_total['CO2_base_incl'][np.where(self.xr_total.Region=='EU')[0], np.array([3, 4])] = np.nan # SSP4, 5 are empty for Europe!
        self.xr_total['GHG_base_excl'][np.where(self.xr_total.Region=='EU')[0], np.array([3, 4])] = np.nan # SSP4, 5 are empty for Europe!
        self.xr_total['CO2_base_excl'][np.where(self.xr_total.Region=='EU')[0], np.array([3, 4])] = np.nan # SSP4, 5 are empty for Europe!
        self.regions_iso = np.array(reg_iso)
        self.regions_name = np.array(reg_name)

    # =========================================================== #
    # =========================================================== #

    def save(self): 
        print('- Save important files')

        xr_normal = self.xr_total.sel(Temperature=np.array(self.settings['dimension_ranges']['peak_temperature_saved']).astype(float).round(2))
        xr_version = xr_normal

        np.save(self.settings['paths']['data']['datadrive'] + "all_regions.npy", self.regions_iso)
        np.save(self.settings['paths']['data']['datadrive'] + "all_regions_names.npy", self.regions_name)
        np.save(self.settings['paths']['data']['datadrive'] + "all_countries.npy", self.countries_iso)
        np.save(self.settings['paths']['data']['datadrive'] + "all_countries_names.npy", self.countries_name)

        xr_version.to_netcdf(self.savepath+'xr_dataread.nc',
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
                "Ambition": {"dtype": "str"},

                "GDP": {"zlib": True, "complevel": 9},
                "Population": {"zlib": True, "complevel": 9},

                "GHG_hist": {"zlib": True, "complevel": 9},
                "GHG_hist_excl": {"zlib": True, "complevel": 9},
                "CO2_hist": {"zlib": True, "complevel": 9},
                "CO2_hist_excl": {"zlib": True, "complevel": 9},

                "GHG_globe": {"zlib": True, "complevel": 9},
                "GHG_globe_excl": {"zlib": True, "complevel": 9},
                "CO2_globe": {"zlib": True, "complevel": 9},
                "CO2_globe_excl": {"zlib": True, "complevel": 9},

                "GHG_base_incl": {"zlib": True, "complevel": 9},
                "GHG_base_excl": {"zlib": True, "complevel": 9},
                "CO2_base_incl": {"zlib": True, "complevel": 9},
                "CO2_base_excl": {"zlib": True, "complevel": 9},

                "GHG_excl_C": {"zlib": True, "complevel": 9},
                "CO2_excl_C": {"zlib": True, "complevel": 9},
                "CO2_neg_C": {"zlib": True, "complevel": 9},
                "CO2_bunkers_C": {"zlib": True, "complevel": 9},

                "GHG_ndc": {"zlib": True, "complevel": 9},
                "GHG_ndc_excl": {"zlib": True, "complevel": 9},                      
                "GHG_ndc_excl_CR": {"zlib": True, "complevel": 9},
            },
            format="NETCDF4",
            engine="netcdf4",
        )

        # AP rbw factors
        for gas in ['CO2', 'GHG']:
            for lulucf_i, lulucf in enumerate(['incl', 'excl']):
                luext = ['', '_excl'][lulucf_i]
                xrt = xr_version.sel(Time=np.arange(self.settings['params']['start_year_analysis'], 2101))
                r1_nom = (xrt.GDP.sel(Region='EARTH') / xrt.Population.sel(Region='EARTH'))
                base_worldsum = xrt[gas+'_base_'+lulucf].sel(Region='EARTH')
                rb_part1 = (xrt.GDP / xrt.Population / r1_nom)**(1/3.)
                rb_part2 = xrt[gas+'_base_'+lulucf]*(base_worldsum - xrt[gas+'_globe'+luext])/base_worldsum
                rbw = (rb_part1*rb_part2).sel(Region=self.countries_iso).sum(dim='Region')
                rbw = rbw.where(rbw != 0)
                rbw.to_netcdf(self.savepath+'xr_rbw_'+gas+'_'+lulucf+'.nc')

        # GDR RCI indices
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
        xr_rci = xr_rci.reindex({"Region": xr_version.Region})
        xr_rci.to_netcdf(self.settings['paths']['data']['datadrive'] +'xr_rci.nc')

    # =========================================================== #
    # =========================================================== #

    def country_specific_datareaders(self):
        # Dutch emissions - harmonized with the KEV # TODO harmonize global emissions with this, as well.
        xr_dataread_nld =  xr.open_dataset(self.savepath + 'xr_dataread.nc').load().copy()
        dutch_time = np.array([1990, 1995, 2000, 2005, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021])
        dutch_ghg = np.array([228.9, 238.0, 225.7, 220.9, 219.8, 206, 202, 201.2, 192.9, 199.8, 200.2, 196.5, 191.4, 185.6, 168.9, 172.0])
        dutch_time_interp = np.arange(1990, self.settings['params']['start_year_analysis']+1)
        dutch_ghg_interp = np.interp(dutch_time_interp, dutch_time, dutch_ghg)
        fraction_1990 = float(dutch_ghg[0] / self.xr_total.GHG_hist.sel(Region='NLD', Time=1990))
        pre_1990_raw = np.array(self.xr_total.GHG_hist.sel(Region='NLD', Time=np.arange(1850, 1990)))*fraction_1990
        total_ghg_nld = np.array(list(pre_1990_raw) + list(dutch_ghg_interp))
        fractions = np.array(xr_dataread_nld.GHG_hist.sel(Region='NLD', Time=np.arange(1850, self.settings['params']['start_year_analysis']+1)) / total_ghg_nld)
        for t_i, t in enumerate(self.time_past):
            xr_dataread_nld.GHG_hist.loc[dict(Time=t, Region='NLD')] = total_ghg_nld[t_i]
            
        xr_dataread_nld.CO2_base_incl.loc[dict(Region='NLD', Time=self.time_future)] = xr_dataread_nld.CO2_base_incl.sel(Region='NLD', Time=self.time_future)/fractions[-1]
        xr_dataread_nld.CO2_base_excl.loc[dict(Region='NLD', Time=self.time_future)] = xr_dataread_nld.CO2_base_excl.sel(Region='NLD', Time=self.time_future)/fractions[-1]
        xr_dataread_nld.GHG_base_incl.loc[dict(Region='NLD', Time=self.time_future)] = xr_dataread_nld.GHG_base_incl.sel(Region='NLD', Time=self.time_future)/fractions[-1]
        xr_dataread_nld.GHG_base_excl.loc[dict(Region='NLD', Time=self.time_future)] = xr_dataread_nld.GHG_base_excl.sel(Region='NLD', Time=self.time_future)/fractions[-1]

        xr_dataread_nld.CO2_hist.loc[dict(Region='NLD', Time=self.time_past)] = xr_dataread_nld.CO2_hist.sel(Region='NLD', Time=self.time_past)/fractions
        xr_dataread_nld.CO2_hist_excl.loc[dict(Region='NLD', Time=self.time_past)] = xr_dataread_nld.CO2_hist_excl.sel(Region='NLD', Time=self.time_past)/fractions
        xr_dataread_nld.GHG_hist_excl.loc[dict(Region='NLD', Time=self.time_past)] = xr_dataread_nld.GHG_hist_excl.sel(Region='NLD', Time=self.time_past)/fractions
        xr_dataread_nld.sel(Temperature=np.array(self.settings['dimension_ranges']['peak_temperature_saved']).astype(float).round(2)).to_netcdf(self.savepath+'xr_dataread_NLD.nc',
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
                        "Ambition": {"dtype": "str"},

                        "GDP": {"zlib": True, "complevel": 9},
                        "Population": {"zlib": True, "complevel": 9},

                        "GHG_hist": {"zlib": True, "complevel": 9},
                        "GHG_hist_excl": {"zlib": True, "complevel": 9},
                        "CO2_hist": {"zlib": True, "complevel": 9},
                        "CO2_hist_excl": {"zlib": True, "complevel": 9},

                        "GHG_globe": {"zlib": True, "complevel": 9},
                        "GHG_globe_excl": {"zlib": True, "complevel": 9},
                        "CO2_globe": {"zlib": True, "complevel": 9},
                        "CO2_globe_excl": {"zlib": True, "complevel": 9},

                        "GHG_base_incl": {"zlib": True, "complevel": 9},
                        "GHG_base_excl": {"zlib": True, "complevel": 9},
                        "CO2_base_incl": {"zlib": True, "complevel": 9},
                        "CO2_base_excl": {"zlib": True, "complevel": 9},

                        "GHG_excl_C": {"zlib": True, "complevel": 9},
                        "CO2_excl_C": {"zlib": True, "complevel": 9},
                        "CO2_neg_C": {"zlib": True, "complevel": 9},

                        "GHG_ndc": {"zlib": True, "complevel": 9},
                        "GHG_ndc_inv": {"zlib": True, "complevel": 9},
                        "GHG_ndc_red": {"zlib": True, "complevel": 9},

                        "GHG_ndc_excl": {"zlib": True, "complevel": 9},
                        "GHG_ndc_excl_inv": {"zlib": True, "complevel": 9},
                        "GHG_ndc_excl_red": {"zlib": True, "complevel": 9},                        
                        "GHG_ndc_excl_CR": {"zlib": True, "complevel": 9},
                        },
                        format="NETCDF4",
                        engine="netcdf4",
                    )
        
        # Norwegian emissions - harmonized with EDGAR
        xr_dataread_nor =  xr.open_dataset(self.savepath + 'xr_dataread.nc').load().copy()
        # Get data and interpolate
        time_axis = np.arange(1990, self.settings['params']['start_year_analysis']+1)
        ghg_axis = np.array(self.xr_primap.sel(Scenario='HISTCR', Region='NOR', time=time_axis, Category='M.0.EL')['KYOTOGHG (AR6GWP100)'])
        time_interp = np.arange(np.min(time_axis), np.max(time_axis)+1)
        ghg_interp = np.interp(time_interp, time_axis, ghg_axis)

        # Get older data by linking to Jones
        fraction_minyear = float(ghg_axis[0] / self.xr_total.GHG_hist_excl.sel(Region='NOR', Time=np.min(time_axis)))
        pre_minyear_raw = np.array(self.xr_total.GHG_hist_excl.sel(Region='NOR', Time=np.arange(1850, np.min(time_axis))))*fraction_minyear
        total_ghg_nor = np.array(list(pre_minyear_raw) + list(ghg_interp))/1e3
        fractions = np.array(xr_dataread_nor.GHG_hist_excl.sel(Region='NOR', Time=self.time_past) / total_ghg_nor)
        for t_i, t in enumerate(self.time_past):
            xr_dataread_nor.GHG_hist_excl.loc[dict(Time=t, Region='NOR')] = total_ghg_nor[t_i]
            
        xr_dataread_nor.CO2_base_incl.loc[dict(Region='NOR', Time=self.time_future)] = xr_dataread_nor.CO2_base_incl.sel(Region='NOR', Time=self.time_future)/fractions[-1]
        xr_dataread_nor.CO2_base_excl.loc[dict(Region='NOR', Time=self.time_future)] = xr_dataread_nor.CO2_base_excl.sel(Region='NOR', Time=self.time_future)/fractions[-1]
        xr_dataread_nor.GHG_base_incl.loc[dict(Region='NOR', Time=self.time_future)] = xr_dataread_nor.GHG_base_incl.sel(Region='NOR', Time=self.time_future)/fractions[-1]
        xr_dataread_nor.GHG_base_excl.loc[dict(Region='NOR', Time=self.time_future)] = xr_dataread_nor.GHG_base_excl.sel(Region='NOR', Time=self.time_future)/fractions[-1]

        xr_dataread_nor.CO2_hist.loc[dict(Region='NOR', Time=self.time_past)] = xr_dataread_nor.CO2_hist.sel(Region='NOR', Time=self.time_past)/fractions
        xr_dataread_nor.CO2_hist_excl.loc[dict(Region='NOR', Time=self.time_past)] = xr_dataread_nor.CO2_hist_excl.sel(Region='NOR', Time=self.time_past)/fractions
        xr_dataread_nor.GHG_hist.loc[dict(Region='NOR', Time=self.time_past)] = xr_dataread_nor.GHG_hist.sel(Region='NOR', Time=self.time_past)/fractions
        xr_dataread_nor.sel(Temperature=np.array(self.settings['dimension_ranges']['peak_temperature_saved']).astype(float).round(2)).to_netcdf(self.savepath+'xr_dataread_NOR.nc',
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
                        "Ambition": {"dtype": "str"},

                        "GDP": {"zlib": True, "complevel": 9},
                        "Population": {"zlib": True, "complevel": 9},

                        "GHG_hist": {"zlib": True, "complevel": 9},
                        "GHG_hist_excl": {"zlib": True, "complevel": 9},
                        "CO2_hist": {"zlib": True, "complevel": 9},
                        "CO2_hist_excl": {"zlib": True, "complevel": 9},

                        "GHG_globe": {"zlib": True, "complevel": 9},
                        "GHG_globe_excl": {"zlib": True, "complevel": 9},
                        "CO2_globe": {"zlib": True, "complevel": 9},
                        "CO2_globe_excl": {"zlib": True, "complevel": 9},

                        "GHG_base_incl": {"zlib": True, "complevel": 9},
                        "GHG_base_excl": {"zlib": True, "complevel": 9},
                        "CO2_base_incl": {"zlib": True, "complevel": 9},
                        "CO2_base_excl": {"zlib": True, "complevel": 9},

                        "GHG_ndc": {"zlib": True, "complevel": 9},
                        "GHG_ndc_excl": {"zlib": True, "complevel": 9},                      
                        "GHG_ndc_excl_CR": {"zlib": True, "complevel": 9},
                        },
                        format="NETCDF4",
                        engine="netcdf4",
                    )