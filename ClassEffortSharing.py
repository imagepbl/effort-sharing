# ======================================== #
# Class for the effort sharing work of ECEMF WP5
# ======================================== #

# =========================================================== #
# PREAMBULE
# Put in packages that we need
# =========================================================== #

from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import xarray as xr
import yaml
import json
import os
import plotly.express as px
from Functions import (gdp_future, pop_func, emis_func, determine_coefficient,
                       popshare_func, emisshare_func, emis_total_func, rho, cumpopshare_func,
                       cumfuturepop_func, emis_f_func, create_groups, gdp_future_reread, grouping, grouping_xarray)

# =========================================================== #
# CLASS OBJECT
# =========================================================== #

class shareefforts(object):

    # =========================================================== #
    # =========================================================== #

    def __init__(self):
        print("# ==================================== #")
        print("# Initializing shareefforts class    #")
        print("# ==================================== #")

        self.current_dir = Path("X:/user/dekkerm/Projects/ECEMF_T5.2")#.cwd()

        # Read in Input YAML file
        with open(self.current_dir / 'input.yml') as file:
            diction = yaml.load(file, Loader=yaml.FullLoader)
        self.reread = diction['reread']

        # Choices
        self.hist_emission_scen = diction['hist_emission_scen']
        self.version_ndcs = diction['version_ndcs']
        self.conditionality_ndcs = diction['conditionality_ndcs']
        self.hotair_ndcs = diction['hotair_ndcs']
        self.gdp_type_scenarios = diction['gdp_type_scenarios']

        # Parameters
        self.gwp_ch4 = int(diction['gwp_ch4'])
        self.gwp_n2o = int(diction['gwp_n2o'])
        self.discount_factor = float(diction['discount_factor'])
        self.historical_emissions_startyear = int(diction['historical_emissions_startyear'])
        self.convergence_moment =int(diction['convergence_moment'])
        self.convergence_year_gdr = int(diction['convergence_year_gdr'])
        self.start_year = int(diction['start_year'])

        # Other stuff
        self.all_categories = np.array(["C1", "C2", "C3", 'C4', 'C5', 'C6', 'C7', 'C8', 'C1+C2', 'C3+C4', 'C5+C6', 'C7+C8'])
        # self.eumemberstates = ["AUT", "BEL", "BGR", "HRV", "CYP", "CZE", "DNK", "EST", "FIN", "FRA",
        #                        "DEU", "GRC", "HUN", "IRL", "ITA", "LVA", "LTU", "LUX", "MLT", "NLD",
        #                        "POL", "PRT", "ROU", "SVK", "SVN", "ESP", "SWE"]
        self.ar6_vars_to_keep = ['Emissions|CO2|Energy and Industrial Processes',
                                'Emissions|CO2',
                                 "Emissions|Kyoto Gases",
                                 "Emissions|CO2|AFOLU|Land",
                                 "Emissions|CH4|AFOLU|Land", 
                                 "Emissions|N2O|AFOLU|Land",
                                 'Carbon Sequestration|CCS',
                                 'Carbon Sequestration|Direct Air Capture',
                                 'Emissions|CO2|Energy|Demand|Transportation|Aviation',
                                 'Emissions|CO2|Energy|Demand|Transportation|Maritime',
                                 'Population',
                                 self.gdp_type_scenarios]
        self.ndc_var = "Emissions|Total GHG excl. LULUCF"
        self.ndcvars_malte = ["LOW NDC Covered + Non-Covered Emissions, excl. LULUCF",
                              "HIGH NDC Covered + Non-Covered Emissions, excl. LULUCF"]

    # =========================================================== #
    # =========================================================== #

    def define_paths(self):
        """ Defining the paths used further in the class """
        print('- Defining paths')

        self.path_current = Path.cwd()
        self.path_main = Path("X:/user/dekkerm/Projects/ECEMF_T5.2/")
        self.path_data = Path("X:/user/dekkerm/Data/")
        self.path_repo_data = self.path_main / "Data"
        self.path_repo_figs = self.path_main / "Figures"
        self.path_ar6_meta_iso3 = self.path_data / "IPCC" / "AR6_ISO3" / "AR6_Scenarios_Database_metadata_indicators_v1.1.xlsx"
        self.path_ar6_data_iso3 = self.path_data / "IPCC" / "AR6_ISO3" / "AR6_Scenarios_Database_ISO3_v1.1.csv"
        self.path_ar6_meta_w = self.path_data / "IPCC" / "AR6_Scenarios_Database_metadata_indicators_v1.1.xlsx"
        self.path_ar6_data_w = self.path_data / "IPCC" / "AR6_Scenarios_Database_World_v1.1.csv"
        self.path_primap = self.path_data / "PRIMAP" / "Guetschow-et-al-2023a-PRIMAP-hist_v2.4.2_final_no_rounding_09-Mar-2023.csv"
        self.path_ndc = self.path_data / "NDC"
        self.path_edgar = self.path_data / "EDGAR" / "essd_ghg_data_gwp100.xlsx"
        self.path_weo = self.path_data / "WEO" / "WEOApr2022all.csv"
        self.path_weo_gr = self.path_data / "WEO" / "WEOApr2022alla.csv"
        self.path_unp = self.path_data / "UN Population" / "WPP2022_GEN_F01_DEMOGRAPHIC_INDICATORS.xlsx"
        self.path_oecd_gdp = Path("X:/IMAGE/data/raw/GDP")
        self.path_ssp_update = self.path_data / "SSPs" / "scenarios_for_navigate5.xlsx"
        self.path_hdi = self.path_data / "HDI" / "HDR21-22_Statistical_Annex_HDI_Table.xlsx"
        self.path_ctygroups = self.path_data / "UNFCCC_Parties_Groups_noeu.xlsx"
        self.path_ndc_cr = "X:/user/dekkerm/Data/NDC/ClimateResourceCalculations.csv"

    # =========================================================== #
    # =========================================================== #

    def read_countrygroups(self):
        print('- Reading country groups')
        # df_regions = pd.read_excel(Path("X:/user/dekkerm/Data/") / "AR6_regionclasses.xlsx")
        # df_regions = df_regions.sort_values(by=['name'])
        # df_regions = df_regions.sort_index()
        # self.countries_iso = np.array(df_regions.ISO)
        # self.countries_name = np.array(df_regions.name)

        df = pd.read_excel(self.path_ctygroups, sheet_name = "Country groups")
        self.countries_iso = np.array(df["Country ISO Code"])
        self.countries_name = np.array(df["Name"])
        countries_iso = np.array(df["Country ISO Code"])
        self.group_cvf = countries_iso[np.array(df["CVF (24/10/22)"]) == 1]
        self.group_g20 = countries_iso[np.array(df["G20"]) == 1]
        self.group_eu = countries_iso[np.array(df["EU"]) == 1]
        self.group_g7 = countries_iso[np.array(df["G7"]) == 1]
        self.group_na = countries_iso[np.array(df["Northern America"]) == 1]
        self.group_um = countries_iso[np.array(df["Umbrella"]) == 1]
        self.group_au = countries_iso[np.array(df["Australasia"]) == 1]
        self.group_af = countries_iso[np.array(df["African Group"]) == 1]
        self.group_sids = countries_iso[np.array(df["SIDS"]) == 1]
        self.group_ldc = countries_iso[np.array(df["LDC"]) == 1]
        self.group_eg = countries_iso[np.array(df["European Group"]) == 1]
        self.group_world = np.copy(self.countries_iso)
        self.groups_ctys = [self.group_cvf, self.group_g20, self.group_eu, self.group_g7, self.group_na, self.group_au,
                            self.group_af, self.group_sids, self.group_ldc, self.group_um, self.group_eg, self.group_world]
        self.groups_iso = ['CVF', 'G20', "EU", "G7", "NA", "AU", "AF", "SIDS", "LDC", "UM", "EG", "WORLD"]
        self.all_regions_iso = np.array(list(self.countries_iso)+['CVF', 'G20', "EU", "G7", "NA", "AU", "AF", "SIDS", "LDC", "UM", "EG", "WORLD"])
        self.all_regions_names = np.array(list(self.countries_name)+['Climate Vulnerable Forum',
                                                                    'Group of 20',
                                                                    "European Union",
                                                                    "Group of 7",
                                                                    "Northern America",
                                                                    "Australasia",
                                                                    "African Group",
                                                                    "Small Island Developing States",
                                                                    "Least Developed Countries",
                                                                    "Umbrella Group",
                                                                    "European Group",
                                                                    "World"])

    # =========================================================== #
    # =========================================================== #

    def read_unpop(self):
        print('- Read UN population data')
        df_unp = pd.read_excel(self.path_unp, sheet_name="Estimates", header=16)
        df_unp = df_unp[["Region, subregion, country or area *", "ISO3 Alpha-code", "Total Population, as of 1 January (thousands)", "Year"]]
        df_unp = df_unp.rename(columns={"Region, subregion, country or area *": "Region", "ISO3 Alpha-code": "ISO", "Total Population, as of 1 January (thousands)": "Population", "Year": "Time"})
        df_unp = df_unp[df_unp['Time'] >= self.historical_emissions_startyear]
        df_unp_f = pd.read_excel(self.path_unp, sheet_name="Medium variant", header=16)
        df_unp_f = df_unp_f[["Region, subregion, country or area *", "ISO3 Alpha-code", "Total Population, as of 1 January (thousands)", "Year"]]
        df_unp_f = df_unp_f.rename(columns={"Region, subregion, country or area *": "Region", "ISO3 Alpha-code": "ISO", "Total Population, as of 1 January (thousands)": "Population", "Year": "Time"})
        df_unp_tot = pd.concat([df_unp, df_unp_f]).reset_index(drop=True)
        vals = np.array(df_unp_tot.Population).astype(str)
        vals[vals == '...'] = 'nan'
        vals = vals.astype(float)
        vals = vals*1e3
        df_unp_tot['Population'] = vals
        df_unp_tot = df_unp_tot.rename(columns={'Region': "Name"})
        df_unp_tot = df_unp_tot[df_unp_tot.ISO.isin(self.countries_iso)].reset_index(drop=True)
        df_unp_tot = df_unp_tot[["ISO", "Time", "Population"]]
        df_unp_g = create_groups(self, df_unp_tot, 'Population', 'sum', 'yes')
        df_unp_tot = pd.concat([df_unp_tot, df_unp_g]).reset_index(drop=True)
        dfdummy = df_unp_tot.set_index(['ISO', 'Time'])
        self.xr_unp = xr.Dataset.from_dataframe(dfdummy)

    # =========================================================== #
    # =========================================================== #

    def read_hdi(self):
        print('- Read Human Development Index data')
        df_regions = pd.read_excel(Path("X:/user/dekkerm/Data/") / "AR6_regionclasses.xlsx")
        df_regions = df_regions.sort_values(by=['name'])
        df_regions = df_regions.sort_index()

        self.df_hdi_raw = pd.read_excel(self.path_hdi, sheet_name='Rawdata')
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
                pop = float(self.xr_unp.sel(ISO=r, Time=2019).Population)
            except:
                pop = np.nan
            hdi_sh_values[r_i] = hdi_values[r_i]*pop
        hdi_sh_values = hdi_sh_values / np.nansum(hdi_sh_values)
        df_hdi = {}
        df_hdi['ISO'] = self.countries_iso
        df_hdi["Name"] = self.countries_name
        df_hdi["HDI"] = hdi_values
        df_hdi = pd.DataFrame(df_hdi)
        df_hdi = df_hdi[["ISO", 'HDI']]

        df_hdi_g = create_groups(self, df_hdi, 'HDI', 'mean', 'no')
        df_hdi = pd.concat([df_hdi, df_hdi_g]).reset_index(drop=True)
        dfdummy = df_hdi.set_index(['ISO'])
        self.xr_hdi = xr.Dataset.from_dataframe(dfdummy)

        df_hdi = {}
        df_hdi['ISO'] = self.countries_iso
        df_hdi["Name"] = self.countries_name
        df_hdi["HDIsh"] = hdi_sh_values
        df_hdi = pd.DataFrame(df_hdi)
        df_hdi = df_hdi[["ISO", 'HDIsh']]

        df_hdi_g = create_groups(self, df_hdi, 'HDIsh', 'sum', 'no')
        df_hdi = pd.concat([df_hdi, df_hdi_g]).reset_index(drop=True)
        dfdummy = df_hdi.set_index(['ISO'])
        self.xr_hdish = xr.Dataset.from_dataframe(dfdummy)

    # =========================================================== #
    # =========================================================== #

    def read_gdp(self):
        print('- Read SSP data for GDP')
        df3 = pd.read_excel(self.path_ssp_update, sheet_name = "GDP_2005")
        df3 = df3[(df3.variable.isin(['gdppc'])) & (df3.scenario.isin(["WDI","SSP2"]))]
        df3 = df3.reset_index(drop=True)

        df = pd.read_excel(self.path_ssp_update, sheet_name = "country_level")
        df = df[(df.variable.isin(['pop'])) & (df.scenario.isin(["WDI","SSP2"]))]
        df = df.reset_index(drop=True)
        regions_df = np.unique(df.countryCode)
        regs = np.copy(self.countries_iso)#np.unique(df.countryCode)
        time = np.array(df.keys()[4:])
        df2 = {}
        df2["Region"] = regs
        df2["Variable"] = ["GDP|PPP"]*len(regs)
        for t_i, t in enumerate(time):
            dat = np.zeros(len(regs))+np.nan
            for r_i, r in enumerate(regs):
                if r in regions_df:
                    if int(t) <= 2020:
                        gdppc = float(df3[(df3.countryCode == r) & (df3.variable == "gdppc") & (df3.scenario == "WDI")][t])
                        if gdppc < 1: gdppc = np.nan
                        dat[r_i] = gdppc * float(df[(df.countryCode == r) & (df.variable == "pop") & (df.scenario == "WDI")][t])*1e6
                    else:
                        gdppc = float(df3[(df3.countryCode == r) & (df3.variable == "gdppc") & (df3.scenario == "SSP2")][t])
                        if gdppc < 1:
                            if int(t) != 2100:
                                gdppc = np.nan
                            else:
                                gdppc = float(df3[(df3.countryCode == r) & (df3.variable == "gdppc") & (df3.scenario == "SSP2")]["2095"])
                                if gdppc < 1: gdppc = np.nan
                        dat[r_i] = gdppc * float(df[(df.countryCode == r) & (df.variable == "pop") & (df.scenario == "SSP2")][t])*1e6
            df2[int(t)] = dat
        df2 = pd.DataFrame(df2)
        df2['Regionname'] = [self.countries_name[self.countries_iso == iso][0] for iso in np.array(df2.Region)]
        df_gdp = df2.melt(id_vars=["Regionname", "Region", "Variable"], var_name="Time", value_name="Value")
        df_gdp = df_gdp[["Region", 'Time', 'Value']]
        df_gdp = df_gdp.rename(columns={"Value": "GDP", "Region":"ISO", "Regionname": "Name"})
        df_gdp_g = create_groups(self, df_gdp, 'GDP', 'sum', 'yes')
        df_gdp = pd.concat([df_gdp, df_gdp_g]).reset_index(drop=True)
        dfdummy = df_gdp.set_index(['ISO', 'Time'])
        self.xr_gdp = xr.Dataset.from_dataframe(dfdummy)

    # =========================================================== #
    # =========================================================== #

    def read_edgar(self):
        print('- Read EDGAR historical emissions')
        df_edgar_r = pd.read_excel(self.path_edgar, sheet_name='data')
        df_edgar = df_edgar_r[['ISO', 'year', 'subsector_title', 'GHG']]
        df_edgar = df_edgar.rename(columns={'year': 'Time'})
        ar_time = np.array(df_edgar.Time)
        ar_ghg = np.array(df_edgar.GHG)
        ar_isos = np.array(df_edgar.ISO)
        uni_isos = np.unique(df_edgar.ISO)
        isos = []
        times = []
        ghgs = []
        for y in np.arange(1970, 2021):
            wh = np.where(ar_time == y)[0]
            for cty in uni_isos:
                wh2 = wh[ar_isos[wh] == cty]
                isos.append(cty)
                times.append(y)
                ghgs.append(np.nansum(ar_ghg[wh2]))
        df_ghg = {}
        df_ghg['ISO'] = isos
        df_ghg['Time'] = times
        df_ghg['GHG'] = ghgs
        df_ghg_e = pd.DataFrame(df_ghg)
        df_ghg_e_g = create_groups(self, df_ghg_e, 'GHG', 'sum', 'yes')
        df_ghg_e = pd.concat([df_ghg_e, df_ghg_e_g]).reset_index(drop=True)
        dfdummy = df_ghg_e.set_index(['ISO', 'Time'])
        self.xr_ghg_e = xr.Dataset.from_dataframe(dfdummy)

    # =========================================================== #
    # =========================================================== #

    def read_primap(self):
        print('- Read PRIMAP historical emissions')
        df_primap = pd.read_csv(self.path_primap)
        df_primap = df_primap[(df_primap['scenario (PRIMAP-hist)'] == self.hist_emission_scen) & (df_primap.entity == "KYOTOGHG (AR4GWP100)")]
        df_primap = df_primap.reset_index(drop=True)
        df_primap = df_primap.rename(columns={"area (ISO3)": "ISO", "category (IPCC2006_PRIMAP)": "Source"})

        # Without LULUCF
        df_primap2 = df_primap[(df_primap.Source == "M.0.EL") & (df_primap.ISO.isin(self.countries_iso))]
        df_primap2 = df_primap2.reset_index(drop=True)
        df_primap2 = df_primap2[['ISO']+list(np.arange(1950, 2020, 1).astype(str))]
        df_primap2 = df_primap2.melt(id_vars=["ISO"], var_name="Time", value_name="GHG")
        df_primap2['Time'] = np.array(df_primap2['Time']).astype(int)
        df_ghg_p_g = create_groups(self, df_primap2, 'GHG', 'sum', 'yes')
        df_ghg_p = pd.concat([df_primap2, df_ghg_p_g]).reset_index(drop=True)
        dfdummy = df_ghg_p.set_index(['ISO', 'Time'])
        self.xr_ghg_p = xr.Dataset.from_dataframe(dfdummy)

        # Only LULUCF
        df_primap3 = df_primap[(df_primap.Source == "M.LULUCF") & (df_primap.ISO.isin(self.countries_iso))]
        df_primap3 = df_primap3.reset_index(drop=True)
        df_primap3 = df_primap3[['ISO']+list(np.arange(1950, 2020, 1).astype(str))]
        df_primap3 = df_primap3.melt(id_vars=["ISO"], var_name="Time", value_name="GHG")
        df_primap3['Time'] = np.array(df_primap3['Time']).astype(int)
        df_ghg_p_g = create_groups(self, df_primap3, 'GHG', 'sum', 'yes')
        df_ghg_p = pd.concat([df_primap3, df_ghg_p_g]).reset_index(drop=True)
        dfdummy = df_ghg_p.set_index(['ISO', 'Time'])
        self.xr_lulucf = xr.Dataset.from_dataframe(dfdummy)
        
        # Save the 2021 values
        df_primap2 = df_primap[(df_primap.Source == "M.0.EL") & (df_primap.ISO.isin(self.countries_iso))]
        df_primap2 = df_primap2.reset_index(drop=True)
        df_primap2 = df_primap2[['ISO', '2021']]
        df_primap2 = df_primap2.melt(id_vars=["ISO"], var_name="Time", value_name="GHG")
        df_primap2['Time'] = np.array(df_primap2['Time']).astype(int)
        df_ghg_p_g = create_groups(self, df_primap2, 'GHG', 'sum', 'yes')
        df_ghg_p = pd.concat([df_primap2, df_ghg_p_g]).reset_index(drop=True)
        dfdummy = df_ghg_p.set_index(['ISO', 'Time'])
        self.xr_2021 = xr.Dataset.from_dataframe(dfdummy).sel(Time=2021)

    # =========================================================== #
    # =========================================================== #

    def read_rci(self):
        print("- Read the RCI data")
        df_rci = pd.read_csv(self.path_data / "RCI" / "RCI.xls", delimiter='\t', skiprows=30)[:-2]
        df_rci = df_rci[['iso3', 'year', 'rci']]
        df_rci['year'] = df_rci['year'].astype(int)
        df_rci = df_rci.rename(columns={"iso3": 'ISO', 'year': 'Time'})
        dfdummy = df_rci.set_index(['ISO', 'Time'])
        self.xr_rci = xr.Dataset.from_dataframe(dfdummy)

    # =========================================================== #
    # =========================================================== #

    def read_ar6(self):
        print('- Read AR6 database')
        DF_raw = pd.read_csv(self.path_ar6_data_w)
        DF_raw = DF_raw[DF_raw.Variable.isin(self.ar6_vars_to_keep)]
        DF_raw = DF_raw.reset_index(drop=True)
        DF_meta = pd.read_excel(self.path_ar6_meta_w, sheet_name='meta_Ch3vetted_withclimate')
        mods = np.array(DF_meta.Model)
        scens = np.array(DF_meta.Scenario)
        modscens_meta = np.array([mods[i]+'|'+scens[i] for i in range(len(scens))])
        DF_meta['ModelScenario'] = modscens_meta
        modcat = np.array([np.array(DF_meta.ModelScenario), np.array(DF_meta.Category)])
        mods = np.array(DF_raw.Model)
        scens = np.array(DF_raw.Scenario)
        modscens = np.array([mods[i]+'|'+scens[i] for i in range(len(scens))])
        DF_raw['ModelScenario'] = modscens
        time_keys = np.array(list(np.arange(1995, 2021, 1))+list(np.arange(2025, 2051, 5))+list(np.arange(2060, 2101, 10))).astype(str)
        rows = []
        for c in self.all_categories:
            if len(c)>2:
                df_i = DF_raw[DF_raw.ModelScenario.isin(modcat[0][(modcat[1] == c[:2]) | (modcat[1] == c[3:])])]
            else:
                df_i = DF_raw[DF_raw.ModelScenario.isin(modcat[0][modcat[1] == c])]
            for v_i, v in enumerate(["Emissions|CO2", "Emissions|Kyoto Gases", "Carbon Sequestration|CCS",
                                        'Carbon Sequestration|Direct Air Capture',
                                        "Emissions|Kyoto Gases|w/o LULUCF",
                                        self.gdp_type_scenarios]):
                if v == "Emissions|Kyoto Gases|w/o LULUCF":
                    for t_i, t in enumerate(time_keys):
                        if c != "C4":
                            kyoto_tot = np.nanmean(np.array(df_i[df_i.Variable == "Emissions|Kyoto Gases"][t]))
                            land_ch4 = np.nanmean(np.array(df_i[df_i.Variable == "Emissions|CH4|AFOLU|Land"][t])*self.gwp_ch4)
                            land_co2 = np.nanmean(np.array(df_i[df_i.Variable == "Emissions|CO2|AFOLU|Land"][t])*1.)
                            land_n2o = np.nanmean(np.array(df_i[df_i.Variable == "Emissions|N2O|AFOLU|Land"][t])*self.gwp_n2o)/1000
                            bunk_co2_av = np.nanmean(np.array(df_i[df_i.Variable == "Emissions|CO2|Energy|Demand|Transportation|Aviation"][t])*1.)
                            bunk_co2_mar = np.nanmean(np.array(df_i[df_i.Variable == "Emissions|CO2|Energy|Demand|Transportation|Maritime"][t])*1.)
                            dat = kyoto_tot - land_ch4 - land_co2 - land_n2o - bunk_co2_av - bunk_co2_mar
                        else:
                            df_i0 = DF_raw[DF_raw.ModelScenario.isin(modcat[0][modcat[1] == "C3"])]
                            df_i1 = DF_raw[DF_raw.ModelScenario.isin(modcat[0][modcat[1] == "C5"])]
                            kyoto_tot = np.nanmean(np.array(df_i0[df_i0.Variable == "Emissions|Kyoto Gases"][t]))
                            land_ch4 = np.nanmean(np.array(df_i0[df_i0.Variable == "Emissions|CH4|AFOLU|Land"][t])*self.gwp_ch4)
                            land_co2 = np.nanmean(np.array(df_i0[df_i0.Variable == "Emissions|CO2|AFOLU|Land"][t])*1.)
                            land_n2o = np.nanmean(np.array(df_i0[df_i0.Variable == "Emissions|N2O|AFOLU|Land"][t])*self.gwp_n2o)/1000
                            bunk_co2_av = np.nanmean(np.array(df_i0[df_i0.Variable == "Emissions|CO2|Energy|Demand|Transportation|Aviation"][t])*1.)
                            bunk_co2_mar = np.nanmean(np.array(df_i0[df_i0.Variable == "Emissions|CO2|Energy|Demand|Transportation|Maritime"][t])*1.)
                            dat_0 = kyoto_tot - land_ch4 - land_co2 - land_n2o - bunk_co2_av - bunk_co2_mar

                            kyoto_tot = np.nanmean(np.array(df_i1[df_i1.Variable == "Emissions|Kyoto Gases"][t]))
                            land_ch4 = np.nanmean(np.array(df_i1[df_i1.Variable == "Emissions|CH4|AFOLU|Land"][t])*self.gwp_ch4)
                            land_co2 = np.nanmean(np.array(df_i1[df_i1.Variable == "Emissions|CO2|AFOLU|Land"][t])*1.)
                            land_n2o = np.nanmean(np.array(df_i1[df_i1.Variable == "Emissions|N2O|AFOLU|Land"][t])*self.gwp_n2o)/1000
                            bunk_co2_av = np.nanmean(np.array(df_i1[df_i1.Variable == "Emissions|CO2|Energy|Demand|Transportation|Aviation"][t])*1.)
                            bunk_co2_mar = np.nanmean(np.array(df_i1[df_i1.Variable == "Emissions|CO2|Energy|Demand|Transportation|Maritime"][t])*1.)
                            dat_1 = kyoto_tot - land_ch4 - land_co2 - land_n2o - bunk_co2_av - bunk_co2_mar
                            dat = np.nanmean([dat_0, dat_1])
                        rows.append([c, v, int(t), dat])
                else:
                    df_iv = df_i[df_i.Variable == v]
                    for t_i, t in enumerate(time_keys):
                        dat = np.array(df_iv[t])
                        rows.append([c, v, int(t), np.nanmean(dat)])
        DF_new = pd.DataFrame(rows, columns=["Category", "Variable", "Time", "Mean"])
        df1 = DF_new[DF_new.Variable == "Carbon Sequestration|CCS"]
        df1 = df1[["Category", "Time", "Mean"]]
        dfdummy = df1.set_index(['Category', 'Time'])
        self.xr_ccs = xr.Dataset.from_dataframe(dfdummy)

        df1 = DF_new[DF_new.Variable == 'Carbon Sequestration|Direct Air Capture']
        df1 = df1[["Category", "Time", "Mean"]]
        dfdummy = df1.set_index(['Category', 'Time'])
        self.xr_dac = xr.Dataset.from_dataframe(dfdummy)

        df1 = DF_new[DF_new.Variable == 'Emissions|Kyoto Gases|w/o LULUCF']
        df1 = df1[["Category", "Time", "Mean"]]
        dfdummy = df1.set_index(['Category', 'Time'])
        self.xr_ghg_ar6 = xr.Dataset.from_dataframe(dfdummy)

        df1 = DF_new[DF_new.Variable == 'Emissions|Kyoto Gases']
        df1 = df1[["Category", "Time", "Mean"]]
        dfdummy = df1.set_index(['Category', 'Time'])
        self.xr_ghg_ar6_incl = xr.Dataset.from_dataframe(dfdummy)

        df1 = DF_new[DF_new.Variable == 'Emissions|CO2']
        df1 = df1[["Category", "Time", "Mean"]]
        dfdummy = df1.set_index(['Category', 'Time'])
        self.xr_co2_ar6 = xr.Dataset.from_dataframe(dfdummy)

        df1 = DF_new[DF_new.Variable == 'Emissions|CO2|']
        df1 = df1[["Category", "Time", "Mean"]]
        dfdummy = df1.set_index(['Category', 'Time'])
        self.xr_co2_ar6 = xr.Dataset.from_dataframe(dfdummy)
        self.DF_raw = DF_raw

    # =========================================================== #
    # =========================================================== #

    def read_ndc_cr(self):
        print('- Read NDC projections from Climate Resource (spreadsheet from Yann)')
        df = pd.read_csv(self.path_ndc_cr)
        df = df[df.conditionality == 'unconditional']
        df = df[df.category == 'Current']
        df = df[df.hot_air == 'exclude']
        df_l = df[df.ambition == 'low'].reset_index(drop=True)
        df_l = df_l[['region', '1990',
            '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999',
            '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008',
            '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017',
            '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025', '2026',
            '2027', '2028', '2029', '2030', '2031', '2032', '2033', '2034', '2035',
            '2036', '2037', '2038', '2039', '2040', '2041', '2042', '2043', '2044',
            '2045', '2046', '2047', '2048', '2049', '2050']]
        DF_tim = df_l.melt(id_vars=["region"], var_name="Time", value_name="Value")
        DF_tim = DF_tim.rename(columns = {'region': "ISO"})
        DF_tim['Time'] = np.array(DF_tim['Time']).astype(int)
        DF_tim = DF_tim.reset_index(drop=True)
        df2 = create_groups(self, DF_tim, 'Value', 'sum', 'yes')
        df1b = pd.concat([DF_tim, df2]).reset_index(drop=True)
        dfdummy = df1b.set_index(["ISO", 'Time'])
        self.xr_ndccr_l = xr.Dataset.from_dataframe(dfdummy)
        
        df_h = df[df.ambition == 'high'].reset_index(drop=True)
        df_h = df_h[['region', '1990',
            '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999',
            '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008',
            '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017',
            '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025', '2026',
            '2027', '2028', '2029', '2030', '2031', '2032', '2033', '2034', '2035',
            '2036', '2037', '2038', '2039', '2040', '2041', '2042', '2043', '2044',
            '2045', '2046', '2047', '2048', '2049', '2050']]
        DF_tim = df_h.melt(id_vars=["region"], var_name="Time", value_name="Value")
        DF_tim = DF_tim.rename(columns = {'region': "ISO"})
        DF_tim['Time'] = np.array(DF_tim['Time']).astype(int)
        DF_tim = DF_tim.reset_index(drop=True)
        df2 = create_groups(self, DF_tim, 'Value', 'sum', 'yes')
        df1b = pd.concat([DF_tim, df2]).reset_index(drop=True)
        dfdummy = df1b.set_index(["ISO", 'Time'])
        self.xr_ndccr_h = xr.Dataset.from_dataframe(dfdummy)
        
        df = pd.read_csv(self.path_ndc_cr)
        df = df[df.conditionality == 'conditional']
        df = df[df.category == 'Current']
        df = df[df.hot_air == 'exclude']
        df_l = df[df.ambition == 'low'].reset_index(drop=True)
        df_l = df_l[['region', '1990',
            '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999',
            '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008',
            '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017',
            '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025', '2026',
            '2027', '2028', '2029', '2030', '2031', '2032', '2033', '2034', '2035',
            '2036', '2037', '2038', '2039', '2040', '2041', '2042', '2043', '2044',
            '2045', '2046', '2047', '2048', '2049', '2050']]
        DF_tim = df_l.melt(id_vars=["region"], var_name="Time", value_name="Value")
        DF_tim = DF_tim.rename(columns = {'region': "ISO"})
        DF_tim['Time'] = np.array(DF_tim['Time']).astype(int)
        DF_tim = DF_tim.reset_index(drop=True)
        df2 = create_groups(self, DF_tim, 'Value', 'sum', 'yes')
        df1b = pd.concat([DF_tim, df2]).reset_index(drop=True)
        dfdummy = df1b.set_index(["ISO", 'Time'])
        self.xr_ndccr_l_c = xr.Dataset.from_dataframe(dfdummy)
        
        df_h = df[df.ambition == 'high'].reset_index(drop=True)
        df_h = df_h[['region', '1990',
            '1991', '1992', '1993', '1994', '1995', '1996', '1997', '1998', '1999',
            '2000', '2001', '2002', '2003', '2004', '2005', '2006', '2007', '2008',
            '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017',
            '2018', '2019', '2020', '2021', '2022', '2023', '2024', '2025', '2026',
            '2027', '2028', '2029', '2030', '2031', '2032', '2033', '2034', '2035',
            '2036', '2037', '2038', '2039', '2040', '2041', '2042', '2043', '2044',
            '2045', '2046', '2047', '2048', '2049', '2050']]
        DF_tim = df_h.melt(id_vars=["region"], var_name="Time", value_name="Value")
        DF_tim = DF_tim.rename(columns = {'region': "ISO"})
        DF_tim['Time'] = np.array(DF_tim['Time']).astype(int)
        DF_tim = DF_tim.reset_index(drop=True)
        df2 = create_groups(self, DF_tim, 'Value', 'sum', 'yes')
        df1b = pd.concat([DF_tim, df2]).reset_index(drop=True)
        dfdummy = df1b.set_index(["ISO", 'Time'])
        self.xr_ndccr_h_c = xr.Dataset.from_dataframe(dfdummy)

    # =========================================================== #
    # =========================================================== #

    def read_ndc(self):
        print('- Read NDC dataset (Climate Resource)')
        os.chdir(self.path_data / "NDC" / ("ClimateResource_"+self.version_ndcs) / "Dummy")
        files = os.listdir()
        isos = np.unique([i.split("_")[0].upper() for i in files])
        series = []

        for cty in isos:
            for cond in ['conditional', 'range', 'unconditional']:
                for hotair in ['include', 'exclude']:
                    for ambition in ['low', 'high']:
                        path = 'X:/user/dekkerm/Data/NDC/ClimateResource_'+self.version_ndcs+'/'+cond+'/'+hotair+'/'+cty+'_ndc_'+self.version_ndcs+'_CR_'+cond+'_'+hotair+'.json'
                        with open(path, 'r') as f:
                            dataset = json.load(f)
                        name = dataset['results']['country']['name']
                        for n, i in enumerate(dataset['results']['series']):
                            if i['columns']['variable'] == self.ndc_var:
                                if i['columns']['category'] == "Updated NDC":
                                    if i['columns']['ambition'] ==  ambition:
                                        time = np.array(list(i['data'].keys()))
                                        ghg = np.array(list(i['data'].values()))
                                        ghg[ghg == 'None'] = np.nan
                                        ghg = ghg.astype(float)
                                        time = time.astype(int)
                                        ghg = ghg[time >= 2010]
                                        time = time[time >= 2010]
                                        series.append([cty.upper(), name, self.ndc_var, cond, hotair, ambition]+list(ghg))
        DF_ndc = pd.DataFrame(series, columns = ['Region', "Region name", 'Variable', 'Condition', "Hotair", "Ambition"]+list(time))
        df_save = DF_ndc[['Region', 'Region name', 'Variable', 'Condition', 2010, 2015, 2019, 2020, 2021, 2025, 2030]]
        df_save.to_csv(self.path_data / "NDC" / "ProcessedNDCdata.csv")
        df_sub = DF_ndc[['Region', 'Condition', "Hotair", "Ambition"]+list(np.arange(2010, 2031, 1))]
        DF_tim = df_sub.melt(id_vars=["Region", 'Condition', "Hotair", "Ambition"], var_name="Time", value_name="Value")
        DF_tim = DF_tim.rename(columns = {'Region': "ISO"})
        DF_tim['Time'] = np.array(DF_tim['Time']).astype(int)
        DF_tim = DF_tim[DF_tim.ISO.isin(self.countries_iso)]
        DF_tim = DF_tim.reset_index(drop=True)
        DF_tim = DF_tim[["ISO", "Time", 'Condition', "Hotair", "Ambition", "Value"]]
        DF_tim = DF_tim[DF_tim.Condition == self.conditionality_ndcs]
        DF_tim = DF_tim[DF_tim.Hotair == self.hotair_ndcs]

        # Selection of data needed (low)
        df1 = DF_tim[DF_tim.Ambition == 'low']
        df1 = df1[["ISO", "Time", "Value"]]
        df = create_groups(self, df1, 'Value', 'sum', 'yes')
        df1b = pd.concat([df1, df]).reset_index(drop=True)
        dfdummy = df1b.set_index(["ISO", 'Time'])
        self.xr_ndc_l = xr.Dataset.from_dataframe(dfdummy)
        
        # Selection of data needed (high)
        df2 = DF_tim[DF_tim.Ambition == 'high']
        df2 = df2[["ISO", "Time", "Value"]]
        df = create_groups(self, df2, 'Value', 'sum', 'yes')
        df2b = pd.concat([df2, df]).reset_index(drop=True)
        dfdummy = df2b.set_index(["ISO", 'Time'])
        self.xr_ndc_h = xr.Dataset.from_dataframe(dfdummy)

    # =========================================================== #
    # =========================================================== #

    def read_baseline(self):
        """ Baseline scenario SSP2 from IMAGE in the AR6 database """
        print("- Read baseline")
        time_keys = np.array(list(np.arange(1995, 2021, 5))+list(np.arange(2025, 2051, 5))+list(np.arange(2060, 2101, 10))).astype(str)
        
        DF_raw_iso3 = pd.read_csv(self.path_ar6_data_iso3)
        DF_raw_iso3 = DF_raw_iso3[DF_raw_iso3.Variable.isin(self.ar6_vars_to_keep)]
        DF_raw_iso3 = DF_raw_iso3.reset_index(drop=True)
        DF_meta_iso3 = pd.read_excel(self.path_ar6_meta_iso3, sheet_name='meta_Ch3vetted_withclimate')
        mods = np.array(DF_meta_iso3.Model)
        scens = np.array(DF_meta_iso3.Scenario)
        modscens_meta = np.array([mods[i]+'|'+scens[i] for i in range(len(scens))])
        DF_meta_iso3['ModelScenario'] = modscens_meta
        modcat = np.array([np.array(DF_meta_iso3.ModelScenario), np.array(DF_meta_iso3.Category)])
        mods = np.array(DF_raw_iso3.Model)
        scens = np.array(DF_raw_iso3.Scenario)
        modscens = np.array([mods[i]+'|'+scens[i] for i in range(len(scens))])
        DF_raw_iso3['ModelScenario'] = modscens

        df_b = self.DF_raw[self.DF_raw.ModelScenario == 'IMAGE 3.2|SSP2-baseline']
        baseline_emissions = (np.array(df_b[df_b.Variable == "Emissions|Kyoto Gases"][time_keys])[0]-
                            np.array(df_b[df_b.Variable == "Emissions|CH4|AFOLU|Land"][time_keys])[0]*self.gwp_ch4 -
                            np.array(df_b[df_b.Variable == "Emissions|CO2|AFOLU|Land"][time_keys])[0] -
                            np.array(df_b[df_b.Variable == "Emissions|N2O|AFOLU|Land"][time_keys])[0]*self.gwp_n2o/1000)
        df_b_iso3 = DF_raw_iso3[DF_raw_iso3.ModelScenario == 'IMAGE 3.2|SSP2-baseline']
        unireg = np.unique(df_b_iso3.Region)
        dict_baselines = {}
        for r_i, r in enumerate(unireg):
            df_b_iso3_r = df_b_iso3[df_b_iso3.Region == r]
            baseline_emissions_r = (np.array(df_b_iso3_r[df_b_iso3_r.Variable == "Emissions|Kyoto Gases"][time_keys])[0] -
                                    np.array(df_b_iso3_r[df_b_iso3_r.Variable == "Emissions|CH4|AFOLU|Land"][time_keys])[0]*self.gwp_ch4 -
                                    np.array(df_b_iso3_r[df_b_iso3_r.Variable == "Emissions|CO2|AFOLU|Land"][time_keys])[0] -
                                    np.array(df_b_iso3_r[df_b_iso3_r.Variable == "Emissions|N2O|AFOLU|Land"][time_keys])[0]*self.gwp_n2o/1000)
            dict_baselines[r] = baseline_emissions_r

        # Reread a number of data files
        DF_raw_raw = pd.read_csv(self.path_ar6_data_w)
        DF_raw_iso3 = pd.read_csv(self.path_ar6_data_iso3)
        DF_raw_iso3_onlyco2 = DF_raw_iso3[DF_raw_iso3.Variable == "Emissions|CO2"]
        DF_raw_iso3_onlyco2 = DF_raw_iso3_onlyco2.reset_index(drop=True)

        # Downscale
        all_baseline_emissions = np.zeros(shape=(len(self.all_regions_iso), len(time_keys)))+np.nan

        # Add world
        all_baseline_emissions[-1] = np.copy(baseline_emissions)

        # Insert baseline data for few countries that are in AR6
        for r_i, r in enumerate(unireg):
            all_baseline_emissions[np.where(self.all_regions_iso == r)[0][0]] = dict_baselines[r]

        # Rest with fractions in PRIMAP (2021)
        restemis = all_baseline_emissions[-1] - np.nansum(all_baseline_emissions[:-12], axis=0)
        occupied_idx = np.where(~np.isnan(all_baseline_emissions[:, 10]))[0]
        occupied_idx_c = np.where(~np.isnan(all_baseline_emissions[:-12, 10]))[0]
        occupied = self.all_regions_iso[occupied_idx]#[:-12]

        all_fractions = np.zeros(len(self.all_regions_iso))
        for cty_i, cty in enumerate(self.all_regions_iso):
            try:
                fr = float(self.xr_2021.sel(ISO=cty).GHG/self.xr_2021.sel(ISO='WORLD').GHG)
                all_fractions[cty_i] = fr
            except:
                all_fractions[cty_i] = np.nan
        all_fractions = all_fractions / (1 - np.nansum(all_fractions[occupied_idx_c]))
        for cty_i, cty in enumerate(self.all_regions_iso):
            if cty not in occupied:
                all_baseline_emissions[cty_i] = all_fractions[cty_i]*restemis

        # # Downscale
        # all_baseline_emissions = np.zeros(shape=(len(self.all_regions_iso), len(time_keys)))

        # # Use fractions in AR6 (2015)
        # emis_globe = np.nanmean(np.array(DF_raw_raw[DF_raw_raw.Variable == "Emissions|CO2"]['2015']))
        # regs = np.array(DF_raw_iso3_onlyco2.Region)
        # all_fractions = np.zeros(len(self.all_regions_iso))
        # for cty_i, cty in enumerate(self.all_regions_iso):
        #     fr = np.nanmean(DF_raw_iso3_onlyco2[regs == cty]['2015']) / emis_globe
        #     all_fractions[cty_i] = fr
        #     all_baseline_emissions[cty_i] = fr*baseline_emissions

        # # Add world
        # all_baseline_emissions[-1] = np.copy(baseline_emissions)

        # # Insert actual baseline data
        # for r_i, r in enumerate(unireg):
        #     all_baseline_emissions[np.where(self.all_regions_iso == r)[0][0]] = dict_baselines[r]

        # # Rest with gdp fraction
        # restemis = all_baseline_emissions[-1] - np.nansum(all_baseline_emissions[:-12], axis=0)
        # occupied = self.all_regions_iso[np.where(~np.isnan(all_baseline_emissions[:, 10]))[0]][:-2]
        # total_occupied_gdpshare = np.zeros(len(time_keys))
        # for y_i, y in enumerate(time_keys):
        #     total_occupied_gdpshare[y_i] = np.nansum([gdp_future_reread(self, int(y), r, 'fraction') for r in occupied])
        # total_occupied_gdpshare[0] = total_occupied_gdpshare[1]
        # for r_i, r in enumerate(self.all_regions_iso):
        #     if np.isnan(all_baseline_emissions[r_i, 10]):
        #         all_baseline_emissions[r_i] = restemis*(gdp_future_reread(self, 2015, r, 'fraction'))/(1-total_occupied_gdpshare)

        # Create groups
        d_base = []
        for r_i, r in enumerate(self.all_regions_iso):
            d_base.append([r]+list(all_baseline_emissions[r_i]))
        df_base = pd.DataFrame(d_base, columns = ['ISO']+list(time_keys.astype(int)))
        df_base = df_base.melt(id_vars = ["ISO"], var_name='Time', value_name='Value')
        df_base = df_base.drop(np.array(df_base[df_base.ISO.isin(['CVF', 'G20', 'G7', 'NA', 'AU', 'AF', 'SIDS', 'LDC', 'UM', 'EG'])].index))
        df_base_onlygroups = create_groups(self, df_base, 'Value', 'sum', 'yes')
        df_base_onlygroups = df_base_onlygroups.drop(np.array(df_base_onlygroups[df_base_onlygroups.ISO.isin(['EU', 'WORLD'])].index))
        df_base_withgroups = pd.concat([df_base, df_base_onlygroups]).reset_index(drop=True)
        df_dummy = df_base_withgroups.set_index(['ISO', "Time"])

        # Create XR
        xr_base = xr.Dataset.from_dataframe(df_dummy)
        xr_base = xr_base.reindex(Time = np.arange(2020, 2101))
        xr_base = xr_base.interpolate_na(dim="Time", method="linear")
        #self.xr_base = xr_base

        # Match
        matched_emissions = np.zeros(shape=(len(np.arange(2020, 2101)), len(self.all_regions_iso)))
        for cty_i, cty in enumerate(self.all_regions_iso):
            base = xr_base.sel(Time=2021, ISO=cty).Value
            try:
                primap = self.xr_2021.sel(ISO=cty).GHG/1e3
                correction = float(base-primap)
            except:
                correction = 0
            matched_emissions[:, cty_i] = np.array(xr_base.sel(Time=np.arange(2020, 2101), ISO=cty).Value) - correction
        d_base = []
        for r_i, r in enumerate(self.all_regions_iso):
            d_base.append([r]+list(matched_emissions[:, r_i]))
        df_base = pd.DataFrame(d_base, columns = ['ISO']+list(np.arange(2020, 2101)))
        df_base = df_base.melt(id_vars = ["ISO"], var_name='Time', value_name='Value')
        df_dummy = df_base.set_index(['ISO', "Time"])
        xr_base = xr.Dataset.from_dataframe(df_dummy)
        xr_base = xr_base.reindex(Time = np.arange(2020, 2101))
        xr_base = xr_base.interpolate_na(dim="Time", method="linear")
        self.xr_base_matched = xr_base

    # =========================================================== #
    # =========================================================== #

    def concat_xrobject(self):
        print('- Collect all data together')
        xr_total = self.xr_unp
        xr_total = xr_total.reindex(ISO = self.all_regions_iso)
        xr_total = xr_total.reindex(Time = np.arange(self.historical_emissions_startyear, 2101))
        xr_total = xr_total.assign(GDP = self.xr_gdp["GDP"])
        xr_total = xr_total.assign(HDI = self.xr_hdi["HDI"])
        xr_total = xr_total.assign(HDIsh = self.xr_hdish["HDIsh"])
        xr_total = xr_total.assign(GHG_e = self.xr_ghg_e["GHG"]/1e9)
        xr_total = xr_total.assign(GHG_p = self.xr_ghg_p["GHG"]/1e6)
        xr_total = xr_total.assign(GHG_p_incl = self.xr_ghg_p["GHG"]/1e6+self.xr_lulucf["GHG"]/1e6)
        xr_total = xr_total.assign(NegGHG = self.xr_ccs['Mean']/1e3+self.xr_dac['Mean']/1e3)
        xr_total = xr_total.assign(GHG_f = self.xr_ghg_ar6['Mean']/1e3)
        xr_total = xr_total.assign(GHG_f_incl = self.xr_ghg_ar6_incl['Mean']/1e3)
        xr_total = xr_total.assign(GHG_f_co2 = self.xr_co2_ar6['Mean']/1e3)
        xr_total = xr_total.assign(GHG_base = self.xr_base_matched['Value']/1e3)
        xr_total = xr_total.assign(NDC_l = self.xr_ndc_l['Value']/1e3)
        xr_total = xr_total.assign(NDC_h = self.xr_ndc_h['Value']/1e3)
        xr_total = xr_total.assign(RCI = self.xr_rci['rci'])
        # xr_total = xr_total.assign(NDCcr_l = self.xr_ndccr_l['Value']/1e3)
        # xr_total = xr_total.assign(NDCcr_h = self.xr_ndccr_h['Value']/1e3)
        # xr_total = xr_total.assign(NDCcr_l_c = self.xr_ndccr_l_c['Value']/1e3)
        # xr_total = xr_total.assign(NDCcr_h_c = self.xr_ndccr_h_c['Value']/1e3)
        xr_total = xr_total.interpolate_na(dim="Time", method="linear")
        xr_total = xr_total.reindex(ISO = self.all_regions_iso)
        xr_total = xr_total.reindex(Category = self.all_categories)
        self.xr_total = xr_total

    # =========================================================== #
    # =========================================================== #

    def save_xrs_databases(self):
        print('- Save XR databases')
        self.xr_total.to_netcdf(self.path_repo_data / str(self.historical_emissions_startyear) / "xr_total.nc")

    # =========================================================== #
    # =========================================================== #

    def load_xrs_databases(self):
        print('- Load XR databases')
        self.xr_total = xr.open_dataset(self.path_repo_data / str(self.historical_emissions_startyear) / "xr_total.nc")

    # =========================================================== #
    # =========================================================== #

    def general_calculations(self):
        print('- General calculations for effort sharing')
        self.all_future_years = np.arange(self.start_year, 2101)
        self.xr_sbudgets = self.xr_total.GHG_f.sel(Category=self.all_categories, Time=self.all_future_years)
        #self.total_world_emissions_start = float(self.xr_total.sel(ISO = "WORLD", Time=self.start_year-1, Category=self.all_categories).GHG_f.mean(dim='Category'))
        self.pc_weighted_budget = (self.xr_total.sel(ISO = self.all_regions_iso, Time = self.all_future_years).Population/self.xr_total.sel(ISO = "WORLD", Time = self.all_future_years).Population * self.xr_total.sel(ISO = "WORLD", Time = self.all_future_years).GHG_f).sum(dim="Time")
        self.discounts = (1.+self.discount_factor)**(np.array(list(np.arange(self.historical_emissions_startyear, self.start_year)-self.start_year+1)+[0]*len(np.arange(self.start_year, 2101))))
        self.historical_debt = ((self.xr_total.Population.sel(ISO=self.all_regions_iso, Time=np.arange(self.historical_emissions_startyear, self.start_year)) /
                                self.xr_total.Population.sel(ISO="WORLD", Time=np.arange(self.historical_emissions_startyear, self.start_year)) *
                                self.xr_total.GHG_p.sel(ISO='WORLD', Time=np.arange(self.historical_emissions_startyear, self.start_year)) -
                                self.xr_total.GHG_p.sel(ISO=self.all_regions_iso, Time=np.arange(self.historical_emissions_startyear, self.start_year)))*(1.+self.discount_factor)**(np.arange(self.historical_emissions_startyear, self.start_year)-(self.start_year+1))).sum(dim = 'Time')
        self.historical_emissions_discounted = (self.xr_total.GHG_p.sel(ISO=self.all_regions_iso, Time=np.arange(self.historical_emissions_startyear, self.start_year))*(1.+self.discount_factor)**(np.arange(self.historical_emissions_startyear, self.start_year)-(self.start_year+1))).sum(dim='Time')
        yearly_netto_budgets = self.xr_total.sel(Category=self.all_categories, Time=self.all_future_years).GHG_f
        cumulative_future_ecpc = (yearly_netto_budgets*self.xr_total.Population.sel(ISO=self.all_regions_iso, Time=self.all_future_years)/self.xr_total.Population.sel(ISO="WORLD", Time=self.all_future_years)).sum(dim='Time')
        yearly_negative_budgets = self.xr_total.sel(Category=self.all_categories, Time=self.all_future_years).NegGHG
        yearly_positive_budgets = yearly_negative_budgets+yearly_netto_budgets
        self.yearly_positive_budgets = yearly_positive_budgets

        # ECPC
        self.historical_emissions_discounted_peryear = self.xr_total.GHG_p.sel(ISO=self.all_regions_iso, Time=np.arange(self.historical_emissions_startyear, self.start_year))*(1.+self.discount_factor)**(np.arange(self.historical_emissions_startyear, self.start_year)-(self.start_year+1))
        self.historical_percapitafractions = self.xr_total.Population.sel(ISO=self.all_regions_iso, Time=np.arange(self.historical_emissions_startyear, self.start_year)) / self.xr_total.Population.sel(ISO="WORLD", Time=np.arange(self.historical_emissions_startyear, self.start_year))
        self.future_percapitafractions = self.xr_total.Population.sel(ISO=self.all_regions_iso, Time=np.arange(self.start_year, 2101)) / self.xr_total.Population.sel(ISO="WORLD", Time=np.arange(self.start_year, 2101))
        self.ecpc_allowed = (self.historical_emissions_discounted_peryear.sel(ISO='WORLD') * self.historical_percapitafractions).sum(dim='Time') + (self.future_percapitafractions * yearly_netto_budgets).sum(dim='Time')
        self.ecpc_spent = self.historical_emissions_discounted_peryear.sum(dim='Time')
        self.ecpc_total = self.ecpc_allowed-self.ecpc_spent
        # self.ecpc_total = self.historical_debt + cumulative_future_ecpc
        ecpc_fractions = (self.ecpc_total) / (self.ecpc_total.sel(ISO='WORLD'))
        self.ecpc = ecpc_fractions * yearly_netto_budgets

        # ECPC - average version
        cumulative_population_fraction = self.xr_total.Population.sel(Time=np.arange(self.historical_emissions_startyear, 2101)).sum(dim='Time') / self.xr_total.Population.sel(ISO="WORLD", Time=np.arange(self.historical_emissions_startyear, 2101)).sum(dim='Time')
        total_emissions_possible = self.historical_emissions_discounted.sel(ISO='WORLD') + yearly_netto_budgets.sum(dim='Time')
        total_emissions_allowance = total_emissions_possible * cumulative_population_fraction
        total_emissions_spent = self.historical_emissions_discounted
        self.ecpc_average = total_emissions_allowance - total_emissions_spent
        ecpc_av_fractions = (self.ecpc_average) / (self.ecpc_average.sel(ISO='WORLD'))
        self.ecpc_average_time = ecpc_av_fractions * yearly_netto_budgets

        # Approach 1 (GDP)
        self.app1_gdp = self.xr_total.sel(ISO = self.all_regions_iso, Time = self.all_future_years).GDP
        self.app1_gdp = self.app1_gdp.where(~self.app1_gdp.ISO.isin(["COK", "VAT", "NIU", "SOM", "GMB", "LIE", "PSE", "MCO", "NRU"]), np.nan)
        self.app1_gdp_neg = self.app1_gdp/self.app1_gdp.sel(ISO = "WORLD", Time = self.all_future_years) * yearly_negative_budgets
        self.app1_gdp_pos_total = self.ecpc_total + self.app1_gdp_neg.sum(dim='Time')
        self.app1_gdp_pos = self.app1_gdp_pos_total / self.app1_gdp_pos_total.sel(ISO='WORLD') * yearly_positive_budgets
        self.app1_gdp_net = self.app1_gdp_pos - self.app1_gdp_neg

        # Approach 1 (HDI)
        self.app1_hdi = self.xr_total.sel(ISO = self.all_regions_iso).HDIsh
        self.app1_hdi = self.app1_hdi.where(~self.app1_hdi.ISO.isin(["COK", "VAT", "NIU", "SOM", "GMB", "LIE", "PSE", "MCO", "NRU"]), np.nan)
        self.app1_hdi_neg = self.app1_hdi * yearly_negative_budgets
        self.app1_hdi_pos_total = self.ecpc_total + self.app1_hdi_neg.sum(dim='Time')
        self.app1_hdi_pos = self.app1_hdi_pos_total / self.app1_hdi_pos_total.sel(ISO='WORLD') * yearly_positive_budgets
        self.app1_hdi_net = self.app1_hdi_pos - self.app1_hdi_neg

        # Approach 2
        self.app2_pos_shares = (self.xr_total.Population**2 / self.xr_total.GDP).sel(ISO=self.all_regions_iso, Time=self.all_future_years)
        self.app2_pos_shares = self.app2_pos_shares.where(~self.app2_pos_shares.ISO.isin(["AND", "ATG", "DMA", "GRD", "KIR", "MHL", "FSM", "MCO", "NRU", "PRK", "PLW", "KNA", "SMR", "SYC", "SSD", "TUV",    "COK", "VAT", "NIU", "SOM", "GMB", "LIE", "PSE"]), np.nan)
        self.app2_pos_shares = self.app2_pos_shares / self.app2_pos_shares.sel(ISO=self.countries_iso, Time=self.all_future_years).sum(dim='ISO')
        app2_current_debt = self.historical_emissions_discounted.copy()
        app2_current_debt = app2_current_debt.where(~app2_current_debt.ISO.isin(["AND", "ATG", "DMA", "GRD", "KIR", "MHL", "FSM", "MCO", "NRU", "PRK", "PLW", "KNA", "SMR", "SYC", "SSD", "TUV",    "COK", "VAT", "NIU", "SOM", "GMB", "LIE", "PSE"]), np.nan)
        self.app2_nets = np.zeros(shape=(len(self.all_categories), len(self.all_regions_iso), len(self.all_future_years)))
        self.app2_negs = np.zeros(shape=(len(self.all_categories), len(self.all_regions_iso), len(self.all_future_years)))
        self.app2_poss = np.zeros(shape=(len(self.all_categories), len(self.all_regions_iso), len(self.all_future_years)))
        for y_i, y in enumerate(self.all_future_years):
            neg = yearly_negative_budgets.sel(Time=y, Category=self.all_categories)*(app2_current_debt/app2_current_debt.sel(ISO=self.countries_iso).sum(dim='ISO'))
            pos = yearly_positive_budgets.sel(Time=y, Category=self.all_categories)*self.app2_pos_shares.sel(Time=y)
            self.app2_nets[:, :, y_i] = pos - neg
            app2_current_debt = app2_current_debt+pos-neg
            self.app2_negs[:, :, y_i] = neg
            self.app2_poss[:, :, y_i] = pos
        self.app2_nets = grouping(self, self.app2_nets)
        self.app2_negs = grouping(self, self.app2_negs)
        self.app2_poss = grouping(self, self.app2_poss)

        # AP
        future_bau = self.xr_total.GHG_base.sel(ISO=self.all_regions_iso, Time=self.all_future_years)
        future_gdp = self.xr_total.GDP.sel(ISO=self.all_regions_iso, Time=self.all_future_years)
        future_pop = self.xr_total.Population.sel(ISO=self.all_regions_iso, Time=self.all_future_years)
        future_emis = self.xr_total.GHG_f.sel(Time=self.all_future_years, Category=self.all_categories)
        RA = ((future_gdp/future_pop) / (future_gdp.sel(ISO='WORLD')/future_pop.sel(ISO='WORLD')) )**(1/3.)*future_bau*(future_bau.sel(ISO='WORLD') - future_emis)/future_bau.sel(ISO='WORLD')
        RA = grouping_xarray(self, RA) # To make sure the groups are equal to the sum of their parts, which is not in the current calculations, given the 1/3 power.
        self.ap = future_bau - RA/RA.sel(ISO='WORLD')*(future_bau.sel(ISO='WORLD') - future_emis)

        # GDR
        gdr = (future_bau - (future_bau.sel(ISO='WORLD') - future_emis)*self.xr_total.RCI)
        yearfracs = xr.Dataset(data_vars={"Value": (['Time'], (np.arange(2020, 2101) - 2030) / (self.convergence_year_gdr - 2030))}, coords={"Time": np.arange(2020, 2101)})
        gdr = gdr.where(gdr.Time < 2030, (future_bau - (future_bau.sel(ISO='WORLD') - future_emis)*self.xr_total.sel(Time=2030).RCI)*(1-yearfracs) + yearfracs*self.ap)
        gdr = grouping_xarray(self, gdr)
        self.gdr = gdr

        # PCC and F2C (transition period stuff)
        self.gf = self.xr_total.GHG_p.sel(Time=2019) / self.xr_total.GHG_p.sel(Time=2019, ISO='WORLD') * yearly_netto_budgets
        self.pc = self.xr_total.Population / self.xr_total.Population.sel(ISO='WORLD') * yearly_netto_budgets
        gf_f = self.gf / yearly_netto_budgets
        pc_f = self.pc / yearly_netto_budgets
        yearfracs_2 = xr.Dataset(data_vars={"Value": (['Time'], (self.all_future_years-2020)/(self.convergence_moment-2020))}, coords={"Time": self.all_future_years})
        pcc = self.pc.copy()
        self.pcc = pcc.where(pcc.Time > self.convergence_moment, (gf_f*(1-yearfracs_2) + pc_f*yearfracs_2)*yearly_netto_budgets)
        self.f2 = xr.Dataset(data_vars={"Value": (['Category', 'ISO', 'Time'], self.app2_nets)}, coords={'Category': self.all_categories, "ISO": self.all_regions_iso, "Time": self.all_future_years})
        f2_f = self.f2 / yearly_netto_budgets
        f2c = self.f2.copy()
        self.f2c = f2c.where(f2c.Time > self.convergence_moment, (gf_f*(1-yearfracs_2) + f2_f*yearfracs_2)*yearly_netto_budgets)

        # Construct Xarray
        fullxr = xr.Dataset(data_vars={"GF": (['ISO', 'Category', 'Time'], self.gf.data),
                                    "PC": (['ISO', 'Time', 'Category'], self.pc.data),
                                    "PCC": (['ISO', 'Time', 'Category'], self.pcc.Value.data),
                                    "AP": (['ISO', 'Time', 'Category'], self.ap.data),
                                    "GDR": (['ISO', 'Time', 'Category'], self.gdr.Value.data),
                                    "ECPC": (['ISO', 'Category', 'Time'], self.ecpc.data),
                                    "ECPCav": (['Category', 'ISO', 'Time'], self.ecpc_average_time.data),
                                    "F1g": (['ISO', 'Category', 'Time'], self.app1_gdp_net.data),
                                    "F1h": (['ISO', 'Category', 'Time'], self.app1_hdi_net.data),
                                    # "F1g": (['Category', 'ISO', 'Time'], self.app1_gdp_net.data),
                                    # "F1h": (['Category', 'ISO', 'Time'], self.app1_hdi_net.data),
                                    "F2": (['Category', 'ISO', 'Time'], self.f2.Value.data),
                                    "F2C": (['Category', 'ISO', 'Time'], self.f2c.Value.data)},
                    coords={'Category': self.all_categories, "ISO": self.all_regions_iso, "Time": self.all_future_years})
        self.xr_budgets_scenario = fullxr.copy()
        self.xr_budgets_static = fullxr.sum(dim='Time')

    # =========================================================== #
    # =========================================================== #

    def save_budgets(self):
        print('- Save budgets')
        self.xr_budgets_static.to_netcdf(self.path_repo_data / str(self.historical_emissions_startyear) / "xr_budgets_static.nc")
        # self.xr_budgets_linear.to_netcdf(self.path_repo_data / str(self.historical_emissions_startyear) / "xr_budgets_linear.nc")
        self.xr_budgets_scenario.to_netcdf(self.path_repo_data / str(self.historical_emissions_startyear) / "xr_budgets_scenario.nc")
        np.save(self.path_repo_data / str(self.historical_emissions_startyear) / "all_regions.npy", self.all_regions_iso)
        np.save(self.path_repo_data / str(self.historical_emissions_startyear) / "all_regions_names.npy", self.all_regions_names)
        np.save(self.path_repo_data / str(self.historical_emissions_startyear) / "all_countries.npy", self.countries_iso)
        np.save(self.path_repo_data / str(self.historical_emissions_startyear) / "all_countries_names.npy", self.countries_name)
        np.save(self.path_repo_data / str(self.historical_emissions_startyear) / "all_future_years.npy", self.all_future_years)

# =========================================================== #
# INITIALIZATION OF CLASS WHEN CALLED
# =========================================================== #

if __name__ == "__main__":

    vardec = shareefforts()