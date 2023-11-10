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
        self.Tlist = np.arange(1.4, 2.4+1e-9, 0.1).astype(float).round(2)           # At least between 1.3 and 2.4
        self.Plist = np.array([.17, 0.33, 0.50, 0.67, 0.83]).round(2)               # At least between 0.17 and 0.83
        #self.NClist = np.arange(0.05, 0.95+1e-9, 0.15).astype(float).round(2)
        #self.Neglist = np.array([0.20, 0.50, 0.80])
        self.Neglist = np.array([0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80])

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
        xr_primap = xr_primap.drop_vars(['source', 'Version'])
        xr_primap['CO2_hist'] = xr_primap['CO2_hist']/1e3
        xr_primap['GHG_hist'] = xr_primap['GHG_hist']/1e3
        xr_primap['CH4_hist'] = xr_primap['CH4_hist']/1e3
        xr_primap['N2O_hist'] = xr_primap['N2O_hist']/1e3
        self.xr_primap = xr_primap.sel(Time=np.arange(1750, self.settings['params']['start_year_analysis']+1), provenance='measured')

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
                                                    'AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile',
                                                    'AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|5.0th Percentile',
                                                    'AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|95.0th Percentile'])]
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
        self.xr_ar6 = xr_scen2.interpolate_na(dim="Time", method="linear")

    # =========================================================== #
    # =========================================================== #

    def determine_global_budgets(self):
        print('- Get global budgets based on IPCC WG1 table')

        # CO2 budgets
        df_budgets = pd.read_csv("X:/user/dekkerm/Data/Budgets_Forster2023/ClimateIndicator-data-ed37002/data/carbon_budget/updateMAGICCscen_temp2013_2022-budget.csv")
        df_budgets = df_budgets[["dT_targets", "0.1", "0.17", "0.33", "0.5", "0.66", "0.83", '0.9']]
        dummy = df_budgets.melt(id_vars=["dT_targets"], var_name="Probability", value_name="Budget")
        ar = np.array(dummy['Probability'])
        ar = ar.astype(float).round(2)
        ar[ar == 0.66] = 0.67
        dummy['Probability'] = ar
        dummy['dT_targets'] = dummy['dT_targets'].astype(float).round(1)
        dummy = dummy.set_index(["dT_targets", "Probability"])
        dummy['Budget'] = dummy['Budget'] + float(self.xr_primap.sel(Region='EARTH', Time=2021).CO2_hist)/1e3 + 40.9 # from Forster 2023 for the year 2022 -> 1 Jan 2021 as starting year! So not + float(self.xr_primap.sel(Region='EARTH', Time=2020).CO2_hist)/1e3 
        xr_bud_co2 = xr.Dataset.from_dataframe(dummy)
        xr_bud_co2 = xr_bud_co2.rename({'dT_targets': "Temperature"}).sel(Temperature = [1.5, 1.7, 2.0])

        # non-CO2 trajectories
        ch4_hist_2020 = self.xr_primap.sel(Region='EARTH').sel(Time=2020).CH4_hist
        n2o_hist_2020 = self.xr_primap.sel(Region='EARTH').sel(Time=2020).N2O_hist
        temps = []
        probs = []
        ch4s = []
        n2os = []
        for p_i, p in enumerate(np.array(xr_bud_co2.Probability)):
            for t_i, t in enumerate([1.5, 1.7, 2.0]):
                ch4_red = [0.48, 0.47, 0.35][t_i]
                n2o_red = [0.25, 0.15, 0.09][t_i]
                ch4_hist_2050 = ch4_hist_2020*(1-ch4_red)
                n2o_hist_2050 = n2o_hist_2020*(1-n2o_red)
                probs.append(p)
                temps.append(t)
                ch4s.append(1e-3*self.settings['params']['gwp_ch4']*float((self.xr_ar6.sel(Variable='Emissions|CH4', Time=np.arange(2021, 2101), ModelScenario = self.xr_ar6.ModelScenario[np.where(np.abs(self.xr_ar6.sel(Variable = 'Emissions|CH4', Time=2050).Value - ch4_hist_2050) < ch4_hist_2050*0.1)[0]]).mean(dim='ModelScenario').Value).sum(dim='Time')))
                n2os.append(1e-6*self.settings['params']['gwp_n2o']*float((self.xr_ar6.sel(Variable='Emissions|N2O', Time=np.arange(2021, 2101), ModelScenario = self.xr_ar6.ModelScenario[np.where(np.abs(self.xr_ar6.sel(Variable = 'Emissions|N2O', Time=2050).Value/1e3 - n2o_hist_2050) < n2o_hist_2050*0.1)[0]]).mean(dim='ModelScenario').Value).sum(dim='Time')))

        dict_ch4 = {}
        dict_ch4['Temperature'] = temps
        dict_ch4['Probability'] = probs
        dict_ch4['Budget'] = ch4s
        df_ch4 = pd.DataFrame(dict_ch4)
        dummy = df_ch4.set_index(["Temperature", "Probability"])
        xr_bud_ch4 = xr.Dataset.from_dataframe(dummy)

        dict_n2o = {}
        dict_n2o['Temperature'] = temps
        dict_n2o['Probability'] = probs
        dict_n2o['Budget'] = n2os
        df_n2o = pd.DataFrame(dict_n2o)
        dummy = df_n2o.set_index(["Temperature", "Probability"])
        xr_bud_n2o = xr.Dataset.from_dataframe(dummy)
        xr_bud_ghg = xr_bud_co2 + xr_bud_ch4 + xr_bud_n2o

        # Interpolate
        Blist = np.zeros(shape=(len(self.Tlist), len(self.Plist)))+np.nan
        for p_i, p in enumerate(self.Plist):
            a, b = np.polyfit(xr_bud_ghg.Temperature, xr_bud_ghg.sel(Probability = np.round(p, 2)).Budget, 1)
            for t_i, t in enumerate(self.Tlist): # Interpolate only on the level of temperature
                Blist[t_i, p_i] = a*t+b
        data2 = xr.DataArray(Blist,
                            coords={'Temperature': self.Tlist,
                                    'Risk': (1-self.Plist).astype(float).round(2)},
                            dims=['Temperature', 'Risk'])
        self.xr_budgets = xr.Dataset({'Budget': data2})

        # # Read WG1 table
        # df_budgets = pd.read_excel(self.settings['paths']['data']['internal']+"CarbonBudgets_IPCC.xlsx", sheet_name='Sheet1')
        # co2_emis_2020 = float(self.xr_primap.sel(Region='WORLD', Time=2020).CO2_hist)
        # data = xr.DataArray(np.array(df_budgets[df_budgets.keys()[1:]])-co2_emis_2020/1e3, coords=[np.array(df_budgets.Temperature).astype(float).round(2), np.array([0.17, 0.33, 0.50, 0.67, 0.83]).astype(float).round(2)], dims=['Temperature', 'TCRE_perc'])
        # dataset = xr.Dataset({'Budget': data})
        # interp2 = scipy.interpolate.RegularGridInterpolator((data.Temperature, data.TCRE_perc), np.array(data))

        # # Get chosen values
        # Blist = np.zeros(shape=(len(self.Tlist), len(self.Plist), len(self.NClist)))+np.nan
        # for t_i, t in enumerate(self.Tlist):
        #     for p_i, p in enumerate(self.Plist):
        #         for nc_i, nc in enumerate(self.NClist):
        #             t = np.round(t, 2)
        #             p = np.round(p, 2)
        #             nc = np.round(nc, 2)
        #             Blist[t_i, p_i, nc_i] = interp2([t, p])+np.linspace(-220, 220, len(self.NClist))[::-1][nc_i]
        # data2 = xr.DataArray(Blist,
        #                     coords={'Temperature': self.Tlist,
        #                             'Risk': (1-self.Plist).astype(float).round(2),
        #                             'NonCO2': self.NClist},
        #                     dims=['Temperature', 'Risk', 'NonCO2'])
        # self.xr_budgets = xr.Dataset({'Budget': data2})

    # =========================================================== #
    # =========================================================== #

    def determine_global_trajectories(self):
        print('- Get global trajectories based on AR6 data')
        # Determine the trajectory, dependent on Trajectory_uncertainty and NegEmis
        startpoint = self.xr_primap.sel(Time=self.settings['params']['start_year_analysis'], Region="EARTH").GHG_hist
        #compensation_form = (np.arange(self.settings['params']['start_year_analysis'], 2101)-self.settings['params']['start_year_analysis'])**0.35    #np.array([0]+[1]*(len(np.arange(self.settings['params']['start_year_analysis'], 2101))-1))#(-1+np.arange(1, 2101-self.settings['params']['start_year_analysis']+1)**0.1)
        compensation_form = np.array(list(np.linspace(0, 1, len(np.arange(self.settings['params']['start_year_analysis'], 2050))))+[1]*len(np.arange(2050, 2101)))
        xr_comp =  xr.DataArray(compensation_form, dims=['Time'], coords={'Time': np.arange(self.settings['params']['start_year_analysis'], 2101)})

        # Create an empty xarray.Dataset
        xr_traj = xr.Dataset(
            coords={
                'NegEmis': self.Neglist,
                'TrajUnc': ['Earliest', 'Early', 'Medium', 'Late', 'Latest'],
                'Temperature': self.Tlist,
                'Risk': self.Plist,
                'Time': np.arange(self.settings['params']['start_year_analysis'], 2101),
                #'NonCO2': self.NClist
            }
        )

        # Initialize data arrays for each variable
        pathways_data = {
            'GHG_globe': xr.DataArray(
                data=np.nan,
                coords=xr_traj.coords,
                dims=('NegEmis', 'TrajUnc', 'Temperature', 'Risk', 'Time'),
                attrs={'description': 'Pathway data'}
            )
        }

        xr_scen2_use = self.xr_ar6.sel(Variable='Emissions|Kyoto Gases')
        xr_scen2_use = xr_scen2_use.reindex(Time = np.arange(2000, 2101, 10))
        xr_scen2_use = xr_scen2_use.reindex(Time = np.arange(2000, 2101))
        xr_scen2_use = xr_scen2_use.interpolate_na(dim="Time", method="linear")
        xr_scen2_use = xr_scen2_use.reindex(Time = np.arange(self.settings['params']['start_year_analysis'], 2101))
        totalnet = xr_scen2_use.sel(Time=np.arange(self.settings['params']['start_year_analysis'], 2101)).sum(dim='Time')/1e3
        kyoto2021 = xr_scen2_use.sel(Time=self.settings['params']['start_year_analysis'])/1e3
        offsets = (startpoint/1e3-kyoto2021)
        emis_all = xr_scen2_use.sel(Time=np.arange(self.settings['params']['start_year_analysis'], 2101))/1e3 + offsets
        totalnets_corr = emis_all.sel(Time=np.arange(self.settings['params']['start_year_analysis'], 2101)).sum(dim='Time')

        xr_scen2_neg = self.xr_ar6.sel(Variable=['Carbon Sequestration|CCS',
                                                'Carbon Sequestration|Land Use']).sum(dim='Variable')
        xr_scen2_neg = xr_scen2_neg.reindex(Time = np.arange(2000, 2101, 10))
        xr_scen2_neg = xr_scen2_neg.reindex(Time = np.arange(2000, 2101))
        xr_scen2_neg = xr_scen2_neg.interpolate_na(dim="Time", method="linear")
        xr_scen2_neg = xr_scen2_neg.reindex(Time = np.arange(self.settings['params']['start_year_analysis'], 2101))
        totalneg = xr_scen2_neg.sel(Time=np.arange(2080, 2101)).mean(dim='Time')

        for temp_i, temp in enumerate(self.Tlist):
            for risk_i, risk in enumerate(self.Plist):
                budget = self.xr_budgets.Budget.sel(Temperature=temp, Risk=risk)
                ms1 = self.xr_ar6.ModelScenario[np.where(np.abs(totalnets_corr.Value - budget) < budget*0.2)[0]]
            
                # The 90-percentile
                totalneg_i = totalneg.sel(ModelScenario=ms1)
                ms_90 = self.xr_ar6.sel(ModelScenario=ms1).ModelScenario[(totalneg_i >= totalneg_i.quantile(0.9-0.1)
                                            ).Value & (totalneg_i <= totalneg_i.quantile(0.9+0.1)).Value]
                time_evo_neg_90 = xr_scen2_neg.sel(ModelScenario=ms_90).mean(dim='ModelScenario')
            
                # The 10-percentile
                totalneg_i = totalneg.sel(ModelScenario=ms1)
                ms_10 = self.xr_ar6.sel(ModelScenario=ms1).ModelScenario[(totalneg_i >= totalneg_i.quantile(0.1-0.1)
                                            ).Value & (totalneg_i <= totalneg_i.quantile(0.1+0.1)).Value]
                time_evo_neg_10 = xr_scen2_neg.sel(ModelScenario=ms_10).mean(dim='ModelScenario')

                # Difference and smoothen this
                surplus_factor = emis_all.sel(ModelScenario = np.intersect1d(ms_90, ms1)).mean(dim='ModelScenario').Value - emis_all.sel(ModelScenario = np.intersect1d(ms_10, ms1)).mean(dim='ModelScenario').Value
                wh = np.where(surplus_factor[:30] < 0)[0]
                surplus_factor[wh] = 0
                surplus_factor = savgol_filter(surplus_factor, 40, 3)

                for neg_i, neg in enumerate(self.Neglist):
                    xset = emis_all.sel(ModelScenario=ms1)+surplus_factor*(neg-0.5)*2
                    factor = (self.xr_budgets.Budget.sel(Temperature=temp, Risk=risk) - xset.sum(dim='Time')) / np.sum(compensation_form)
                    all_pathways = (1e3*(xset+factor*xr_comp)).Value/1e3
                    if len(all_pathways)>0:
                        for unc_i, unc in enumerate(['Earliest', 'Early', 'Medium', 'Late', 'Latest']):
                            ms_path = all_pathways.ModelScenario[np.where(all_pathways.sel(Time=2030) < np.nanpercentile(all_pathways.sel(Time=2030), [10, 25, 50, 75, 90][unc_i], axis=0))[0]]
                            pathway = all_pathways.sel(ModelScenario = ms_path).mean(dim='ModelScenario')#np.array(all_pathways.where(all_pathways.sel(Time=2050) < np.percentile(all_pathways.sel(Time=2050), [10, 25, 50, 75, 90][unc_i], axis=0)).mean(dim='ModelScenario'))#np.percentile(all_pathways, [20, 50, 80][unc_i], axis=0)
                            pathway = savgol_filter(pathway, 20, 3)
                            offset = float(startpoint)/1e3 - pathway[0]
                            pathways_data['GHG_globe'][neg_i, unc_i, temp_i, risk_i, :] = (pathway.T+offset)*1e3
        self.xr_traj = xr_traj.update(pathways_data)
        # # First get quantiles of negative emissions
        # neg_quants = {}
        # for lim in np.arange(0, 1.01, 0.05):
        #     neg_quants[np.round(lim, 2)] = float(self.xr_ar6.sel(Variable = ['Carbon Sequestration|CCS',
        #                                                                      'Carbon Sequestration|Land Use'],
        #                                     Time=np.arange(2090, 2101)).sum(dim=['Time', 'Variable']).quantile(lim).Value)

        # # Then determine the trajectory, dependent on Trajectory_uncertainty and NegEmis
        # startpoint = self.xr_primap.sel(Time=self.settings['params']['start_year_analysis'], Region="EARTH").GHG_hist
        # compensation_form = (np.arange(self.settings['params']['start_year_analysis'], 2101)-self.settings['params']['start_year_analysis'])**0.35    #np.array([0]+[1]*(len(np.arange(self.settings['params']['start_year_analysis'], 2101))-1))#(-1+np.arange(1, 2101-self.settings['params']['start_year_analysis']+1)**0.1)
        # xr_comp =  xr.DataArray(compensation_form, dims=['Time'], coords={'Time': np.arange(self.settings['params']['start_year_analysis'], 2101)})

        # # Create an empty xarray.Dataset
        # xr_traj = xr.Dataset(
        #     coords={
        #         'NegEmis': self.Neglist,
        #         'TrajUnc': ['Earliest', 'Early', 'Medium', 'Late', 'Latest'],
        #         'Temperature': self.Tlist,
        #         'Risk': self.Plist,
        #         'Time': np.arange(self.settings['params']['start_year_analysis'], 2101),
        #         #'NonCO2': self.NClist
        #     }
        # )

        # # Initialize data arrays for each variable
        # pathways_data = {
        #     'GHG_globe': xr.DataArray(
        #         data=np.nan,
        #         coords=xr_traj.coords,
        #         dims=('NegEmis', 'TrajUnc', 'Temperature', 'Risk', 'Time'),
        #         attrs={'description': 'Pathway data'}
        #     )
        # }

        # xr_scen2_use = self.xr_ar6.sel(Variable='Emissions|Kyoto Gases')
        # xr_scen2_use = xr_scen2_use.reindex(Time = np.arange(2000, 2101, 10))
        # xr_scen2_use = xr_scen2_use.reindex(Time = np.arange(2000, 2101))
        # xr_scen2_use = xr_scen2_use.interpolate_na(dim="Time", method="linear")
        # xr_scen2_use = xr_scen2_use.reindex(Time = np.arange(self.settings['params']['start_year_analysis'], 2101))
        # rows = []
        # for neg_i, neg in enumerate(self.Neglist):
        #     ms1 = self.xr_ar6.ModelScenario[(self.xr_ar6.sel(Variable = ['Carbon Sequestration|CCS', 'Carbon Sequestration|Land Use'], Time=np.arange(2090, 2101)).sum(dim=['Time', 'Variable']) >= neg_quants[np.round(neg-0.20, 1)]
        #                                 ).Value & (self.xr_ar6.sel(Variable = ['Carbon Sequestration|CCS', 'Carbon Sequestration|Land Use'], Time=np.arange(2090, 2101)).sum(dim=['Time', 'Variable']) <= neg_quants[np.round(neg+0.20, 1)]).Value]
        #     for temp_i, temp in enumerate(self.Tlist):
        #         ms2 = self.xr_ar6.ModelScenario[(self.xr_ar6.sel(Variable = 'AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile', Time=2100) <= temp+0.5
        #                                         ).Value & (self.xr_ar6.sel(Variable = 'AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile', Time=2100) >= temp-0.5).Value]
        #         for risk_i, risk in enumerate(self.Plist):
        #             ms3 = self.xr_ar6.ModelScenario[(self.xr_ar6.sel(Variable = 'AR6 climate diagnostics|Exceedance Probability '+str(np.round(temp*2, 0)/2)+'C|MAGICCv7.5.3', Time=2100) <= risk).Value]

        #             # Based on parameters, get correct model scenario pairs for the functional form
        #             ms = np.intersect1d(np.array(ms1), np.array(ms2))
        #             ms = np.intersect1d(np.array(ms), np.array(ms3))
        #             if len(ms) < 5: ms = []

        #             # Create functional form and harmonize (2021 and budget)
        #             xset = xr_scen2_use.sel(ModelScenario = np.array(ms))
        #             offset = (startpoint-xset).sel(Time=self.settings['params']['start_year_analysis']).drop('Time')
        #             xset2 = xset+offset
        #             factor = (self.xr_budgets.Budget.sel(Temperature=temp, Risk=risk) - xset2.sum(dim='Time')/1e3 ) / np.sum(compensation_form)
        #             all_pathways = (1e3*(xset2/1e3+factor*xr_comp)).Value
        #             if len(all_pathways)>0:
        #                 for unc_i, unc in enumerate(['Earliest', 'Early', 'Medium', 'Late', 'Latest']):
        #                     pathway = np.array(all_pathways.where(all_pathways.sel(Time=2030) < np.percentile(all_pathways.sel(Time=2030), [10, 25, 50, 75, 90][unc_i], axis=0)).mean(dim='ModelScenario'))#np.percentile(all_pathways, [20, 50, 80][unc_i], axis=0)
        #                     pathways_data['GHG_globe'][neg_i, unc_i, temp_i, risk_i, :] = pathway.T
        # self.xr_traj = xr_traj.update(pathways_data)

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
            df_ssps.append(df_base)
        df_base_all = pd.concat(df_ssps)
        df_base_all = df_base_all.reset_index(drop=True)
        dummy = df_base_all.melt(id_vars=["Region", "Scenario"], var_name="Time", value_name="CO2_base")
        dummy['Time'] = np.array(dummy['Time'].astype(int))
        dummy = dummy.set_index(["Region", "Scenario", "Time"])
        xr_base = xr.Dataset.from_dataframe(dummy)
        xr_base = xr_base.reindex(Time = np.arange(self.settings['params']['start_year_analysis'], 2101))

        # Using an offset, get total CO2 emissions
        offset = self.xr_primap.sel(Time=self.settings['params']['start_year_analysis']).CO2_hist - xr_base.sel(Time=self.settings['params']['start_year_analysis']).CO2_base
        total_co2_base = xr_base + offset

        # Get offset of CO2 vs GHG emissions by country
        #fraction_ghg_co2_startyear = self.xr_primap.sel(Time=self.settings['params']['start_year_analysis']).GHG_hist / self.xr_primap.sel(Time=self.settings['params']['start_year_analysis']).CO2_hist
        offset_ghg_co2_startyear =  self.xr_primap.sel(Time=self.settings['params']['start_year_analysis']).GHG_hist - self.xr_primap.sel(Time=self.settings['params']['start_year_analysis']).CO2_hist

        # Convert baseline emissions into GHG using this fraction (or offset)
        total_ghg_base = total_co2_base+offset_ghg_co2_startyear#*fraction_ghg_co2_startyear
        total_ghg_base = total_ghg_base.rename({'CO2_base': "GHG_base"})
        self.xr_base = total_ghg_base

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

    # =========================================================== #
    # =========================================================== #

    def merge_xr(self):
        print('- Merging xrarray object')
        xr_total = xr.merge([self.xr_ssp, self.xr_primap, self.xr_unp, self.xr_budgets, self.xr_traj, self.xr_base, self.xr_ndc])
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
        self.xr_total.to_netcdf(self.settings['paths']['data']['datadrive']+'xr_dataread.nc',
            encoding={
                "Region": {"dtype": "str"},
                "Scenario": {"dtype": "str"},
                "Time": {"dtype": "int"},

                "Temperature": {"dtype": "float"},
                "NegEmis": {"dtype": "float"},
                "Risk": {"dtype": "float"},
                "TrajUnc": {"dtype": "str"},

                "Conditionality": {"dtype": "str"},
                "Hot_air": {"dtype": "str"},
                "Ambition": {"dtype": "str"},

                "GDP": {"zlib": True, "complevel": 9},
                "Population": {"zlib": True, "complevel": 9},
                "GHG_hist": {"zlib": True, "complevel": 9},
                "GHG_globe": {"zlib": True, "complevel": 9},
                "GHG_base": {"zlib": True, "complevel": 9},
                "GHG_ndc": {"zlib": True, "complevel": 9},
            },
            format="NETCDF4",
            engine="netcdf4",
        )

        print('- Some pre-calculations for the AP allocation rule')
        xrt = self.xr_total.sel(Time=np.arange(self.settings['params']['start_year_analysis'], 2101))
        r1_nom = (xrt.GDP.sel(Region=self.countries_iso).sum(dim='Region') / xrt.Population.sel(Region=self.countries_iso).sum(dim='Region'))
        base_worldsum = xrt.GHG_base.sel(Region=self.countries_iso).sum(dim='Region')
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