# ======================================== #
# Class that does the budget allocation
# ======================================== #

# =========================================================== #
# PREAMBULE
# Put in packages that we need
# =========================================================== #

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

# =========================================================== #
# CLASS OBJECT
# =========================================================== #


class allocation_comb:
    # =========================================================== #
    # =========================================================== #

    def __init__(self, dataread_file="xr_dataread.nc"):
        self.current_dir = Path.cwd()

        # Read in Input YAML file
        with open(self.current_dir / "config.yml") as file:
            self.settings = yaml.load(file, Loader=yaml.FullLoader)
        self.countries_iso = np.load(
            self.settings["paths"]["output"] + "/all_countries.npy", allow_pickle=True
        )
        self.savepath = (
            self.settings["paths"]["output"]
            + "/startyear_"
            + str(self.settings["params"]["start_year_analysis"])
            + "/"
        )
        self.xr_dataread = (
            xr.open_dataset(self.savepath + dataread_file)
            .load()
            .sel(
                Temperature=[1.6, 2.0],
                Risk=[0.5, 0.33],
                NonCO2red=0.5,
                Timing="Immediate",
                NegEmis=0.5,
            )
        )
        self.dataread_file = dataread_file

        # Region and Time variables
        self.start_year_analysis = self.settings["params"]["start_year_analysis"]
        self.analysis_timeframe = np.arange(self.start_year_analysis, 2101)

        # Set emission variables -> There is a single default because of the link to Robiou paper
        self.emis_hist = self.xr_dataread.GHG_hist_excl
        self.emis_fut = self.xr_dataread.GHG_excl_C - self.xr_dataread.CO2_bunkers_C
        self.varhist = "GHG_hist_excl"
        self.CO2_neg = self.xr_dataread.CO2_neg_C
        self.rbw = xr.open_dataset(self.savepath + "xr_rbw_GHG_excl.nc").load()
        self.all_future_years = np.arange(self.settings["params"]["start_year_analysis"], 2101)
        self.start_year_analysis = self.settings["params"]["start_year_analysis"]

    # =========================================================== #
    # =========================================================== #

    def discounting_historical_emissions(self):
        """
        ECPC computation
        """

        hist_emissions_startyears = [1850, 1950, 1990]
        discount_rates = [0, 1.6, 2.0, 2.8]
        xrs = []
        for hist_emissions_startyear in hist_emissions_startyears:
            for discount_rate in discount_rates:
                hist_emissions = self.emis_hist.sel(
                    Time=np.arange(
                        hist_emissions_startyear, self.settings["params"]["start_year_analysis"]
                    )
                )
                hist_emissions_discounted = (
                    hist_emissions
                    * (
                        (1 - discount_rate / 100)
                        ** (
                            1
                            + np.arange(
                                0,
                                self.settings["params"]["start_year_analysis"]
                                - hist_emissions_startyear,
                            )
                        )
                    )[::-1]
                ).expand_dims(
                    Discount_factor=[discount_rate], Historical_startyear=[hist_emissions_startyear]
                )
                xrs.append(hist_emissions_discounted)
        self.historical_emissions_discounted = xr.merge(xrs)

    # =========================================================== #
    # =========================================================== #

    def ecpc(self):
        """
        ECPC computation
        """
        compensation_form_sqrt = np.sqrt(
            np.arange(0, 2101 - self.settings["params"]["start_year_analysis"])
        )  # make sqrt curve
        compensation_form_sqrt = compensation_form_sqrt / np.sum(
            compensation_form_sqrt
        )  # sum of values has to be 1

        xr_comp = xr.DataArray(
            compensation_form_sqrt, dims=["Time"], coords={"Time": self.analysis_timeframe}
        )

        # Defining the timeframes for historical and future emissions
        xrs = []
        hist_emissions_startyears = [1850, 1950, 1990]
        for startyear_i, startyear in enumerate(hist_emissions_startyears):
            hist_emissions_timeframe = np.arange(startyear, 1 + self.start_year_analysis)
            future_emissions_timeframe = np.arange(self.start_year_analysis + 1, 2101)

            # Summing all historical emissions over the hist_emissions_timeframe
            hist_emissions = self.emis_hist.sel(Time=hist_emissions_timeframe)

            # Discounting -> We only do past discounting here
            for discount_i, discount in enumerate([0, 1.6, 2.0, 2.8]):
                past_timeline = np.arange(startyear, self.start_year_analysis + 1)
                xr_dc = xr.DataArray(
                    (1 - discount / 100) ** (self.start_year_analysis - past_timeline),
                    dims=["Time"],
                    coords={"Time": past_timeline},
                )
                hist_emissions_dc = (hist_emissions * xr_dc).sum(dim="Time")
                hist_emissions_w = float(hist_emissions_dc.sel(Region="EARTH"))
                hist_emissions_r = np.array(hist_emissions_dc)

                # Summing all future emissions over the future_emissions_timeframe
                future_emissions_w = self.emis_fut.sel(Time=future_emissions_timeframe).sum(
                    dim="Time"
                )

                total_emissions_w = hist_emissions_w + future_emissions_w

                # Calculating the cumulative population shares for region and world
                cum_pop = self.xr_dataread.Population.sel(Time=self.analysis_timeframe).sum(
                    dim="Time"
                )
                cum_pop_r = cum_pop
                cum_pop_w = cum_pop.sel(Region="EARTH")
                share_cum_pop = cum_pop_r / cum_pop_w
                budget_rightful = total_emissions_w * share_cum_pop
                budget_left = budget_rightful - hist_emissions_r

                # Now temporal allocation
                # globalbudget = self.xr_total.GHG_globe.sel(Time=self.analysis_timeframe).sum(dim='Time')
                globalpath = self.emis_fut

                emis_start_i = self.emis_hist.sel(Time=self.start_year_analysis)
                emis_start_w = self.emis_hist.sel(Time=self.start_year_analysis, Region="EARTH")
                path_scaled_0 = emis_start_i / emis_start_w * globalpath
                budget_without_assumptions = path_scaled_0.sum(dim="Time")
                budget_surplus = budget_left - budget_without_assumptions

                def ecpc_factor(f):
                    return path_scaled_0 + xr_comp * f

                ecpc = (
                    ecpc_factor(budget_surplus)
                    .expand_dims(Discount_factor=[discount], Historical_startyear=[startyear])
                    .to_dataset(name="ECPC")
                )
                xrs.append(ecpc)
        self.xr_ecpc = xr.merge(xrs)

    # =========================================================== #
    # =========================================================== #

    def pc(self):
        """
        ECPC computation
        """
        self.xr_pc = (
            self.emis_fut
            * self.xr_dataread.Population
            / self.xr_dataread.Population.sel(Region="EARTH")
        ).to_dataset(name="PC")

    # =========================================================== #
    # =========================================================== #

    def approach1gdp(self):
        """
        Methods for Robiou et al. (2023), under review.
        """
        yearly_netto = self.emis_fut.sel(Time=self.all_future_years)
        yearly_neg = self.CO2_neg.sel(
            Time=self.all_future_years
        )  # This should be hard-coded CO2_neg_globe
        yearly_pos = yearly_neg + yearly_netto

        app1_gdp = self.xr_dataread.sel(Time=self.all_future_years).GDP
        app1_gdp = app1_gdp.where(
            ~app1_gdp.Region.isin(["COK", "VAT", "NIU", "SOM", "GMB", "LIE", "PSE", "MCO", "NRU"]),
            np.nan,
        )
        app1_gdp_neg = (
            app1_gdp / app1_gdp.sel(Region="EARTH", Time=self.all_future_years) * yearly_neg
        )
        app1_gdp_pos_total = self.xr_ecpc.ECPC.sum(dim="Time") + app1_gdp_neg.sum(dim="Time")
        app1_gdp_pos = app1_gdp_pos_total / app1_gdp_pos_total.sel(Region="EARTH") * yearly_pos
        self.COMB1g = app1_gdp_pos - app1_gdp_neg

    # =========================================================== #
    # =========================================================== #

    def approach1hdi(self):
        """
        Methods for Robiou et al. (2023), under review.
        """
        yearly_netto = self.emis_fut.sel(Time=self.all_future_years)
        yearly_neg = self.CO2_neg.sel(
            Time=self.all_future_years
        )  # This should be hard-coded CO2_neg_globe
        yearly_pos = yearly_neg + yearly_netto

        app1_hdi = self.xr_dataread.sel(Time=self.all_future_years).HDIsh
        app1_hdi = app1_hdi.where(
            ~app1_hdi.Region.isin(["COK", "VAT", "NIU", "SOM", "GMB", "LIE", "PSE", "MCO", "NRU"]),
            np.nan,
        )
        app1_hdi_neg = app1_hdi / np.sum(app1_hdi.sel(Region=self.countries_iso)) * yearly_neg
        app1_hdi_pos_total = self.xr_ecpc.ECPC.sum(dim="Time") + app1_hdi_neg.sum(dim="Time")
        app1_hdi_pos = app1_hdi_pos_total / app1_hdi_pos_total.sel(Region="EARTH") * yearly_pos
        self.COMB1h = app1_hdi_pos - app1_hdi_neg

    # =========================================================== #
    # =========================================================== #

    def approach2(self):
        """
        Methods for Robiou et al. (2023), under review.
        """
        yearly_netto = self.emis_fut.sel(Time=self.all_future_years)
        yearly_neg = self.CO2_neg.sel(
            Time=self.all_future_years
        )  # This should be hard-coded CO2_neg_globe
        yearly_pos = yearly_neg + yearly_netto

        pop2gdp = (self.xr_dataread.Population**2 / self.xr_dataread.GDP).sel(
            Time=self.all_future_years
        )
        app2_pos_shares = pop2gdp.where(
            ~pop2gdp.Region.isin(
                [
                    "AND",
                    "ATG",
                    "DMA",
                    "GRD",
                    "KIR",
                    "MHL",
                    "FSM",
                    "MCO",
                    "NRU",
                    "PRK",
                    "PLW",
                    "KNA",
                    "SMR",
                    "SYC",
                    "SSD",
                    "TUV",
                    "COK",
                    "VAT",
                    "NIU",
                    "SOM",
                    "GMB",
                    "LIE",
                    "PSE",
                ]
            ),
            np.nan,
        )
        app2_pos_shares = app2_pos_shares / app2_pos_shares.sel(
            Region=self.countries_iso, Time=self.all_future_years
        ).sum(dim="Region")
        app2_pos_shares = app2_pos_shares
        app2_current_debt = self.historical_emissions_discounted.sum(dim="Time")
        app2_current_debt = app2_current_debt.where(
            ~app2_current_debt.Region.isin(
                [
                    "AND",
                    "ATG",
                    "DMA",
                    "GRD",
                    "KIR",
                    "MHL",
                    "FSM",
                    "MCO",
                    "NRU",
                    "PRK",
                    "PLW",
                    "KNA",
                    "SMR",
                    "SYC",
                    "SSD",
                    "TUV",
                    "COK",
                    "VAT",
                    "NIU",
                    "SOM",
                    "GMB",
                    "LIE",
                    "PSE",
                ]
            ),
            np.nan,
        )
        app2_nets = []
        app2_negs = []
        app2_poss = []
        for y_i, y in enumerate(self.all_future_years):
            neg = (
                yearly_neg.sel(Time=y) * (app2_current_debt / app2_current_debt.sel(Region="EARTH"))
            ).drop_vars(["Time"])
            pos = (yearly_pos.sel(Time=y) * app2_pos_shares.sel(Time=y)).drop_vars(["Time"])
            app2_nets.append((pos - neg).expand_dims(Time=[y]))
            app2_current_debt = app2_current_debt + pos - neg
            app2_negs.append(neg.expand_dims(Time=[y]))
            app2_poss.append(pos.expand_dims(Time=[y]))
        self.COMB2 = xr.concat(app2_nets, dim="Time")

    # =========================================================== #
    # =========================================================== #

    def approach2_transition(self):
        """
        Methods for Robiou et al. (2023), under review.
        """
        yearly_netto = self.emis_fut.sel(Time=self.all_future_years)
        yearly_neg = self.CO2_neg.sel(
            Time=self.all_future_years
        )  # This should be hard-coded CO2_neg_globe
        yearly_pos = yearly_neg + yearly_netto

        self.gf = (
            self.emis_hist.sel(Time=self.settings["params"]["start_year_analysis"])
            / self.emis_hist.sel(
                Time=self.settings["params"]["start_year_analysis"], Region="EARTH"
            )
            * yearly_netto
        )
        self.pc = (
            self.xr_dataread.Population
            / self.xr_dataread.Population.sel(Region="EARTH")
            * yearly_netto
        )
        gf_f = self.gf / yearly_netto
        pc_f = self.pc / yearly_netto

        if self.start_year_analysis == 2021: self.convergence_moment = 2040
        elif self.start_year_analysis == 2015: self.convergence_moment = 2035
        yearfracs_2 = xr.Dataset(
            data_vars={
                "Value": (
                    ["Time"],
                    (self.all_future_years - self.settings["params"]["start_year_analysis"])
                    / (self.convergence_moment - self.settings["params"]["start_year_analysis"]),
                )
            },
            coords={"Time": self.all_future_years},
        )
        f2_f = (self.COMB2 / yearly_netto)[self.varhist]
        f2c = self.COMB2[self.varhist].copy()
        self.COMB2t = (
            f2c.where(
                f2c.Time > self.convergence_moment,
                (gf_f * (1 - yearfracs_2) + f2_f * yearfracs_2) * yearly_netto,
            )
        ).expand_dims(Convergence_year=[self.convergence_moment])

        """ Add country groups """
        path_ctygroups = "X:/user/dekkerm/Data/" + "UNFCCC_Parties_Groups_noeu.xlsx"
        df = pd.read_excel(path_ctygroups, sheet_name="Country groups")
        countries_iso = np.array(df["Country ISO Code"])
        list_of_regions = list(np.array(self.xr_dataread.Region).copy())
        for group_of_choice in [
            "G20",
            "EU",
            "G7",
            "SIDS",
            "LDC",
            "Northern America",
            "Australasia",
            "African Group",
            "Umbrella",
        ]:
            group_indices = countries_iso[np.array(df[group_of_choice]) == 1]
            country_to_eu = {}
            for cty in np.array(self.xr_dataread.Region):
                if cty in group_indices:
                    country_to_eu[cty] = [group_of_choice]
                else:
                    country_to_eu[cty] = [""]
            group_coord = xr.DataArray(
                [
                    group
                    for country in np.array(self.xr_dataread["Region"])
                    for group in country_to_eu[country]
                ],
                dims=["Region"],
                coords={
                    "Region": [
                        country
                        for country in np.array(self.xr_dataread["Region"])
                        for group in country_to_eu[country]
                    ]
                },
            )
            if group_of_choice == "EU":
                xr_eu = self.COMB2t[["Value"]].groupby(group_coord).sum()  # skipna=False)
            else:
                xr_eu = self.COMB2t[["Value"]].groupby(group_coord).sum(skipna=False)
            xr_eu2 = xr_eu.rename({"group": "Region"})
            xr_eu2 = xr_eu2.sel(Region=[x for x in np.array(xr_eu2.Region) if x != ""])
            dummy = self.COMB2t.reindex(Region=[x for x in list_of_regions if x != group_of_choice])
            self.COMB2t = xr.merge([dummy, xr_eu2])
            self.COMB2t = self.COMB2t.reindex(Region=list_of_regions)
        self.COMB2t = self.COMB2t

    # =========================================================== #
    # =========================================================== #

    def combine(self):
        self.xr_combs = xr.merge(
            [
                self.COMB1g.to_dataset(name="Approach1_gdp"),
                self.COMB1h.to_dataset(name="Approach1_hdi"),
                self.COMB2[self.varhist].to_dataset(name="Approach2"),
                self.COMB2t.rename({"Value": "Approach2t"}),
                self.gf.to_dataset(name="GF"),
                self.xr_ecpc,
                self.xr_pc,
            ]
        )

    # =========================================================== #
    # =========================================================== #

    def get_relation_2030emis_temp(self):
        df_ar6_2 = pd.read_csv(
            self.settings["paths"]["data"]["external"]
            + "IPCC/AR6_Scenarios_Database_World_v1.1.csv"
        )
        df_ar6_2 = df_ar6_2[
            df_ar6_2.Variable.isin(
                [
                    "Emissions|Kyoto Gases",
                    "AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|5.0th Percentile",
                    "AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile",
                    "AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|95.0th Percentile",
                ]
            )
        ]
        mods = np.array(df_ar6_2.Model)
        scens = np.array(df_ar6_2.Scenario)
        modscens = np.array([mods[i] + "|" + scens[i] for i in range(len(scens))])
        df_ar6_2["ModelScenario"] = modscens
        df_ar6_2 = df_ar6_2.drop(["Model", "Scenario", "Region", "Unit"], axis=1)
        dummy = df_ar6_2.melt(
            id_vars=["ModelScenario", "Variable"], var_name="Time", value_name="Value"
        )
        dummy["Time"] = np.array(dummy["Time"].astype(int))
        dummy = dummy.set_index(["ModelScenario", "Variable", "Time"])
        xr_ar6_2 = xr.Dataset.from_dataframe(dummy)
        x_data = xr_ar6_2.sel(
            Time=2030, Variable="Emissions|Kyoto Gases"
        ).Value  # Technically, this is GHG incl LULUCF
        y_data = xr_ar6_2.sel(
            Variable="AR6 climate diagnostics|Surface Temperature (GSAT)|MAGICCv7.5.3|50.0th Percentile"
        ).Value.max(dim="Time")  # Peak temperature!
        mask = ~np.isnan(y_data)
        x_fit = x_data[mask]
        y_fit = y_data[mask]
        mask = ~np.isnan(x_fit)
        x_fit = x_fit[mask]
        y_fit = y_fit[mask]
        self.coef_ghg_2030 = np.polyfit(
            x_fit, y_fit, self.settings["params"]["polynomial_fit_2030relation"]
        )

    # =========================================================== #
    # =========================================================== #

    def determine_tempoutcomes_regression(self):
        rules = ["Approach1_gdp", "Approach1_hdi", "Approach2", "Approach2t", "GF", "PC"]
        percs = self.xr_combs[rules] / self.xr_combs.sel(Region=self.countries_iso).GF.sum(
            dim="Region"
        )
        condition = percs < 0
        percs = percs.where(~condition, 1e-9)
        ndc_globalversion_raw = self.xr_dataread.GHG_ndc_excl_CR / percs
        condition = ndc_globalversion_raw < 10000
        mod_data = ndc_globalversion_raw.where(~condition, 10000)
        condition = mod_data > 75000
        ndc_globalversion = mod_data.where(~condition, 75000)
        for n in range(
            self.settings["params"]["polynomial_fit_2030relation"] + 1
        ):  # self.coef_ghg_2030[0]*x_fit**5+ self.coef_ghg_2030[1]*x_fit**4+ self.coef_ghg_2030[2]*x_fit**3 +self.coef_ghg_2030[3]*x_fit**2 +self.coef_ghg_2030[4]*x_fit**1 + self.coef_ghg_2030[5]
            if n == 0:
                xr_temps = (
                    ndc_globalversion**n
                    * self.coef_ghg_2030[self.settings["params"]["polynomial_fit_2030relation"] - n]
                )
            else:
                xr_temps += (
                    ndc_globalversion**n
                    * self.coef_ghg_2030[self.settings["params"]["polynomial_fit_2030relation"] - n]
                )
        self.xr_temps = xr_temps

    # =========================================================== #
    # =========================================================== #

    def determine_tempoutcomes_discrete(self):
        rules = ["Approach1_gdp", "Approach1_hdi", "Approach2", "Approach2t", "GF", "PC"]
        results = self.xr_combs[rules]
        xr_eval_raw = results.where(
            results
            > self.xr_dataread.GHG_ndc_excl_CR.sel(
                Conditionality="unconditional", Hot_air="include"
            ).mean(dim="Ambition"),
            np.nan,
        )
        xr_eval_raw = xr_eval_raw.where(xr_eval_raw > -1e9, other=0)
        xr_eval_raw = xr_eval_raw.where(xr_eval_raw == 0, other=1)

        # Now without C1+C2:
        xr_eval = xr_eval_raw.sel(Category=["C1", "C2", "C3", "C6", "C7"]).sum(dim="Category")
        xr_eval = xr_eval.where(xr_eval != 5, other=0.01)
        xr_eval = xr_eval.where(xr_eval != 4, other=1.01)
        xr_eval = xr_eval.where(xr_eval != 3, other=2.01)
        xr_eval = xr_eval.where(xr_eval != 2, other=3.01)
        xr_eval = xr_eval.where(xr_eval != 1, other=4.01)
        xr_eval = xr_eval.where(xr_eval != 0, other=5.01)
        self.xr_temps = xr_eval

        # And with C1+C2:
        xr_eval = xr_eval_raw.sel(Category=["C1+C2", "C3", "C6", "C7"]).sum(dim="Category")
        xr_eval = xr_eval.where(xr_eval != 4, other=1.01)
        xr_eval = xr_eval.where(xr_eval != 3, other=2.01)
        xr_eval = xr_eval.where(xr_eval != 2, other=3.01)
        xr_eval = xr_eval.where(xr_eval != 1, other=4.01)
        xr_eval = xr_eval.where(xr_eval != 0, other=5.01)
        self.xr_temps_c12 = xr_eval

    # =========================================================== #
    # =========================================================== #

    def save(self):
        self.xr_combs.to_netcdf(self.savepath + "xr_comb.nc", format="NETCDF4")
        self.xr_temps.to_netcdf(self.savepath + "xr_combtemps.nc", format="NETCDF4")
        self.xr_temps_c12.to_netcdf(self.savepath + "xr_combtemps_c12.nc", format="NETCDF4")
