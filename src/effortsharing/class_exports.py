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
from tqdm import tqdm

# =========================================================== #
# CLASS OBJECT
# =========================================================== #


class dataexportcl(object):
    # =========================================================== #
    # =========================================================== #

    def __init__(
        self, input_file="../../input.yml", settings_file="./export_settings.yml"
    ):
        print("# ==================================== #")
        print("# DATAREADING class                    #")
        self.current_dir = Path.cwd()

        # Read in export settings YAML file and Input YAML file
        with open(settings_file) as file:
            self.export_settings = yaml.load(file, Loader=yaml.FullLoader)
        with open(input_file) as file:
            self.settings = yaml.load(file, Loader=yaml.FullLoader)

        # Set up main data objects
        self.savepath = (
            self.settings["paths"]["data"]["export"]
            + "startyear_"
            + str(self.settings["params"]["start_year_analysis"])
            + "/"
        )
        self.xr_dataread = xr.open_dataset(
            self.settings["paths"]["data"]["datadrive"]
            + "startyear_"
            + str(self.settings["params"]["start_year_analysis"])
            + "/"
            + "xr_dataread.nc"
        ).load()
        self.countries_iso = np.load(
            self.settings["paths"]["data"]["datadrive"] + "all_countries.npy",
            allow_pickle=True,
        )
        print("# startyear: ", self.settings["params"]["start_year_analysis"])
        print("# ==================================== #")

    # ====================================================================================================================== #
    # GLOBAL EMISSION PATHWAYS EXPORTS
    # ====================================================================================================================== #

    def global_default(self):
        """
        Export default 1.5(6) and 2.0 pathways that roughly match the IPCC pathways
        """
        dataframe = self.xr_dataread.sel(
            Time=np.arange(self.settings["params"]["start_year_analysis"], 2101),
            **self.export_settings["dimensions_global"],
        )[["GHG_globe", "CO2_globe", "NonCO2_globe"]].to_dataframe()
        dataframe.columns = pd.MultiIndex.from_tuples(
            [
                ["GHG_globe", "Mt CO2e/yr"],
                ["CO2_globe", "Mt CO2/yr"],
                ["NonCO2_globe", "Mt CO2e/yr"],
            ]
        )
        dataframe.reset_index(inplace=True)
        dataframe.to_csv(self.savepath + "emissionspathways_default.csv", index=False)

    # =========================================================== #
    # =========================================================== #

    def negative_nonlulucf_emissions(self):
        """
        Export negative emissions pathways
        """
        dataframe = self.xr_dataread.sel(
            Time=np.arange(self.settings["params"]["start_year_analysis"], 2101),
            **self.export_settings["dimensions_global"],
        )[["CO2_neg_globe"]].to_dataframe()
        dataframe.columns = pd.MultiIndex.from_tuples([["CO2_neg_globe", "Mt CO2/yr"]])
        dataframe.reset_index(inplace=True)
        dataframe.to_csv(self.savepath + "emissionspathways_co2neg.csv", index=False)

    # =========================================================== #
    # =========================================================== #

    def global_all(self):
        """
        Export a large set of pathways (still a subset)
        """
        dataframe = self.xr_dataread.sel(
            Time=np.arange(self.settings["params"]["start_year_analysis"], 2101),
        )[["GHG_globe", "CO2_globe", "GHG_globe_excl", "CO2_globe_excl"]].to_dataframe()
        dataframe.columns = pd.MultiIndex.from_tuples(
            [
                ["GHG_globe", "Mt CO2e/yr"],
                ["CO2_globe", "Mt CO2/yr"],
                ["GHG_globe_excl", "Mt CO2e/yr"],
                ["CO2_globe_excl", "Mt CO2/yr"],
            ]
        )
        dataframe.reset_index(inplace=True)
        dataframe.to_csv(self.savepath + "emissionspathways_all.csv", index=False)

    # ====================================================================================================================== #
    # INPUT DATA EXPORTS
    # ====================================================================================================================== #

    def ndcdata(self):
        """
        Export NDC data
        """
        dataframe = self.xr_dataread.sel(
            Time=np.arange(self.settings["params"]["start_year_analysis"], 2101),
        )[["GHG_ndc"]].to_dataframe()
        dataframe.columns = pd.MultiIndex.from_tuples([("GHG_ndc", "(Mt CO2e/yr)")])
        dataframe.reset_index(inplace=True)
        dataframe.to_csv(self.savepath + "inputdata_ndc.csv", index=False)

    # =========================================================== #
    # =========================================================== #

    def sspdata(self):
        """
        Export SSP data
        """
        dataframe = self.xr_dataread.sel(
            Time=np.arange(self.settings["params"]["start_year_analysis"], 2101),
        )[["Population", "GDP"]].to_dataframe()
        dataframe.columns = pd.MultiIndex.from_tuples(
            [("Population", "Inhabitants"), ("GDP", "PPP, billion USD_2017/yr")]
        )
        dataframe.reset_index(inplace=True)
        dataframe.to_csv(self.savepath + "inputdata_ssp.csv", index=False)

    # =========================================================== #
    # =========================================================== #

    def emisdata(self):
        """
        Export historical emission data
        """
        dataframe = self.xr_dataread.sel(
            Time=np.arange(1850, 1 + self.settings["params"]["start_year_analysis"]),
        )[
            ["GHG_hist", "GHG_hist_excl", "CO2_hist", "CH4_hist", "N2O_hist"]
        ].to_dataframe()
        dataframe.columns = pd.MultiIndex.from_tuples(
            [
                ("GHG_hist", "Mt CO2e/yr"),
                ("GHG_hist_excl", "Mt CO2e/yr"),
                ("CO2_hist", "Mt CO2/yr"),
                ("CH4_hist", "Mt CO2e/yr"),
                ("N2O_hist", "Mt CO2e/yr"),
            ]
        )
        dataframe.reset_index(inplace=True)
        dataframe.to_csv(self.savepath + "inputdata_histemis.csv", index=False)

    # ====================================================================================================================== #
    # ALLOCATIONS DATA EXPORTS
    # ====================================================================================================================== #

    def reduce_country_files(self):
        """
        Get reduced-form country files, omitting some parameter settings that users won't use and reducing the file size through compression
        """
        path_toread = (
            self.savepath
            + "../../DataUpdate_ongoing/startyear_2021/Allocations_GHG_incl/"
        )
        path_tosave = self.savepath + "Allocations_GHG_incl_reduced/"

        for cty_i, cty in tqdm(enumerate(np.array(self.xr_dataread.Region))):
            ds = (
                xr.open_dataset(path_toread + "xr_alloc_" + cty + ".nc")
                .sel(
                    Time=self.export_settings["time_axis"],
                    **self.export_settings["dimensions_global"],
                    **self.export_settings["dimension_rules"],
                )
                .expand_dims(Region=[cty])
                .load()
            )
            ds = ds.drop_vars(["PCB"])
            ds.to_netcdf(
                path_tosave + "/reduced_allocations_" + cty + ".nc",
                encoding={
                    "GF": {"zlib": True, "complevel": 9},
                    "PC": {"zlib": True, "complevel": 9},
                    "PCC": {"zlib": True, "complevel": 9},
                    "PCB_lin": {"zlib": True, "complevel": 9},
                    "GDR": {"zlib": True, "complevel": 9},
                    "ECPC": {"zlib": True, "complevel": 9},
                    "AP": {"zlib": True, "complevel": 9},
                },
                format="NETCDF4",
                engine="netcdf4",
            )
            ds.close()

    # =========================================================== #
    # =========================================================== #

    def allocations_default(self):
        """
        Export default emission allocations and reductions
        """
        path_toread = self.savepath + "Allocations_GHG_incl_reduced/"

        for default_i in range(2):
            dss = []
            for cty in np.array(self.xr_dataread.Region):
                param_set = [
                    self.export_settings["default_15"],
                    self.export_settings["default_20"],
                ][default_i]
                ds = (
                    xr.open_dataset(path_toread + "reduced_allocations_" + cty + ".nc")
                    .sel(
                        Time=self.export_settings["time_axis"],
                        **param_set,
                        Discount_factor=0,
                        Historical_startyear=1990,
                        Scenario="SSP2",
                        Convergence_year=2050,
                        # **self.export_settings['default_rules']
                    )
                    .drop_vars(
                        [
                            "Capability_threshold",
                            "RCI_weight",
                            "Scenario",
                            "Convergence_year",
                            "Discount_factor",
                            "Historical_startyear",
                            "NegEmis",
                            "Temperature",
                            "Risk",
                            "NonCO2red",
                            "Timing",
                        ]
                    )
                )  # Drop all constant variables that are not varied in this dataset
                dss.append(ds)
                ds.close()
            ds_total = xr.merge(dss)
            allocations_df = ds_total[
                ["GF", "PC", "PCC", "ECPC", "AP", "GDR", "PCB_lin"]
            ].to_dataframe()
            allocations_df.columns = pd.MultiIndex.from_tuples(
                [
                    (
                        "GF",
                        "Mt CO2e/yr",
                    ),  # ALWAYS double check ordering -> it is not always the same. Maybe we should build in some automatic detection.
                    ("PC", "Mt CO2e/yr"),
                    ("PCC", "Mt CO2/yr"),
                    ("ECPC", "Mt CO2e/yr"),
                    ("AP", "Mt CO2e/yr"),
                    ("GDR", "Mt CO2e/yr"),
                    ("PCB_lin", "Mt CO2e/yr"),
                ]
            )
            allocations_df.reset_index(inplace=True)
            allocations_df.to_csv(
                self.savepath
                + "allocations_default_"
                + ["15overshoot", "20"][default_i]
                + ".csv",
                index=False,
            )
            if default_i == 0:
                self.allocations_df = allocations_df

            cur = self.xr_dataread.GHG_hist.sel(Time=2015)
            reductions_df = (-(cur - ds_total) / cur)[
                ["GF", "PC", "PCC", "ECPC", "AP", "GDR", "PCB_lin"]
            ].to_dataframe()
            reductions_df.columns = pd.MultiIndex.from_tuples(
                [
                    ("GF", "% w.r.t. 2015"),
                    ("PC", "% w.r.t. 2015"),
                    ("PCC", "% w.r.t. 2015"),
                    ("ECPC", "% w.r.t. 2015"),
                    ("AP", "% w.r.t. 2015"),
                    ("GDR", "% w.r.t. 2015"),
                    ("PCB_lin", "% w.r.t. 2015"),
                ]
            )
            reductions_df.reset_index(inplace=True)
            reductions_df.to_csv(
                self.savepath
                + "reductions_default_"
                + ["15overshoot", "20"][default_i]
                + ".csv",
                index=False,
            )

    # ====================================================================================================================== #
    # BUDGETS
    # ====================================================================================================================== #

    def budgets_key_variables(self, lulucf="incl"):
        """
        Specify several key variables for the computation of budgets
        Note that budgets are only in CO2, not in GHG (while most of the alloations are in GHG)
        """
        self.xr_dataread_forbudgets = self.xr_dataread.sel(
            **self.export_settings["dimensions_global"]
        )
        if lulucf == "incl":
            self.emis_hist = self.xr_dataread_forbudgets.CO2_hist
            self.emis_fut = self.xr_dataread_forbudgets.CO2_globe
            self.emis_base = self.xr_dataread_forbudgets.CO2_base_incl
            self.budgets = self.xr_dataread_forbudgets.Budget
            self.xr_rbw = (
                xr.open_dataset(
                    self.settings["paths"]["data"]["datadrive"]
                    + "startyear_"
                    + str(self.settings["params"]["start_year_analysis"])
                    + "/"
                    + "xr_rbw_co2_incl.nc"
                )
                .load()
                .sel(**self.export_settings["dimensions_global"])
            )
        elif lulucf == "excl":
            self.emis_hist = self.xr_dataread_forbudgets.CO2_hist_excl
            self.emis_fut = self.xr_dataread_forbudgets.CO2_globe_excl
            self.emis_base = self.xr_dataread_forbudgets.CO2_base_excl
            temporalemis = self.xr_dataread_forbudgets.CO2_globe_excl
            temporalemis = temporalemis.where(temporalemis > 0, 0)
            self.budgets = temporalemis.sum(dim="Time") / 1e3
            self.xr_rbw = (
                xr.open_dataset(
                    self.settings["paths"]["data"]["datadrive"]
                    + "startyear_"
                    + str(self.settings["params"]["start_year_analysis"])
                    + "/"
                    + "xr_rbw_co2_excl.nc"
                )
                .load()
                .sel(**self.export_settings["dimensions_global"])
            )

    # =========================================================== #
    # =========================================================== #

    def co2_budgets_ap(self):
        """
        CO2 budgets AP
        """
        xrt = self.xr_dataread_forbudgets.sel(
            Time=np.arange(self.settings["params"]["start_year_analysis"], 2101)
        )
        GDP_sum_w = xrt.GDP.sel(Region="EARTH")
        pop_sum_w = xrt.Population.sel(Region="EARTH")
        r1_nom = GDP_sum_w / pop_sum_w

        base_worldsum = self.emis_base.sel(
            Region="EARTH",
            Time=np.arange(self.settings["params"]["start_year_analysis"], 2101),
        )
        rb_part1 = (xrt.GDP / xrt.Population / r1_nom) ** (1 / 3.0)
        rb_part2 = (
            self.emis_base.sel(
                Time=np.arange(self.settings["params"]["start_year_analysis"], 2101)
            )
            * (
                base_worldsum
                - self.emis_fut.sel(
                    Time=np.arange(self.settings["params"]["start_year_analysis"], 2101)
                )
            )
            / base_worldsum
        )
        rb = rb_part1 * rb_part2

        # Step 2: Correction factor
        corr_factor = (1e-9 + self.xr_rbw.__xarray_dataarray_variable__) / (
            base_worldsum
            - self.emis_fut.sel(
                Time=np.arange(self.settings["params"]["start_year_analysis"], 2101)
            )
        )

        # Step 3: Budget after correction factor
        ap = self.emis_base - rb / corr_factor
        self.xr_ap = (
            (
                ap.sel(
                    Time=np.arange(self.settings["params"]["start_year_analysis"], 2101)
                )
                * xr.where(
                    self.emis_fut.sel(
                        Time=np.arange(
                            self.settings["params"]["start_year_analysis"], 2101
                        )
                    )
                    > 0,
                    1,
                    0,
                )
            )
            .to_dataset(name="AP")
            .sum(dim="Time")
        )  # .sel(Scenario='SSP2', NegEmis=0.5, NonCO2red=0.5, Timing='Immediate')

    # =========================================================== #
    # =========================================================== #

    def co2_budgets_pc(self):
        """
        CO2 budgets PC
        """
        pop_region = self.xr_dataread_forbudgets.sel(
            Time=self.settings["params"]["start_year_analysis"]
        ).Population
        pop_earth = self.xr_dataread_forbudgets.sel(
            Region="EARTH", Time=self.settings["params"]["start_year_analysis"]
        ).Population
        pop_fraction = pop_region / pop_earth
        self.xr_pc = (pop_fraction * self.budgets * 1e3).to_dataset(name="PC")

    # =========================================================== #
    # =========================================================== #

    def co2_budgets_ecpc(self):
        """
        CO2 budgets ECPC
        """
        hist_emissions_startyears = self.settings["dimension_ranges"][
            "hist_emissions_startyears"
        ]
        discount_rates = self.settings["dimension_ranges"]["discount_rates"]
        xrs = []
        for focusregion in tqdm(np.array(self.xr_dataread_forbudgets.Region)):
            # Defining the timeframes for historical and future emissions
            for startyear_i, startyear in enumerate(hist_emissions_startyears):
                hist_emissions_timeframe = np.arange(
                    startyear, 1 + self.settings["params"]["start_year_analysis"]
                )
                future_emissions_timeframe = np.arange(
                    self.settings["params"]["start_year_analysis"] + 1, 2101
                )

                # Summing all historical emissions over the hist_emissions_timeframe
                hist_emissions = self.emis_hist.sel(Time=hist_emissions_timeframe)

                # Discounting -> We only do past discounting here
                for discount_i, discount in enumerate(discount_rates):
                    past_timeline = np.arange(
                        startyear, self.settings["params"]["start_year_analysis"] + 1
                    )
                    xr_dc = xr.DataArray(
                        (1 - discount / 100)
                        ** (
                            self.settings["params"]["start_year_analysis"]
                            - past_timeline
                        ),
                        dims=["Time"],
                        coords={"Time": past_timeline},
                    )
                    hist_emissions_dc = (hist_emissions * xr_dc).sum(dim="Time")
                    hist_emissions_w = float(hist_emissions_dc.sel(Region="EARTH"))
                    hist_emissions_r = float(hist_emissions_dc.sel(Region=focusregion))

                    # CO2 budget
                    future_emissions_w = self.budgets * 1e3
                    total_emissions_w = hist_emissions_w + future_emissions_w

                    # Calculating the cumulative population shares for region and world
                    cum_pop = self.xr_dataread_forbudgets.Population.sel(
                        Time=np.arange(
                            self.settings["params"]["start_year_analysis"], 2101
                        )
                    ).sum(dim="Time")
                    cum_pop_r = cum_pop.sel(Region=focusregion)
                    cum_pop_w = cum_pop.sel(Region="EARTH")
                    share_cum_pop = cum_pop_r / cum_pop_w
                    budget_rightful = total_emissions_w * share_cum_pop
                    budget_left = budget_rightful - hist_emissions_r
                    ecpc = budget_left.to_dataset(name="ECPC")
                    xrs.append(
                        ecpc.expand_dims(
                            Region=[focusregion],
                            Discount_factor=[discount],
                            Historical_startyear=[startyear],
                        )
                    )
        self.xr_ecpc = xr.merge(xrs)

    # =========================================================== #
    # =========================================================== #

    def concat_co2budgets(self, lulucf="incl"):
        """
        CO2 budgets ECPC, AP and PC
        """
        self.xr_budgets = xr.merge([self.xr_pc, self.xr_ecpc, self.xr_ap])
        self.xr_budgets = xr.merge(
            [
                xr.where(
                    self.xr_budgets.sel(Region="EARTH").expand_dims(["Region"]),
                    self.xr_budgets.sel(Region=self.countries_iso).sum(dim="Region"),
                    0,
                ),
                self.xr_budgets.drop_sel(Region="EARTH"),
            ]
        )
        self.xr_budgets.to_netcdf(
            self.settings["paths"]["data"]["datadrive"]
            + "CO2budgets_"
            + lulucf
            + ".nc",
            format="NETCDF4",
            engine="netcdf4",
        )
        budgets_df = self.xr_budgets.drop_vars(["Time"]).to_dataframe()

        budgets_df.columns = pd.MultiIndex.from_tuples(
            [("PC", "Mt CO2"), ("ECPC", "Mt CO2"), ("AP", "Mt CO2")]
        )
        budgets_df.reset_index(inplace=True)
        budgets_df.to_csv(self.savepath + "CO2budgets_" + lulucf + ".csv")

    # ====================================================================================================================== #
    # PROJECT-SPECIFIC ALLOCATIONS
    # ====================================================================================================================== #

    def project_COMMITTED(self):
        """
        Export files for COMMITTED
        """
        # Pathways
        df = pd.read_csv(
            "K:/Data/Data_effortsharing/EffortSharingExports/allocations_default_15overshoot.csv"
        )
        df = df[["Time", "Region", "PCC", "ECPC", "AP"]]
        df["Temperature"] = ["1.5 deg at 50% with small overshoot"] * len(df)

        df2 = pd.read_csv(
            "K:/Data/Data_effortsharing/EffortSharingExports/allocations_default_20.csv"
        )
        df2 = df2[["Time", "Region", "PCC", "ECPC", "AP"]]
        df2["Temperature"] = ["2.0 deg at 67%"] * len(df2)

        df3 = pd.concat([df, df2])
        df3.to_csv(
            "K:/Data/Data_effortsharing/EffortSharingExports/allocations_COMMITTED.csv",
            index=False,
        )

        # Budgets
        xr_traj_16 = xr.open_dataset(
            self.settings["paths"]["data"]["datadrive"] + "/xr_traj_t16_r50.nc"
        )
        xr_traj_20 = xr.open_dataset(
            self.settings["paths"]["data"]["datadrive"] + "/xr_traj_t20_r67.nc"
        )
        self.ecpc = xr_traj_16.ECPC.sum(dim="Time")

    # =========================================================== #
    # =========================================================== #

    def project_DGIS(self):
        """
        Export files for DGIS
        """
        df = pd.read_csv(
            "K:/Data/Data_effortsharing/EffortSharingExports/allocations_default_15overshoot.csv"
        )
        df = df[["Time", "Region", "PCC", "ECPC", "AP"]]
        df["Temperature"] = ["1.5 deg at 50% with small overshoot"] * len(df)

        df2 = pd.read_csv(
            "K:/Data/Data_effortsharing/EffortSharingExports/allocations_default_20.csv"
        )
        df2 = df2[["Time", "Region", "PCC", "ECPC", "AP"]]
        df2["Temperature"] = ["2.0 deg at 67%"] * len(df2)

        df3 = pd.concat([df, df2])
        df3.to_csv(self.savepath + "allocations_DGIS.csv", index=False)

    # ====================================================================================================================== #
    # COUNTRY-SPECIFIC ALLOCATIONS (through harmonization data)
    # ====================================================================================================================== #

    def countr_to_csv(self, cty, adapt="", lulucf="incl", gas="GHG"):
        """
        Convert .nc to .csv for a specific country
        """
        ds = xr.open_dataset(
            self.settings["paths"]["data"]["datadrive"]
            + "/Allocations_"
            + gas
            + "_"
            + lulucf
            + "/xr_alloc_"
            + cty
            + adapt
            + ".nc"
        ).sel(
            Temperature=[1.5, 1.6, 2.0],
            Risk=[0.5, 0.33, 0.17],
            Time=[2021] + list(np.arange(2025, 2101, 5)),
            Convergence_year=[2040, 2050, 2060, 2070],
        )
        # Export ds to csv
        for rule in ["GF", "PC", "PCC", "ECPC", "AP", "GDR", "PCB", "PCB_lin"]:
            ds[rule].to_dataframe().to_csv(
                "K:/Data/Data_effortsharing/EffortSharingExports/Country_CSVs/"
                + cty
                + "/"
                + gas
                + "_"
                + lulucf
                + "/xr_alloc_"
                + cty
                + adapt
                + "_"
                + rule
                + ".csv"
            )

    # =========================================================== #
    # =========================================================== #


if __name__ == "__main__":
    region = input("Choose a focus country or region: ")
    dataexporter = dataexportcl(region)
