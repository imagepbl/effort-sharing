import json

import numpy as np
import pandas as pd
import xarray as xr

from effortsharing.config import Config


def read_ndc_climateresource(config: Config, countries):
    print("- Reading NDC data from Climate resource")

    countries_iso = list(countries.values())
    version_ndcs = config.params.version_ndcs
    data_root = config.paths.input / f"ClimateResource_{version_ndcs}"

    ghg_data = np.zeros(shape=(len(countries_iso) + 1, 3, 2, 2, len(np.arange(2010, 2051))))
    for cty_i, cty in enumerate(countries_iso):
        for cond_i, cond in enumerate(["conditional", "range", "unconditional"]):
            for hot_i, hot in enumerate(["include", "exclude"]):
                for amb_i, amb in enumerate(["low", "high"]):
                    filename = f"{cty.lower()}_ndc_{version_ndcs}_CR_{cond}_{hot}.json"
                    path = data_root / cond / hot / filename
                    try:
                        with open(path) as file:
                            json_data = json.load(file)
                        country_name = json_data["results"]["country"]["name"]
                        series_items = json_data["results"]["series"]
                        for item in series_items:
                            columns = item["columns"]
                            if (
                                columns["variable"] == "Emissions|Total GHG excl. LULUCF"
                                and columns["category"] == "Updated NDC"
                                and columns["ambition"] == amb
                            ):
                                data = item["data"]
                                time_values = [int(year) for year in data.keys()]
                                ghg_values = np.array(list(item["data"].values()))
                                ghg_values[ghg_values == "None"] = np.nan
                                ghg_values = ghg_values.astype(float)
                                ghg_values = ghg_values[np.array(time_values) >= 2010]
                                ghg_data[cty_i, cond_i, hot_i, amb_i] = ghg_values
                                # series.append([country_iso.upper(), country_name, "Emissions|Total GHG excl. LULUCF", conditionality, hot_air, ambition] + list(ghg_values))
                    except:
                        continue

    # Now also for EU
    for cond_i, cond in enumerate(["conditional", "range", "unconditional"]):
        for hot_i, hot in enumerate(["include", "exclude"]):
            for amb_i, amb in enumerate(["low", "high"]):
                filename = f"groupeu27_ndc_{version_ndcs}_CR_{cond}_{hot}.json"
                path = data_root / cond / hot / "regions" / filename
                try:
                    with open(path) as file:
                        json_data = json.load(file)
                    country_name = json_data["results"]["country"]["name"]
                    series_items = json_data["results"]["series"]
                    for item in series_items:
                        columns = item["columns"]
                        if (
                            columns["variable"] == "Emissions|Total GHG excl. LULUCF"
                            and columns["category"] == "Updated NDC"
                            and columns["ambition"] == amb
                        ):
                            data = item["data"]
                            time_values = [int(year) for year in data.keys()]
                            ghg_values = np.array(list(item["data"].values()))
                            ghg_values[ghg_values == "None"] = np.nan
                            ghg_values = ghg_values.astype(float)
                            ghg_values = ghg_values[np.array(time_values) >= 2010]
                            ghg_data[cty_i + 1, cond_i, hot_i, amb_i] = ghg_values
                            # series.append([country_iso.upper(), country_name, "Emissions|Total GHG excl. LULUCF", conditionality, hot_air, ambition] + list(ghg_values))
                except:
                    continue

    coords = {
        "Region": list(countries_iso) + ["EU"],
        "Conditionality": ["conditional", "range", "unconditional"],
        "Hot_air": ["include", "exclude"],
        "Ambition": ["min", "max"],
        "Time": np.array(time_values)[np.array(time_values) >= 2010],
    }
    data_vars = {
        "GHG_ndc_excl_CR": (
            ["Region", "Conditionality", "Hot_air", "Ambition", "Time"],
            ghg_data,
        ),
    }
    xr_ndc = xr.Dataset(data_vars, coords=coords)
    xr_ndc_CR = xr_ndc.sel(Time=2030)

    return xr_ndc_CR


def read_ndc(config: Config, countries, xr_hist):
    print("- Reading NDC data")

    data_root = config.paths.input
    filename = "Infographics PBL NDC Tool 4Oct2024_for CarbonBudgetExplorer.xlsx"

    # TODO: use package or better dict mapping method instead
    countries_name = np.array(list(countries.keys()))
    countries_iso = np.array(list(countries.values()))

    df_ndc_raw = pd.read_excel(
        data_root / filename, sheet_name="Reduction All_GHG_incl", header=[0, 1]
    )
    regs = df_ndc_raw["(Mt CO2 equivalent)"]["Country name"]
    regs_iso = []

    for r in regs:
        wh = np.where(countries_name == r)[0]
        if len(wh) == 0:
            if r == "United States":
                regs_iso.append("USA")
            elif r == "EU27":
                regs_iso.append("EU")
            elif r == "Turkey":
                regs_iso.append("TUR")
            else:
                regs_iso.append(np.nan)
        else:
            regs_iso.append(countries_iso[wh[0]])
    regs_iso = np.array(regs_iso)
    df_ndc_raw["ISO"] = regs_iso

    df_regs = []
    df_amb = []
    df_con = []
    df_emis = []
    df_lulucf = []
    df_red = []
    df_abs = []
    df_inv = []
    histemis = xr_hist.GHG_hist.sel(Time=2015)
    for r in list(countries_iso) + ["EU"]:
        histemis_r = float(histemis.sel(Region=r))
        df_ndc_raw_sub = df_ndc_raw[df_ndc_raw["ISO"] == r]
        if len(df_ndc_raw_sub) > 0:
            val_2015 = float(df_ndc_raw_sub["(Mt CO2 equivalent)"][2015])
            for lulucf in ["incl"]:  # Maybe add excl later?
                for emis_i, emis in enumerate(["NDC"]):  # , 'CP']):
                    key = ["2030 NDCs", "Domestic actions 2030"][emis_i]
                    for cond_i, cond in enumerate(["unconditional", "conditional"]):
                        condkey = ["Unconditional NDCs", "Conditional NDCs"][cond_i]
                        for ambition_i, ambition in enumerate(["min", "max"]):
                            add = ["", ".1"][ambition_i]
                            val = float(df_ndc_raw_sub[key][condkey + add])
                            red = 1 - val / val_2015
                            abs_jones = histemis_r * (1 - red)
                            df_regs.append(r)
                            df_amb.append(ambition)
                            df_con.append(cond)
                            df_emis.append(emis)
                            df_lulucf.append(lulucf)
                            df_red.append(red)
                            df_abs.append(abs_jones)
                            df_inv.append(val)

    dict_ndc = {
        "Region": df_regs,
        "Ambition": df_amb,
        "Conditionality": df_con,
        "GHG_ndc_red": df_red,
        "GHG_ndc": df_abs,
        "GHG_ndc_inv": df_inv,
    }
    df_ndc = pd.DataFrame(dict_ndc)
    xr_ndc = xr.Dataset.from_dataframe(df_ndc.set_index(["Region", "Ambition", "Conditionality"]))

    # Now for GHG excluding LULUCF
    df_ndc_raw = pd.read_excel(
        data_root / filename, sheet_name="Reduction All_GHG_excl", header=[0, 1]
    )
    regs = df_ndc_raw["(Mt CO2 equivalent)"]["Country name"]
    regs_iso = []
    for r in regs:
        wh = np.where(countries_name == r)[0]
        if len(wh) == 0:
            if r == "United States":
                regs_iso.append("USA")
            elif r == "EU27":
                regs_iso.append("EU")
            elif r == "Turkey":
                regs_iso.append("TUR")
            else:
                regs_iso.append(np.nan)
        else:
            regs_iso.append(countries_iso[wh[0]])
    regs_iso = np.array(regs_iso)
    df_ndc_raw["ISO"] = regs_iso

    df_regs = []
    df_amb = []
    df_con = []
    df_emis = []
    df_lulucf = []
    df_red = []
    df_abs = []
    df_inv = []
    histemis = xr_hist.GHG_hist.sel(Time=2015)
    for r in list(countries_iso) + ["EU"]:
        histemis_r = float(histemis.sel(Region=r))
        df_ndc_raw_sub = df_ndc_raw[df_ndc_raw["ISO"] == r]
        if len(df_ndc_raw_sub) > 0:
            val_2015 = float(df_ndc_raw_sub["(Mt CO2 equivalent)"][2015])
            for lulucf in ["incl"]:  # Maybe add excl later?
                for emis_i, emis in enumerate(["NDC"]):  # , 'CP']):
                    key = ["2030 NDCs", "Domestic actions 2030"][emis_i]
                    for cond_i, cond in enumerate(["unconditional", "conditional"]):
                        condkey = ["Unconditional NDCs", "Conditional NDCs"][cond_i]
                        for ambition_i, ambition in enumerate(["min", "max"]):
                            add = ["", ".1"][ambition_i]
                            val = float(df_ndc_raw_sub[key][condkey + add])
                            red = 1 - val / val_2015
                            abs_jones = histemis_r * (1 - red)
                            df_regs.append(r)
                            df_amb.append(ambition)
                            df_con.append(cond)
                            df_emis.append(emis)
                            df_lulucf.append(lulucf)
                            df_red.append(red)
                            df_abs.append(abs_jones)
                            df_inv.append(val)

    dict_ndc = {
        "Region": df_regs,
        "Ambition": df_amb,
        "Conditionality": df_con,
        "GHG_ndc_excl_red": df_red,
        "GHG_ndc_excl": df_abs,
        "GHG_ndc_excl_inv": df_inv,
    }
    df_ndc = pd.DataFrame(dict_ndc)
    xr_ndc_excl = xr.Dataset.from_dataframe(
        df_ndc.set_index(["Region", "Ambition", "Conditionality"])
    )

    return xr_ndc, xr_ndc_excl


# TODO: check this function
def read_and_save_all(config: Config, countries, xr_hist) -> float:
    print("- Reading NDC data")

    # Read NDC data from Climate resource
    xr_ndc_CR = read_ndc_climateresource(config, countries)

    # Read NDC data from PBL
    xr_ndc, xr_ndc_excl = read_ndc(config, countries, xr_hist)

    # Merge datasets
    xr_ndc = xr.merge([xr_ndc, xr_ndc_excl])
    xr_ndc = xr.merge([xr_ndc, xr_ndc_CR])

    return xr_ndc
