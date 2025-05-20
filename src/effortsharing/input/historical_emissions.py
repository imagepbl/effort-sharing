import logging

import numpy as np
import pandas as pd
import xarray as xr

from effortsharing.config import Config

logger = logging.getLogger(__name__)


def read_primap(config: Config):
    """Read PRIMAP data."""
    logger.info("Reading PRIMAP data")

    # Define input
    data_root = config.paths.input
    guetschow_et_al = "Guetschow_et_al_2024-PRIMAP-hist_v2.5.1_final_no_rounding_27-Feb-2024.nc"

    # TODO: prefer method chaining or more explicit steps?

    # Read data
    ds = xr.open_dataset(data_root / guetschow_et_al)

    # Name coordinates
    ds = ds.rename(
        {
            "area (ISO3)": "Region",
            "scenario (PRIMAP-hist)": "Scenario",
            "category (IPCC2006_PRIMAP)": "Category",
        }
    )

    # Select relevant data
    ds = ds.sel(provenance="derived", source="PRIMAP-hist_v2.5.1_final_nr")

    # Simplify time coordinate to use years instead of full datetimes
    ds = ds.assign_coords(time=ds.time.dt.year)

    # TODO: rename time to Time here? Then we can omit it in the other primap functions
    return ds


def extract_primap_agri(primap: xr.Dataset):
    """Extract agricultural emissions from PRIMAP data."""
    primap_agri = (
        primap["KYOTOGHG (AR6GWP100)"]
        .sel(Scenario="HISTTP", Category=["M.AG"])
        .sum(dim="Category")
        .drop_vars(["source", "provenance", "Scenario"])
        .rename({"time": "Time"})
    )

    return primap_agri


def extract_primap_co2(primap: xr.Dataset):
    """Extract CO2 emissions from PRIMAP data."""
    primap_co2 = (
        primap["CO2"]
        .sel(Scenario="HISTTP", Category=["M.AG"])
        .sum(dim="Category")
        .drop_vars(["source", "provenance", "Scenario"])
        .rename({"time": "Time"})
    )

    return primap_co2


def read_jones_alternative(config: Config, regions):
    # No harmonization with the KEV anymore, but it's also much closer now
    logger.info("Reading historical emissions (jones)")

    # Define input
    # TODO: separate functions for reading each of these files?
    data_root = config.paths.input
    emissions_file = "EMISSIONS_ANNUAL_1830-2022.csv"
    edgar_file = "EDGARv8.0_FT2022_GHG_booklet_2023.xlsx"
    country_groups_file = "UNFCCC_Parties_Groups_noeu.xlsx"

    # Read primap data
    xr_primap = read_primap(config)
    xr_primap_agri = extract_primap_agri(xr_primap)
    xr_primap_agri_co2 = extract_primap_co2(xr_primap)

    df_nwc = pd.read_csv(data_root / emissions_file)
    xr_nwc = xr.Dataset.from_dataframe(
        df_nwc.drop(columns=["CNTR_NAME", "Unit"]).set_index(["ISO3", "Gas", "Component", "Year"])
    )

    # Rename GLOBAL to EARTH
    regs = np.array(xr_nwc.ISO3)
    regs[regs == "GLOBAL"] = "EARTH"
    xr_nwc["ISO3"] = regs

    # Calculate total(?)
    xr_nwc_tot = (
        xr_nwc.sel(Gas="CH[4]") * config.params.gwp_ch4 / 1e3
        + xr_nwc.sel(Gas="N[2]*O") * config.params.gwp_n2o / 1e3
        + xr_nwc.sel(Gas="CO[2]") * 1
    ).drop_vars(["Gas"])

    # Calculate individual contributions
    # TODO: adding these will yield same total as above and seems cleaner
    xr_nwc_co2 = xr_nwc.sel(Gas="CO[2]").drop_vars(["Gas"])
    xr_nwc_ch4 = xr_nwc.sel(Gas="CH[4]").drop_vars(["Gas"]) * config.params.gwp_ch4 / 1e3
    xr_nwc_n2o = xr_nwc.sel(Gas="N[2]*O").drop_vars(["Gas"]) * config.params.gwp_n2o / 1e3

    # Select historical data from NWC
    xr_ghghist = (
        xr_nwc_tot.rename({"ISO3": "Region", "Year": "Time", "Data": "GHG_hist"})
        .sel(Component="Total")
        .drop_vars("Component")
    )
    xr_co2hist = (
        xr_nwc_co2.rename({"ISO3": "Region", "Year": "Time", "Data": "CO2_hist"})
        .sel(Component="Total")
        .drop_vars("Component")
    )
    xr_ch4hist = (
        xr_nwc_ch4.rename({"ISO3": "Region", "Year": "Time", "Data": "CH4_hist"})
        .sel(Component="Total")
        .drop_vars("Component")
    )
    xr_n2ohist = (
        xr_nwc_n2o.rename({"ISO3": "Region", "Year": "Time", "Data": "N2O_hist"})
        .sel(Component="Total")
        .drop_vars("Component")
    )

    # Calculate emissions excluding LULUCF
    xr_ghgexcl = (
        xr_nwc_tot.rename({"ISO3": "Region", "Year": "Time"})
        .sel(Component="Total")
        .drop_vars("Component")
        - xr_nwc_tot.rename({"ISO3": "Region", "Year": "Time"})
        .sel(Component="LULUCF")
        .drop_vars("Component")
        + xr_primap_agri / 1e6
    ).rename({"Data": "GHG_hist_excl"})
    xr_co2excl = (
        xr_nwc_co2.rename({"ISO3": "Region", "Year": "Time"})
        .sel(Component="Total")
        .drop_vars("Component")
        - xr_nwc_co2.rename({"ISO3": "Region", "Year": "Time"})
        .sel(Component="LULUCF")
        .drop_vars("Component")
        + xr_primap_agri_co2 / 1e6
    ).rename({"Data": "CO2_hist_excl"})

    # Combine historical data into single xarray dataset
    regions_iso = list(regions.values())
    xr_hist = (
        xr.merge([xr_ghghist, xr_ghgexcl, xr_co2hist, xr_co2excl, xr_ch4hist, xr_n2ohist]) * 1e3
    ).reindex({"Region": regions_iso})

    # Store LULUCF (?)
    xr_ghg_afolu = (
        xr_nwc_tot.rename({"ISO3": "Region", "Year": "Time"})
        .sel(Component="LULUCF")
        .drop_vars("Component")
    )

    # Change units of agri
    # TODO: move this line closer to other agri steps?
    xr_ghg_agri = xr_primap_agri / 1e6

    # Also read EDGAR for purposes of using CR data (note that this is GHG excl LULUCF)
    # TODO: move to separate function?
    df_edgar = (
        pd.read_excel(data_root / edgar_file, sheet_name="GHG_totals_by_country")
        .drop(["Country"], axis=1)
        .set_index("EDGAR Country Code")
    )
    df_edgar.columns = df_edgar.columns.astype(int)

    # drop second-to-last row
    df_edgar = df_edgar.drop(df_edgar.index[-2])

    # Rename index column
    df_edgar.index.name = "Region"

    # Melt time columns into a time index
    df_edgar = (
        df_edgar.reset_index()
        .melt(id_vars="Region", var_name="Time", value_name="Emissions")
        .set_index(["Region", "Time"])
    )

    # Convert to xarray
    xr_edgar = df_edgar.to_xarray()

    # Add EU (this is required for the NDC data reading)
    df = pd.read_excel(data_root / country_groups_file, sheet_name="Country groups")
    countries_iso = np.array(df["Country ISO Code"])
    group_eu = countries_iso[np.array(df["EU"]) == 1]
    xr_hist.GHG_hist.loc[dict(Region="EU")] = xr_hist.GHG_hist.sel(Region=group_eu).sum("Region")

    return (
        xr_hist,
        # xr_ghg_afolu,  # TODO: not used, remove?
        # xr_ghg_agri,  # TODO: not used, remove?
        # xr_edgar,  # TODO: not used, remove?
        xr_primap,
    )
