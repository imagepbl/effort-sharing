import logging

import pandas as pd

from effortsharing.config import Config

logger = logging.getLogger(__name__)


def read_general(config: Config):
    """Read country names and ISO from UNFCCC table."""
    logger.info("Reading unfccc country data")

    data_root = config.paths.input
    filename = "UNFCCC_Parties_Groups_noeu.xlsx"

    # Read and transform countries
    columns = {"Name": "name", "Country ISO Code": "iso"}
    countries = (
        pd.read_excel(
            data_root / filename,
            sheet_name="Country groups",
            usecols=columns.keys(),
        )
        .rename(columns=columns)
        .set_index("name")["iso"]
        .to_dict()
    )

    # Extend countries with non-country regions
    regions = {**countries, **ADDITIONAL_EU_AND_EARTH}

    return countries, regions


ADDITIONAL_EU_AND_EARTH = {"European Union": "EU", "Earth": "EARTH"}
ADDITIONAL_REGIONS_SSPS = {
    "Aruba": "ABW",
    "Bahamas": "BHS",
    "C?te d'Ivoire": "CIV",
    "Cabo Verde": "CPV",
    "Cura?ao": "CUW",
    "Czechia": "CZE",
    "Democratic Republic of the Congo": "COD",
    "French Guiana": "GUF",
    "French Polynesia": "PYF",
    "Gambia": "GMB",
    "Guadeloupe": "GLP",
    "Guam": "GUM",
    "Hong Kong": "HKG",
    "Iran": "IRN",
    "Macao": "MAC",
    "Martinique": "MTQ",
    "Mayotte": "MYT",
    "Moldova": "MDA",
    "New Caledonia": "NCL",
    "Palestine": "PSE",
    "Puerto Rico": "PRI",
    "R?union": "REU",
    "Syria": "SYR",
    "Taiwan": "TWN",
    "Tanzania": "TZA",
    "Turkey": "TUR",
    "United States Virgin Islands": "VIR",
    "United States": "USA",
    "Venezuela": "VEN",
    "Viet Nam": "VNM",
    "Western Sahara": "ESH",
    "World": "EARTH",
}