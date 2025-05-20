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
ADDITIONAL_REGIONS_HDI = {
    "Bahamas": "BHS",
    "Bolivia (Plurinational State of)": "BOL",
    "Cabo Verde": "CPV",
    # "China": "CHN",  # TODO: check which one it should be
    "China": "TWN",
    "Congo (Democratic Republic of the)": "COD",
    "Congo": "COG",
    "Czechia": "CZE",
    "Egypt": "EGY",
    "Eswatini (Kingdom of)": "SWZ",
    "Gambia": "GMB",
    "Hong Kong, China (SAR)": "HKG",
    "Iran (Islamic Republic of)": "IRN",
    "Korea (Democratic People's Rep. of)": "PRK",
    "Korea (Republic of)": "KOR",
    "Kyrgyzstan": "KGZ",
    "Lao People's Democratic Republic": "LAO",
    "Micronesia (Federated States of)": "FSM",
    "Moldova (Republic of)": "MDA",
    "Palestine, State of": "PSE",
    "Saint Kitts and Nevis": "KNA",
    "Saint Lucia": "LCA",
    "Saint Vincent and the Grenadines": "VCT",
    "Sao Tome and Principe": "STP",
    "Slovakia": "SVK",
    "Tanzania (United Republic of)": "TZA",
    "Türkiye": "TUR",
    "United States": "USA",
    "Venezuela (Bolivarian Republic of)": "VEN",
    "Viet Nam": "VNM",
    "Yemen": "YEM",
}


""" 
# How to use (for example)
from effortsharing import datareading
from effortsharing import regions

countries = read_general(filename)
regions = {**countries, **additional_eu_and_earth}
ssp_regions = {**countries, **additional_eu_and_earth, **additional_regions_ssps}
hdi_countries = {**countries, **additional_regions_hdi}

# Get individual countries' ISO
countries.get("Netherlands")  # "NLD"
ssp_regions.get("World")  # "EARTH"
hdi_countries.get("Yemen")  # "YEM"

# Get ISO for a list of countries
list(map(countries.get, ["Netherlands", "Germany"]))
# ['NLD', 'DEU']

# Get ISO for a list of countries with missing data
list(map(additional_regions_ssps.get, ["French Guiana", "French Polynesia", "French Fries"]))
# ['GUF', 'PYF', None]

# Get ISO for a list of countries with custom default when missing (do we really need this?)
list(map(lambda name: countries.get(name, "oeps"), ["Ireland", "South Africa", "Mordor"]))
# ['IRL', 'ZAF', 'oeps']

# The same thing achieved with a convenience function
def get_iso(names, lookup_table, default="oeps"):
    return list(map(lambda name: lookup_table.get(name, default), names))


get_iso(["Belgium", "France", "Atlantis"], hdi_countries, default="sunken")
# ['BEL', 'FRA', 'sunken']
"""
