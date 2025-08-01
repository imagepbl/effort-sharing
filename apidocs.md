# Table of Contents

* [effortsharing](#effortsharing)
* [effortsharing.config](#effortsharing.config)
  * [Config](#effortsharing.config.Config)
* [effortsharing.cli](#effortsharing.cli)
  * [get\_version](#effortsharing.cli.get_version)
  * [use\_rich\_logger](#effortsharing.cli.use_rich_logger)
  * [generate\_config](#effortsharing.cli.generate_config)
  * [get\_input\_data](#effortsharing.cli.get_input_data)
  * [global\_pathways](#effortsharing.cli.global_pathways)
  * [policy\_scenarios](#effortsharing.cli.policy_scenarios)
  * [allocate](#effortsharing.cli.allocate)
  * [aggregate](#effortsharing.cli.aggregate)
* [effortsharing.country\_specific.netherlands](#effortsharing.country_specific.netherlands)
* [effortsharing.country\_specific](#effortsharing.country_specific)
* [effortsharing.country\_specific.norway](#effortsharing.country_specific.norway)
* [effortsharing.pathways.co2\_trajectories](#effortsharing.pathways.co2_trajectories)
* [effortsharing.pathways.global\_pathways](#effortsharing.pathways.global_pathways)
* [effortsharing.pathways.nonco2](#effortsharing.pathways.nonco2)
* [effortsharing.pathways.global\_budgets](#effortsharing.pathways.global_budgets)
* [effortsharing.allocation.pcb](#effortsharing.allocation.pcb)
  * [pcb](#effortsharing.allocation.pcb.pcb)
* [effortsharing.allocation](#effortsharing.allocation)
  * [allocations\_for\_year](#effortsharing.allocation.allocations_for_year)
  * [allocations\_for\_region](#effortsharing.allocation.allocations_for_region)
  * [save\_allocations](#effortsharing.allocation.save_allocations)
* [effortsharing.allocation.pcc](#effortsharing.allocation.pcc)
  * [pcc](#effortsharing.allocation.pcc.pcc)
* [effortsharing.allocation.pc](#effortsharing.allocation.pc)
  * [pc](#effortsharing.allocation.pc.pc)
* [effortsharing.allocation.ecpc](#effortsharing.allocation.ecpc)
  * [ecpc](#effortsharing.allocation.ecpc.ecpc)
* [effortsharing.allocation.gf](#effortsharing.allocation.gf)
  * [gf](#effortsharing.allocation.gf.gf)
* [effortsharing.allocation.ap](#effortsharing.allocation.ap)
  * [ap](#effortsharing.allocation.ap.ap)
* [effortsharing.allocation.utils](#effortsharing.allocation.utils)
* [effortsharing.allocation.gdr](#effortsharing.allocation.gdr)
  * [gdr](#effortsharing.allocation.gdr.gdr)
* [effortsharing.allocations.allocation\_combinedapproaches](#effortsharing.allocations.allocation_combinedapproaches)
  * [allocation\_comb](#effortsharing.allocations.allocation_combinedapproaches.allocation_comb)
    * [discounting\_historical\_emissions](#effortsharing.allocations.allocation_combinedapproaches.allocation_comb.discounting_historical_emissions)
    * [ecpc](#effortsharing.allocations.allocation_combinedapproaches.allocation_comb.ecpc)
    * [pc](#effortsharing.allocations.allocation_combinedapproaches.allocation_comb.pc)
    * [approach1gdp](#effortsharing.allocations.allocation_combinedapproaches.allocation_comb.approach1gdp)
    * [approach1hdi](#effortsharing.allocations.allocation_combinedapproaches.allocation_comb.approach1hdi)
    * [approach2](#effortsharing.allocations.allocation_combinedapproaches.allocation_comb.approach2)
    * [approach2\_transition](#effortsharing.allocations.allocation_combinedapproaches.allocation_comb.approach2_transition)
* [effortsharing.cache](#effortsharing.cache)
  * [intermediate\_file](#effortsharing.cache.intermediate_file)
* [effortsharing.exports](#effortsharing.exports)
  * [dataexportcl](#effortsharing.exports.dataexportcl)
    * [global\_default](#effortsharing.exports.dataexportcl.global_default)
    * [negative\_nonlulucf\_emissions](#effortsharing.exports.dataexportcl.negative_nonlulucf_emissions)
    * [global\_all](#effortsharing.exports.dataexportcl.global_all)
    * [ndcdata](#effortsharing.exports.dataexportcl.ndcdata)
    * [sspdata](#effortsharing.exports.dataexportcl.sspdata)
    * [emisdata](#effortsharing.exports.dataexportcl.emisdata)
    * [reduce\_country\_files](#effortsharing.exports.dataexportcl.reduce_country_files)
    * [allocations\_default](#effortsharing.exports.dataexportcl.allocations_default)
    * [budgets\_key\_variables](#effortsharing.exports.dataexportcl.budgets_key_variables)
    * [co2\_budgets\_ap](#effortsharing.exports.dataexportcl.co2_budgets_ap)
    * [co2\_budgets\_pc](#effortsharing.exports.dataexportcl.co2_budgets_pc)
    * [co2\_budgets\_ecpc](#effortsharing.exports.dataexportcl.co2_budgets_ecpc)
    * [concat\_co2budgets](#effortsharing.exports.dataexportcl.concat_co2budgets)
    * [project\_COMMITTED](#effortsharing.exports.dataexportcl.project_COMMITTED)
    * [project\_DGIS](#effortsharing.exports.dataexportcl.project_DGIS)
    * [countr\_to\_csv](#effortsharing.exports.dataexportcl.countr_to_csv)
* [effortsharing.postanalysis.variancedecomp](#effortsharing.postanalysis.variancedecomp)
* [effortsharing.postanalysis.tempalign](#effortsharing.postanalysis.tempalign)
* [effortsharing.input.socioeconomics](#effortsharing.input.socioeconomics)
  * [read\_general](#effortsharing.input.socioeconomics.read_general)
  * [load\_socioeconomics](#effortsharing.input.socioeconomics.load_socioeconomics)
* [effortsharing.input.policyscens](#effortsharing.input.policyscens)
* [effortsharing.input](#effortsharing.input)
* [effortsharing.input.all](#effortsharing.input.all)
  * [load\_all](#effortsharing.input.all.load_all)
* [effortsharing.input.emissions](#effortsharing.input.emissions)
  * [read\_primap](#effortsharing.input.emissions.read_primap)
  * [extract\_primap\_agri](#effortsharing.input.emissions.extract_primap_agri)
  * [extract\_primap\_agri\_co2](#effortsharing.input.emissions.extract_primap_agri_co2)
  * [read\_jones](#effortsharing.input.emissions.read_jones)
  * [read\_edgar](#effortsharing.input.emissions.read_edgar)
  * [load\_emissions](#effortsharing.input.emissions.load_emissions)
* [effortsharing.input.ndcs](#effortsharing.input.ndcs)
  * [load\_ndcs](#effortsharing.input.ndcs.load_ndcs)
* [effortsharing.save](#effortsharing.save)
  * [save\_total](#effortsharing.save.save_total)
  * [save\_rbw](#effortsharing.save.save_rbw)
  * [load\_rci](#effortsharing.save.load_rci)

<a id="effortsharing"></a>

# effortsharing

<a id="effortsharing.config"></a>

# effortsharing.config

<a id="effortsharing.config.Config"></a>

## Config Objects

```python
@dataclass
class Config()
```

Configuration of effort-sharing experiments.

For example:

    config = Config.from_file('config.yml')


You can get a nice print of the config using rich:

    from rich import print
    print(config)

<a id="effortsharing.cli"></a>

# effortsharing.cli

CLI tool for Effort Sharing.

<a id="effortsharing.cli.get_version"></a>

#### get\_version

```python
def get_version() -> str
```

Get the package version at runtime.

<a id="effortsharing.cli.use_rich_logger"></a>

#### use\_rich\_logger

```python
def use_rich_logger(level: LogLevel = "INFO")
```

Set up logging with RichHandler.

**Arguments**:

- `level` - The logging level to set.

<a id="effortsharing.cli.generate_config"></a>

#### generate\_config

```python
@app.command
def generate_config(
        config: Path = Path("config.yml"), log_level: LogLevel = "INFO")
```

Generate a configuration file by downloading the default one from GitHub.

**Arguments**:

- `config` - Path to configuration YAML file to write.
- `log_level` - Set the logging level.

<a id="effortsharing.cli.get_input_data"></a>

#### get\_input\_data

```python
@app.command
def get_input_data(
        config: Path = Path("config.yml"), log_level: LogLevel = "INFO")
```

Download input data files.

**Arguments**:

- `config` - Path to configuration YAML file.
- `log_level` - Set the logging level.

<a id="effortsharing.cli.global_pathways"></a>

#### global\_pathways

```python
@app.command
def global_pathways(
        config: Path = Path("config.yml"), log_level: LogLevel = "INFO")
```

Generate global pathways data.

**Arguments**:

- `config` - Path to configuration YAML file.
- `log_level` - Set the logging level.

<a id="effortsharing.cli.policy_scenarios"></a>

#### policy\_scenarios

```python
@app.command
def policy_scenarios(
        config: Path = Path("config.yml"), log_level: LogLevel = "INFO")
```

Generate policy scenarios data.

**Arguments**:

- `config` - Path to configuration YAML file.
- `log_level` - Set the logging level.

<a id="effortsharing.cli.allocate"></a>

#### allocate

```python
@app.command
def allocate(region: str,
             gas: Gas = "GHG",
             lulucf: LULUCF = "incl",
             config: Path = Path("config.yml"),
             log_level: LogLevel = "INFO")
```

Allocate emissions for a region.

**Arguments**:

- `region` - Region to allocate emissions for.
- `gas` - Gas type.
- `lulucf` - Land Use, Land-Use Change, and Forestry inclusion/exclusion.
- `config` - Path to configuration YAML file.
- `log_level` - Set the logging level.

<a id="effortsharing.cli.aggregate"></a>

#### aggregate

```python
@app.command
def aggregate(year: int,
              gas: Gas = "GHG",
              lulucf: LULUCF = "incl",
              config: Path = Path("config.yml"),
              log_level: LogLevel = "INFO")
```

Aggregate emissions data.

Expects that allocation has been generated for each region and the given gas and lulucf.

**Arguments**:

- `year` - Year to aggregate emissions for.
- `gas` - Gas type.
- `lulucf` - Land Use, Land-Use Change, and Forestry inclusion/exclusion.
- `config` - Path to configuration YAML file.
- `log_level` - Set the logging level.

<a id="effortsharing.country_specific.netherlands"></a>

# effortsharing.country\_specific.netherlands

<a id="effortsharing.country_specific"></a>

# effortsharing.country\_specific

<a id="effortsharing.country_specific.norway"></a>

# effortsharing.country\_specific.norway

<a id="effortsharing.pathways.co2_trajectories"></a>

# effortsharing.pathways.co2\_trajectories

<a id="effortsharing.pathways.global_pathways"></a>

# effortsharing.pathways.global\_pathways

Module with the full workflow to obtain global GHG/CO2 budgets and trajectories.

It collects data from all input files, combines them into one big dataset, which is saved as xr_dataread.nc.
Also, some country-specific datareaders are executed.

<a id="effortsharing.pathways.nonco2"></a>

# effortsharing.pathways.nonco2

<a id="effortsharing.pathways.global_budgets"></a>

# effortsharing.pathways.global\_budgets

<a id="effortsharing.allocation.pcb"></a>

# effortsharing.allocation.pcb

<a id="effortsharing.allocation.pcb.pcb"></a>

#### pcb

```python
def pcb(config: Config,
        region,
        gas: Gas = "GHG",
        lulucf: LULUCF = "incl") -> tuple[xr.DataArray, xr.DataArray]
```

Per capita on a budget basis

<a id="effortsharing.allocation"></a>

# effortsharing.allocation

<a id="effortsharing.allocation.allocations_for_year"></a>

#### allocations\_for\_year

```python
def allocations_for_year(config: Config, regions, gas: Gas, lulucf: LULUCF,
                         year: int)
```

Extract allocations for a specific year from the regional allocations.

<a id="effortsharing.allocation.allocations_for_region"></a>

#### allocations\_for\_region

```python
def allocations_for_region(config: Config,
                           region,
                           gas: Gas = "GHG",
                           lulucf: LULUCF = "incl") -> list[xr.DataArray]
```

Run all allocation methods and return list of xr.DataArray per method.

<a id="effortsharing.allocation.save_allocations"></a>

#### save\_allocations

```python
def save_allocations(config: Config,
                     region: str,
                     dss: Iterable[xr.DataArray],
                     gas: Gas = "GHG",
                     lulucf: LULUCF = "incl")
```

Combine data arrays returned by each allocation method into a NetCDF file

<a id="effortsharing.allocation.pcc"></a>

# effortsharing.allocation.pcc

<a id="effortsharing.allocation.pcc.pcc"></a>

#### pcc

```python
def pcc(config: Config,
        region,
        gas: Gas = "GHG",
        lulucf: LULUCF = "incl",
        gf_da: xr.DataArray | None = None,
        pc_da: xr.DataArray | None = None) -> xr.DataArray
```

Per Capita Convergence: Grandfathering converging into per capita

<a id="effortsharing.allocation.pc"></a>

# effortsharing.allocation.pc

<a id="effortsharing.allocation.pc.pc"></a>

#### pc

```python
def pc(config: Config,
       region,
       gas: Gas = "GHG",
       lulucf: LULUCF = "incl") -> xr.DataArray
```

Per Capita: Divide the global budget equally per capita

<a id="effortsharing.allocation.ecpc"></a>

# effortsharing.allocation.ecpc

<a id="effortsharing.allocation.ecpc.ecpc"></a>

#### ecpc

```python
def ecpc(config: Config,
         region,
         gas: Gas = "GHG",
         lulucf: LULUCF = "incl") -> xr.DataArray
```

Equal Cumulative per Capita: Uses historical emissions, discount factors and
population shares to allocate the global budget

<a id="effortsharing.allocation.gf"></a>

# effortsharing.allocation.gf

<a id="effortsharing.allocation.gf.gf"></a>

#### gf

```python
def gf(config: Config,
       region,
       gas: Gas = "GHG",
       lulucf: LULUCF = "incl") -> xr.DataArray
```

Grandfathering: Divide the global budget over the regions based on
their historical CO2 emissions

<a id="effortsharing.allocation.ap"></a>

# effortsharing.allocation.ap

<a id="effortsharing.allocation.ap.ap"></a>

#### ap

```python
def ap(config: Config,
       region,
       gas: Gas = "GHG",
       lulucf: LULUCF = "incl") -> xr.DataArray
```

Ability to Pay: Uses GDP per capita to allocate the global budget
Equation from van den Berg et al. (2020)

<a id="effortsharing.allocation.utils"></a>

# effortsharing.allocation.utils

<a id="effortsharing.allocation.gdr"></a>

# effortsharing.allocation.gdr

<a id="effortsharing.allocation.gdr.gdr"></a>

#### gdr

```python
def gdr(config: Config,
        region,
        gas: Gas = "GHG",
        lulucf: LULUCF = "incl",
        ap_da: xr.DataArray | None = None) -> xr.DataArray
```

Greenhouse Development Rights: Uses the Responsibility-Capability Index
(RCI) weighed at 50/50 to allocate the global budget
Calculations from van den Berg et al. (2020)

<a id="effortsharing.allocations.allocation_combinedapproaches"></a>

# effortsharing.allocations.allocation\_combinedapproaches

<a id="effortsharing.allocations.allocation_combinedapproaches.allocation_comb"></a>

## allocation\_comb Objects

```python
class allocation_comb()
```

<a id="effortsharing.allocations.allocation_combinedapproaches.allocation_comb.discounting_historical_emissions"></a>

#### discounting\_historical\_emissions

```python
def discounting_historical_emissions()
```

ECPC computation

<a id="effortsharing.allocations.allocation_combinedapproaches.allocation_comb.ecpc"></a>

#### ecpc

```python
def ecpc()
```

ECPC computation

<a id="effortsharing.allocations.allocation_combinedapproaches.allocation_comb.pc"></a>

#### pc

```python
def pc()
```

ECPC computation

<a id="effortsharing.allocations.allocation_combinedapproaches.allocation_comb.approach1gdp"></a>

#### approach1gdp

```python
def approach1gdp()
```

Methods for Robiou et al. (2023), under review.

<a id="effortsharing.allocations.allocation_combinedapproaches.allocation_comb.approach1hdi"></a>

#### approach1hdi

```python
def approach1hdi()
```

Methods for Robiou et al. (2023), under review.

<a id="effortsharing.allocations.allocation_combinedapproaches.allocation_comb.approach2"></a>

#### approach2

```python
def approach2()
```

Methods for Robiou et al. (2023), under review.

<a id="effortsharing.allocations.allocation_combinedapproaches.allocation_comb.approach2_transition"></a>

#### approach2\_transition

```python
def approach2_transition()
```

Methods for Robiou et al. (2023), under review.

<a id="effortsharing.cache"></a>

# effortsharing.cache

<a id="effortsharing.cache.intermediate_file"></a>

#### intermediate\_file

```python
def intermediate_file(filename, loader=None, saver=None)
```

Decorator for caching function results to/from intermediate files.

**Arguments**:

- `filename` - Name of the intermediate file to use
- `loader` - Optional custom loader function, otherwise determined by file extension
- `saver` - Optional custom saver function, otherwise determined by file extension
  
  The decorated function should accept a config object as argument that should
  contain the following parameters:
  
  - path.intermediate: path to intermediate data directory.
  - load_intermediate_files: Whether to load from cache if available (default: True)
  - save_intermediate_files: Whether to save results to cache (default: True)
  
  These parameters are automatically available in the wrapper scope and should
  be passed down to any nested decorated functions to maintain consistent behavior.

<a id="effortsharing.exports"></a>

# effortsharing.exports

<a id="effortsharing.exports.dataexportcl"></a>

## dataexportcl Objects

```python
class dataexportcl()
```

<a id="effortsharing.exports.dataexportcl.global_default"></a>

#### global\_default

```python
def global_default()
```

Export default 1.5(6) and 2.0 pathways that roughly match the IPCC pathways

<a id="effortsharing.exports.dataexportcl.negative_nonlulucf_emissions"></a>

#### negative\_nonlulucf\_emissions

```python
def negative_nonlulucf_emissions()
```

Export negative emissions pathways

<a id="effortsharing.exports.dataexportcl.global_all"></a>

#### global\_all

```python
def global_all()
```

Export a large set of pathways (still a subset)

<a id="effortsharing.exports.dataexportcl.ndcdata"></a>

#### ndcdata

```python
def ndcdata()
```

Export NDC data

<a id="effortsharing.exports.dataexportcl.sspdata"></a>

#### sspdata

```python
def sspdata()
```

Export SSP data

<a id="effortsharing.exports.dataexportcl.emisdata"></a>

#### emisdata

```python
def emisdata()
```

Export historical emission data

<a id="effortsharing.exports.dataexportcl.reduce_country_files"></a>

#### reduce\_country\_files

```python
def reduce_country_files()
```

Get reduced-form country files, omitting some parameter settings that users won't use and reducing the file size through compression

<a id="effortsharing.exports.dataexportcl.allocations_default"></a>

#### allocations\_default

```python
def allocations_default()
```

Export default emission allocations and reductions

<a id="effortsharing.exports.dataexportcl.budgets_key_variables"></a>

#### budgets\_key\_variables

```python
def budgets_key_variables(lulucf="incl")
```

Specify several key variables for the computation of budgets
Note that budgets are only in CO2, not in GHG (while most of the alloations are in GHG)

<a id="effortsharing.exports.dataexportcl.co2_budgets_ap"></a>

#### co2\_budgets\_ap

```python
def co2_budgets_ap()
```

CO2 budgets AP

<a id="effortsharing.exports.dataexportcl.co2_budgets_pc"></a>

#### co2\_budgets\_pc

```python
def co2_budgets_pc()
```

CO2 budgets PC

<a id="effortsharing.exports.dataexportcl.co2_budgets_ecpc"></a>

#### co2\_budgets\_ecpc

```python
def co2_budgets_ecpc()
```

CO2 budgets ECPC

<a id="effortsharing.exports.dataexportcl.concat_co2budgets"></a>

#### concat\_co2budgets

```python
def concat_co2budgets(lulucf="incl")
```

CO2 budgets ECPC, AP and PC

<a id="effortsharing.exports.dataexportcl.project_COMMITTED"></a>

#### project\_COMMITTED

```python
def project_COMMITTED()
```

Export files for COMMITTED

<a id="effortsharing.exports.dataexportcl.project_DGIS"></a>

#### project\_DGIS

```python
def project_DGIS()
```

Export files for DGIS

<a id="effortsharing.exports.dataexportcl.countr_to_csv"></a>

#### countr\_to\_csv

```python
def countr_to_csv(cty, adapt="", lulucf="incl", gas="GHG")
```

Convert .nc to .csv for a specific country

<a id="effortsharing.postanalysis.variancedecomp"></a>

# effortsharing.postanalysis.variancedecomp

<a id="effortsharing.postanalysis.tempalign"></a>

# effortsharing.postanalysis.tempalign

<a id="effortsharing.input.socioeconomics"></a>

# effortsharing.input.socioeconomics

Functions to read and process socio-economic input data from various sources.

Import as library:

    from effortsharing.input import socioeconomics


Or use as standalone script:

    python src/effortsharing/input/socioeconomics.py config.yml

<a id="effortsharing.input.socioeconomics.read_general"></a>

#### read\_general

```python
def read_general(config: Config)
```

Read country names and ISO from UNFCCC table.

<a id="effortsharing.input.socioeconomics.load_socioeconomics"></a>

#### load\_socioeconomics

```python
@intermediate_file("socioeconomics.nc")
def load_socioeconomics(config: Config)
```

Collect socio-economic input data from various sources to intermediate file.

**Arguments**:

- `config` - effortsharing.config.Config object
  

**Returns**:

- `xarray.Dataset` - Socio-economic data

<a id="effortsharing.input.policyscens"></a>

# effortsharing.input.policyscens

<a id="effortsharing.input"></a>

# effortsharing.input

Input module for processing various data sources.

Functions:
- load_all: Process all available data sources and save to disk

Submodules (for access to individual data reading functions):
- emissions
- socioeconomics
- ndcs
- all (source file for load_all)

<a id="effortsharing.input.all"></a>

# effortsharing.input.all

<a id="effortsharing.input.all.load_all"></a>

#### load\_all

```python
def load_all(config: Config)
```

Load all input data.

**Arguments**:

- `config` - effortsharing.config.Config object
- `from_intermediate` - Whether to read from intermediate files if available (default: True)
- `save` - Whether to save intermediate data to disk (default: True)

<a id="effortsharing.input.emissions"></a>

# effortsharing.input.emissions

<a id="effortsharing.input.emissions.read_primap"></a>

#### read\_primap

```python
@intermediate_file("primap.nc")
def read_primap(config: Config)
```

Read PRIMAP data.

<a id="effortsharing.input.emissions.extract_primap_agri"></a>

#### extract\_primap\_agri

```python
def extract_primap_agri(primap: xr.Dataset)
```

Extract agricultural emissions from PRIMAP data.

<a id="effortsharing.input.emissions.extract_primap_agri_co2"></a>

#### extract\_primap\_agri\_co2

```python
def extract_primap_agri_co2(primap: xr.Dataset)
```

Extract CO2 emissions from PRIMAP data.

<a id="effortsharing.input.emissions.read_jones"></a>

#### read\_jones

```python
@intermediate_file("emissions_history.nc")
def read_jones(config: Config, regions)
```

Read Jones historical emission data.

<a id="effortsharing.input.emissions.read_edgar"></a>

#### read\_edgar

```python
@intermediate_file("edgar.nc")
def read_edgar(config: Config)
```

Read EDGAR data.

<a id="effortsharing.input.emissions.load_emissions"></a>

#### load\_emissions

```python
@intermediate_file("emissions_all.nc")
def load_emissions(config: Config)
```

Collect emission input data from various sources to intermediate file.

**Arguments**:

- `config` - effortsharing.config.Config object
  

**Returns**:

- `xarray.Dataset` - Emission data

<a id="effortsharing.input.ndcs"></a>

# effortsharing.input.ndcs

<a id="effortsharing.input.ndcs.load_ndcs"></a>

#### load\_ndcs

```python
@intermediate_file("ndcs.nc")
def load_ndcs(config: Config, xr_hist=None)
```

Collect NDC input data from various sources to intermediate file.

**Arguments**:

- `config` - effortsharing.config.Config object
- `xr_hist` - xarray dataset containing variable "GHG_hist" for year 2015
- `from_intermediate` - Whether to read from intermediate files if available (default: True)
- `save` - Whether to save intermediate data to disk (default: True)
  

**Returns**:

- `xarray.Dataset` - NDC data

<a id="effortsharing.save"></a>

# effortsharing.save

<a id="effortsharing.save.save_total"></a>

#### save\_total

```python
def save_total(config: Config, xr_version)
```

Save xr_total to netcdf file.

<a id="effortsharing.save.save_rbw"></a>

#### save\_rbw

```python
def save_rbw(config: Config, xr_version, countries)
```

Save rbw factors to netcdf file.

<a id="effortsharing.save.load_rci"></a>

#### load\_rci

```python
@intermediate_file("xr_rci.nc")
def load_rci(config: Config) -> xr.Dataset
```

Load responsibility capability index (RCI) data from netcdf file.

