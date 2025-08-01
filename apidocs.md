# Table of Contents

- [Table of Contents](#table-of-contents)
- [effortsharing](#effortsharing)
- [effortsharing.config](#effortsharingconfig)
  - [Config Objects](#config-objects)
- [effortsharing.allocations.allocation\_combinedapproaches](#effortsharingallocationsallocation_combinedapproaches)
  - [allocation\_comb Objects](#allocation_comb-objects)
      - [discounting\_historical\_emissions](#discounting_historical_emissions)
      - [ecpc](#ecpc)
      - [pc](#pc)
      - [approach1gdp](#approach1gdp)
      - [approach1hdi](#approach1hdi)
      - [approach2](#approach2)
      - [approach2\_transition](#approach2_transition)
- [effortsharing.save](#effortsharingsave)
      - [save\_total](#save_total)
      - [save\_rbw](#save_rbw)
      - [load\_rci](#load_rci)
- [effortsharing.allocation.gf](#effortsharingallocationgf)
      - [gf](#gf)
- [effortsharing.allocation.gdr](#effortsharingallocationgdr)
      - [gdr](#gdr)
- [effortsharing.allocation.pc](#effortsharingallocationpc)
      - [pc](#pc-1)
- [effortsharing.allocation.ap](#effortsharingallocationap)
      - [ap](#ap)
- [effortsharing.allocation.pcc](#effortsharingallocationpcc)
      - [pcc](#pcc)
- [effortsharing.allocation.pcb](#effortsharingallocationpcb)
      - [pcb](#pcb)
- [effortsharing.allocation.ecpc](#effortsharingallocationecpc)
      - [ecpc](#ecpc-1)
- [effortsharing.allocation.utils](#effortsharingallocationutils)
- [effortsharing.allocation](#effortsharingallocation)
      - [determine\_allocations](#determine_allocations)
      - [save\_allocations](#save_allocations)
- [effortsharing.postanalysis.tempalign](#effortsharingpostanalysistempalign)
- [effortsharing.postanalysis.variancedecomp](#effortsharingpostanalysisvariancedecomp)
- [effortsharing.pathways.global\_budgets](#effortsharingpathwaysglobal_budgets)
- [effortsharing.pathways.co2\_trajectories](#effortsharingpathwaysco2_trajectories)
- [effortsharing.pathways.nonco2](#effortsharingpathwaysnonco2)
- [effortsharing.main](#effortsharingmain)
- [effortsharing.exports](#effortsharingexports)
  - [dataexportcl Objects](#dataexportcl-objects)
      - [global\_default](#global_default)
      - [negative\_nonlulucf\_emissions](#negative_nonlulucf_emissions)
      - [global\_all](#global_all)
      - [ndcdata](#ndcdata)
      - [sspdata](#sspdata)
      - [emisdata](#emisdata)
      - [reduce\_country\_files](#reduce_country_files)
      - [allocations\_default](#allocations_default)
      - [budgets\_key\_variables](#budgets_key_variables)
      - [co2\_budgets\_ap](#co2_budgets_ap)
      - [co2\_budgets\_pc](#co2_budgets_pc)
      - [co2\_budgets\_ecpc](#co2_budgets_ecpc)
      - [concat\_co2budgets](#concat_co2budgets)
      - [project\_COMMITTED](#project_committed)
      - [project\_DGIS](#project_dgis)
      - [countr\_to\_csv](#countr_to_csv)
- [effortsharing.country\_specific.norway](#effortsharingcountry_specificnorway)
- [effortsharing.country\_specific.netherlands](#effortsharingcountry_specificnetherlands)
- [effortsharing.country\_specific](#effortsharingcountry_specific)
- [effortsharing.input.all](#effortsharinginputall)
      - [load\_all](#load_all)
- [effortsharing.input.policyscens](#effortsharinginputpolicyscens)
- [effortsharing.input.ndcs](#effortsharinginputndcs)
      - [load\_ndcs](#load_ndcs)
- [effortsharing.input.emissions](#effortsharinginputemissions)
      - [read\_primap](#read_primap)
      - [extract\_primap\_agri](#extract_primap_agri)
      - [extract\_primap\_agri\_co2](#extract_primap_agri_co2)
      - [read\_jones](#read_jones)
      - [read\_edgar](#read_edgar)
      - [load\_emissions](#load_emissions)
- [effortsharing.input.socioeconomics](#effortsharinginputsocioeconomics)
      - [read\_general](#read_general)
      - [load\_socioeconomics](#load_socioeconomics)
- [effortsharing.input](#effortsharinginput)
- [effortsharing.cache](#effortsharingcache)
      - [intermediate\_file](#intermediate_file)

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

<a id="effortsharing.allocation.utils"></a>

# effortsharing.allocation.utils

<a id="effortsharing.allocation"></a>

# effortsharing.allocation

<a id="effortsharing.allocation.determine_allocations"></a>

#### determine\_allocations

```python
def determine_allocations(config: Config,
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

<a id="effortsharing.postanalysis.tempalign"></a>

# effortsharing.postanalysis.tempalign

<a id="effortsharing.postanalysis.variancedecomp"></a>

# effortsharing.postanalysis.variancedecomp

<a id="effortsharing.pathways.global_budgets"></a>

# effortsharing.pathways.global\_budgets

<a id="effortsharing.pathways.co2_trajectories"></a>

# effortsharing.pathways.co2\_trajectories

<a id="effortsharing.pathways.nonco2"></a>

# effortsharing.pathways.nonco2

<a id="effortsharing.main"></a>

# effortsharing.main

main.py

This file executes the full workflow to obtain global GHG/CO2 budgets and
trajectories, similar to the first cell in notebooks/Main.ipynb

It collects data from all input files, combines them into one big dataset, which is saved as xr_dataread.nc.
Also, some country-specific datareaders are executed.

Run with:
    python src/effortsharing/main.py notebooks/config.yml

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

<a id="effortsharing.country_specific.norway"></a>

# effortsharing.country\_specific.norway

<a id="effortsharing.country_specific.netherlands"></a>

# effortsharing.country\_specific.netherlands

<a id="effortsharing.country_specific"></a>

# effortsharing.country\_specific

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

<a id="effortsharing.input.policyscens"></a>

# effortsharing.input.policyscens

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

