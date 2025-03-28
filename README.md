## Introduction
This code combines a variety of data sources to compute fair national emissions allocations, studies variability in these allocations and compares them with NDC estimates and cost-optimal scenario projections. We plan to make the code more accessible in terms of commenting and cleaning up old code over time. The output data of this code is publicly available on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12188104.svg)](https://doi.org/10.5281/zenodo.12188104)

## Main features
The main notebook you will be using is `Main.ipynb` in the main directory. Each cell in that notebook loads in a particular class and then goes through all of its methods.
- The first one being called is the class in the file `class_datareading.py`. This class gathers all data from external sources, such as population, GDP, historical emissions, etc. on a country level, but also computes global pathways into the future. It does not do any effort sharing. The output is a file called `xr_dataread.nc`, which is your friend in case you need any country-specific data for debugging, for example. The main uncertainties / work from our side is around generating those future pathways under various circumstances (peak temperature, risk, non-CO2 assumptions, etc.).
- The second class being called is the one from `class_allocation.py`. That one does all the effort-sharing. It loads in `xr_dataread.nc` and its methods are named after the allocation rule being computed. The class requires an input: the region (ISO3) that is focused on (called `self.FocusRegion` in `class_allocation.py`). So in `Main.ipynb`, you see that there is a loop over all countries and even some country groups (e.g., G20) in the world. The output are files in the form of `xr_alloc_XYZ.nc` where XYZ is the ISO3 code for a country, and they are saved to K:/ECEMF/T5.2/Allocations/.
- The third and fourth classes (`class_allocation_combinedapproaches` and `class_tempalign`) are for future research.
- The fifth class (`class_policyscens`) is for reading cost-optimal scenarios and NDCs
- The final class (`class_variancedecomp`) conducts the variance decomposition (Sobol analysis)

One final important notebook is `Aggregator.ipynb`. This script aggregates the country-individual files `xr_alloc_XYZ.nc`. One output is the aggregation of those files into a single file for a single year, for example `xr_alloc_2030.nc`. Useful for analysis purposes. Input variables can be changed in `input.yml`.

## Installation instructions (currently aimed at direct colleagues)

It is recommended to clone the code to the `K:\` directory for best performance. 
```shell
git clone https://github.com/imagepbl/EffortSharing
```

### Setting up the environment

#### conda
We recommend to install the Conda environment also on the `K:\` drive to increase performance. Create folders `environments\effortsharing_env` on `K:\`.
Create a new conda environment with a `--prefix` in that folder. Activate it and update it the content based on the file `environment.yml`. 

```shell
cd .\environments\effortsharing_env\
conda create --prefix .
conda activate K:\environments\effortsharing_env
conda env update --name K:\environments\effortsharing_env --file K:\<FILEPATH>\environment.yml
```

#### pip

```shell
# First install the requirements
pip install -r /path/to/requirements.txt

# Then install the effortsharing package in editable mode
pip install -e .

# To update the requirements file for full reproducibility
pip freeze > requirements-lock.txt

# To install from the lock file
pip install -r /path/to/requirements.txt
pip install -e .
```

### Setting up required folder structure

You will need to make a folder+subfolder in the K:/ drive as follows: `K:/Data/Data_effortsharing/DataUpdate_ongoing`. There all the data files will appear. Most output files that are generated here are in `netcdf` (or `xarray`) format.

```shell
cd K:
mkdir Data\Data_effortsharing\DataUpdate_ongoing\Allocations
mkdir Data\Data_effortsharing\DataUpdate_ongoing\Allocations_CO2_excl
```
