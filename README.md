## Installation

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
pip install -r /path/to/requirements_pip.txt
```

### Setting up required folder structure

You will need to make a folder+subfolder in the K:/ drive as follows: `K:/ECEMF/T5.2/Allocations/`. There (and in K:/ECEMF/T5.2) all the data files will appear. Most output files that are generated here are in `netcdf` (or `xarray`) format.

```shell
cd K:
mkdir ECEMF\T5.2\Allocations
```

## Main features
The main notebook you will be using is `Main.ipynb` in the main directory. Each cell in that notebook loads in a particular class and then goes through all of its methods. The first one being called is the class in the file `class_datareading.py`. This class gathers all data from external sources, such as population, GDP, historical emissions, etc. on a country level, but also computes global pathways into the future. It does not do any effort sharing. The output is a file called `xr_dataread.nc`, which is your friend in case you need any country-specific data for debugging, for example. The main uncertainties / work from our side is around generating those future pathways under various circumstances (peak temperature, risk, non-CO2 assumptions, etc.).

The second class being called is the one from `class_allocation.py`. That one does all the effort-sharing. It loads in `xr_dataread.nc` and its methods are named after the allocation rule being computed. The class requires an input: the region (ISO3) that is focused on (called `self.FocusRegion` in `class_allocation.py`). So in `Main.ipynb`, you see that there is a loop over all countries and even some country groups (e.g., G20) in the world. The output are files in the form of `xr_alloc_XYZ.nc` where XYZ is the ISO3 code for a country, and they are saved to K:/ECEMF/T5.2/Allocations/.

The third and fourth classes called in `Main.ipynb` are associated with loading in ENGAGE scenarios and evaluating NDCs. Not relevant right now and mostly used in the paper.

One final important notebook is `Aggregator.ipynb`. This script aggregates the country-individual files `xr_alloc_XYZ.nc`. One output is the aggregation of those files into a single file for a single year, for example `xr_alloc_2030.nc`. Useful for analysis purposes.

Input variables can be changed in `input.yml`.

## Workflow
Please feel free to add and/or fix any of the issues on the Github issue list if you have time to contribute. But if you adjust code, please create your own branch or work in one of the existing ones - and use pull requests to merge them into the `main` branch. Ask someone (for now Chantal or me) to review it before merging, there is a button for that.

## Data exports for PBL plots
I (Mark) will be working on the data exports for the figures in the PBL reports shortly. Will give a description here in due time. UPDATE: a first script for data exports is now under review in branch `data-exports`, called `DataExports.ipynb`.

## Releases
No release of this code has been done yet. First is planned in March/April, together with the PBL report, the launch of the carbon budget explorer (CABE) version 1.0 and the preprint of this research. We should appropriately clean up the repo, this ReadMe in particular, and then release the code via a button within Github towards Zenodo.

## Plotting
All plotting scripts concerning this research are also in this repository. In order to save figures from these scripts, create a `Figures` folder in your local repository directory and the appropriate subfolders (not to be added to the github). In particular, those for the paper we are writing (location: `/Plotting/ECEMF_paper/`), but also those for the PBL report (location: `/Plotting/PBL_report/`). The latter are of course the ones that we created ourselves as concepts, not the ones from the visualisation team. One of these plots can be found as an example here, but is slowly being updated iteratively as the code develops:

![Fig_NLD_nonumbers](https://github.com/imagepbl/EffortSharing/assets/47416602/6db2557d-db11-4457-9ba3-adc9a8ea1a66)
