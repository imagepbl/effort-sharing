## Introduction

This code combines a variety of data sources to compute fair national emissions allocations, studies variability in these allocations and compares them with NDC estimates and cost-optimal scenario projections. 

* Gather data from external sources such as population, GDP, historical emissions, etc. on a country level
* Compute global future emission pathways based on configurable emission reduction scenarios
* Calculate corresponding allocations for individual countries/regions based on various effort-sharing rules
* Combine allocations for all regions for a given target year
* Load cost-optimal scenarios and NDCs in the same format for easy comparison
* Conduct variance decomposition (Sobol analysis)

The output data of this code is publicly available on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12188104.svg)](https://doi.org/10.5281/zenodo.12188104)

## Installation instructions

The effort-sharing package is available on PyPI and can be installed with pip:

```shell
pip install effort-sharing
```

If you are planning to actively develop the code in this repository (e.g. add or modify notebooks, or modifying the core algorithms), you should install the package from source. See the [developer instructions](#developer-instructions) below.

## Obtaining input data

The effortsharing package combines data from various sources. We are currently exploring whether we can re-share the input data as a complete package, or provide direct access to (most of) the original sources. In the meantime, if you want to quickly get started, please reach out to <mailto:mark.dekker@pbl.nl>.

## Using the package

Currently, the code is optimized for running end-to-end workflows for producing data / figures for publications and the [Carbon Budget Explorer](https://www.carbonbudgetexplorer.eu). These workflows are available in the `scripts` and `notebooks` folder. Recently, we've been working to make the code more flexible to facilitate more interactive workflows as well. Consequently, we briefly document the following use cases:

### Running complete workflows as script

For example, `scripts/cabe_export.py` loads all input data, calculates global pathways, calculates allocations for all countries and aggregations for 2030 and 2040, collects policy scenarios, and writes all output to a dedicated folder. Adapt the parameters at teh top of the script as you see fit, then invoke with `python cabe_export.py` or run it from within your favourite editor. Scripts are stored in the folder `scripts` in the root of the repo. They are not part of the "package" that's installed from PyPI.

### (Interactive) exploration via the command line

The end-to-end workflows combine a number of high level steps. The effort-sharing package exposes a simple command line interface to run these steps independently. After installing the package, they are available as effortsharing:

```shell
effortsharing --help
effortsharing generate-config
effortsharing get-input-data
effortsharing global-pathways
effortsharing policy-scenarios
effortsharing allocate NLD
effortsharing aggregate 2040
# You can also overwrite defaults 
effortsharing allocate NLD --config config.yml --log-level WARNING --gas CO2 --lulucf excl
```

This simple command line interface allows you to quickly run part of the full workflow, e.g. to find the allocations for a given country of interest. Especially the first step (global_pathways) is useful, as this takes quite long, and all other functionality depend on it.

### Interactive exploration in a notebook or (I)Python shell

The high-level functions used by the command line scripts, as well as some of the lower level functions, can also be imported in an interactive Python session. This is especially convenient for analyzing results, generating visualizations, documenting workflows for publications, et cetera.

All functions that can be imported are documented [here](...). Additionally, the `notebooks` folder in the root of the repository contains various analyses that we've conducted in the past. Please note that the package was developed in tandemw with these notebooks, and we do not intend to maintain the notebooks as the package evolves further. So, while the notebooks may serve as a starting point for exploration, most of them will usually be outdated. We *do* intend to make a dedicated release whenever we publish results generated from these notebooks. In that case, the results are reproducible by rewinding to the version that was associated with this publication.

### Config file

The package uses a configuration file to store some important settings. 

TODO: insert documentation.

## Developer instructions

If your name is Mark Dekker and/or you are planning to actively contribute to this package or the scripts/notebooks, please take note of the following contribution guidelines and respect the [code of conduct](CODE_OF_CONDUCT.md).

### Source installation

To run the code as you develop it, you will want to follow a slightly different installation procedure, cloning the repo and installing inside a (conda) virtual environment. The specification in `environment.yml` contains a relatively complete set of dependencies including things like Jupyter Lab and matplotlib, that are needed for running the notebooks. For best performance in PBL werkomgeving, we recommend working from the K:/ drive.

```shell
# Clone the repo 
git clone https://github.com/imagepbl/effort-sharing
cd effort-sharing

# Create the environment
conda env create --file environment.yml

# Activate the environment
conda activate effortsharing_env

# To update the existing environment with any changes in environment.yml
conda env update -f environment.yml
```

For full reproducibility across platforms, we also maintain a conda-lock file. When adding new dependencies, make sure to add re-generate the lockfile:

```shell
# Use conda-lock to export to/install from a reproducible package list
conda-lock lock

# Install from lock file
conda-lock install name effortsharing_env
pip install -e .[dev]  # conda-lock doesn't install local **libraries**
```

### Code style / formatting

We try to maintain a consistent code style, using [ruff](https://docs.astral.sh/ruff/) as our go-to linter and formatter. Ruff configuration is defined in the `ruff` section of `pyproject.toml`. To check / fix the code you can run:

```shell
# Lint all files in the src directory
ruff check src

# Try to apply automatic fixes for found issues
ruff check --fix src

# Format the code according to guidelines
ruff format src
```

We highly recommend using the ruff plugin in VS Code. It will automatically warn you about code style issues and help you solve them from within your editor.

### Making a release

In general, we intend to make new releases at least when we publish results based on updated data (e.g. in a journal or on the Carbon Budget Explorer website). Additional releases can be made as we see fit. 

As of August 2025, we will use date versioning as in xarray, e.g. 2025.8.1 would
correspond to the first release in August 2025. To make a new release:

- [ ] Update version in pyproject.toml 
- [ ] Make sure min python version in pyproject.toml is still (reasonably) up to date; update if needed.
- [ ] Make sure python version in environment.yaml is (reasonably) up to date; update if needed.
- [ ] Make sure any newly introduced dependencies have been added
  - [ ] Dependencies in `src` folder should be in `pyproject.toml`
  - [ ] Additional dependencies in `scripts` and `notebooks` folders should be in `environment.yaml`
  - [ ] Regenerate the conda lock file
- [ ] Run the tests (if any)
- [ ] Make sure all scripts, command lines arguments, and relevant notebooks (still) run without problems
- [ ] If there are new contributors, make sure to add them to CITATION.CFF
- [ ] Create a release on GitHub and wait for the GitHub actions workflow to finish
- [ ] Verify that the release is available on PyPI, and that you can install it (`pip install effort-sharing`)
- [ ] Verify that the release is available on Zenodo
- [ ] 

## Referencing this repository

...


# TODO
- move config.yml to root of repo?
- check cli commands 
  - now they're grouped under effortsharing, do we really want this?
  - do we want to add things like help, version
  - do we want to add the CLI for policy scenarios et cetera that I've now included
  - do we want separate CLI commands for allocate_region and allocate_all?
- generate API docs and insert link.
- document the config file and insert link/inline docs
- TODO: flip switch in zenodo and add CFF for citation info


Update python version also in github action
