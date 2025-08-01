# Effort Sharing

Compute fair national emissions allocations using transparent, reproducible workflows. Designed by researchers for researchers.


- [Introduction](#introduction)
- [Installation Instructions](#installation-instructions)
- [Obtaining Input Data](#obtaining-input-data)
- [Usage Overview](#usage-overview)
  - [Command Line Interface (CLI)](#command-line-interface-cli)
  - [Scripts](#scripts)
  - [Interactive Notebooks \& API](#interactive-notebooks--api)
  - [Configuration File](#configuration-file)
- [Developer Instructions](#developer-instructions)
  - [Source Installation](#source-installation)
  - [Code Style / Formatting](#code-style--formatting)
  - [Documentation](#documentation)
  - [Testing](#testing)
  - [Making a Release](#making-a-release)
- [Referencing this Repository](#referencing-this-repository)

## Introduction

This package combines a variety of data sources to compute fair national emissions allocations, study variability, and compare results with NDC estimates and cost-optimal scenario projections.

- Gather country-level data (population, GDP, historical emissions, etc.)
- Compute global future emission pathways based on configurable scenarios
- Calculate allocations for countries/regions using various effort-sharing rules
- Compare allocations, NDCs, and cost-optimal scenarios
- Conduct variance decomposition (Sobol analysis)

## Installation Instructions

If you want to use some of the existing functionality, simply install from PyPI:

```shell
pip install effort-sharing
```

If you plan to develop the code or modify notebooks/scripts, install from source
(see [Developer Instructions](#developer-instructions)).

## Obtaining Input Data

We're working to provide input data directly or via original sources. For now, contact <mailto:mark.dekker@pbl.nl> for quick access.

## Usage Overview

### Command Line Interface (CLI)

Run step-by-step workflows using the CLI:

```shell
effortsharing --help
effortsharing generate-config
effortsharing get-input-data
effortsharing global-pathways
effortsharing policy-scenarios
effortsharing allocate NLD
effortsharing aggregate 2040
# Overwrite defaults:
effortsharing --config config.yml --log-level WARNING allocate NLD --gas CO2 --lulucf excl
```

The CLI lets you quickly run complete workflows or focus on specific countries/years.

### Scripts

Automate workflows using scripts in the `scripts` folder.  
Example: `scripts/cabe_export.py` loads data, calculates pathways/allocations, aggregates results, and exports everything.  
Edit parameters at the top of the script, then run with:

```shell
python scripts/cabe_export.py
```

Scripts are not included in the PyPI package.

### Interactive Notebooks & API

Import high-level and low-level functions in Python or Jupyter notebooks for custom analysis and visualization.  

The internal structure of the effortsharing package is documented in [apidocs.md](apidocs.md). While unpolished, it may serve as a starting point when diving into the internals of the code. 

See the `notebooks` folder for examples.  
Note: Notebooks may be outdated as the package evolves, but dedicated releases ensure reproducibility for published results.

### Configuration File

Many commands require a configuration file.  
Generate a default config:

```shell
effortsharing generate-config
```

The config file controls:

- Data paths
- Whether to load/save intermediate results
- Experiment parameters
- Dimension ranges for pathway parameters (shorter ranges = faster runs, but less variability)

See comments in the generated config for details.

## Developer Instructions

If you plan to contribute, please follow these guidelines and respect the [code of conduct](CODE_OF_CONDUCT.md).

### Source Installation

Clone the repo and set up a (conda) environment:

```shell
git clone https://github.com/imagepbl/effort-sharing
cd effort-sharing
conda env create --file environment.yml
conda activate effortsharing_env
conda env update -f environment.yml
```

For reproducibility, use the conda-lock file:

```shell
conda-lock lock
conda-lock install --name effortsharing_env
pip install -e .[dev]  # conda-lock doesn't install local libraries
```

### Code Style / Formatting

We use [ruff](https://docs.astral.sh/ruff/) for linting and formatting.  
Configuration is in `pyproject.toml`.

```shell
ruff check src
ruff check --fix src
ruff format src
```

VS Code users: install the ruff plugin for in-editor feedback.

### Documentation

API docs are generated from docstrings using [pydoc-markdown](https://niklasrosenstein.github.io/pydoc-markdown/):

```shell
pydoc-markdown -I src --render-toc > apidocs.md
```

### Testing

A test script is included to quickly check if / how the results have been affected since a previous run. Use it as such:

```shell
python scripts/compare_dirs.py data/reference data/current
``` 

where you replace the paths to reference and current with actual folders you
want to compare.

### Making a Release

We release at least when publishing new results (e.g. journal, Carbon Budget Explorer). To generate a new release, open a [new issue using the `release` template](https://github.com/imagepbl/effort-sharing/issues/new?template=01_release.md) and follow the steps in the checklist.

## Referencing this Repository

Cite the code: ...

Output data is publicly available on Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12188104.svg)](https://doi.org/10.5281/zenodo.12188104)

