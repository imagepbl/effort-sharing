import functools
from pathlib import Path


import logging


# Format-specific loader functions
def load_netcdf(path):
    import xarray as xr

    return xr.load_dataset(path)


def load_csv(path):
    import pandas as pd

    return pd.read_csv(path)


def load_parquet(path):
    import pandas as pd

    return pd.read_parquet(path)


def load_json(path):
    import json

    with open(path) as f:
        return json.load(f)


# Format-specific saver functions
def save_netcdf(data, path):
    data.to_netcdf(path)
    # TODO: add compression


def save_csv(data, path):
    data.to_csv(path, index=False)


def save_parquet(data, path):
    data.to_parquet(path, index=False)


def save_json(data, path):
    import json

    with open(path, "w") as f:
        json.dump(data, f)


# Separate lookup tables for loaders and savers
LOADERS = {
    ".nc": load_netcdf,
    ".csv": load_csv,
    ".parquet": load_parquet,
    ".json": load_json,
}

SAVERS = {
    ".nc": save_netcdf,
    ".csv": save_csv,
    ".parquet": save_parquet,
    ".json": save_json,
}


def intermediate_file(filename, loader=None, saver=None):
    """Decorator for caching function results to/from intermediate files.

    Args:
        filename: Name of the intermediate file to use
        loader: Optional custom loader function, otherwise determined by file extension
        saver: Optional custom saver function, otherwise determined by file extension

    The decorated function should accept a config object as argument that should
    contain the following parameters:

    - path.intermediate: path to intermediate data directory.
    - load_intermediate_files: Whether to load from cache if available (default: True)
    - save_intermediate_files: Whether to save results to cache (default: True)

    These parameters are automatically available in the wrapper scope and should
    be passed down to any nested decorated functions to maintain consistent behavior.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(config, *args, **kwargs):
            # Get logger from the decorated function's module
            logger = logging.getLogger(func.__module__)

            path = Path(config.paths.intermediate) / filename
            ext = path.suffix

            # Get loaders/savers or use provided ones
            load_fn = loader or LOADERS.get(ext)
            save_fn = saver or SAVERS.get(ext)

            if not load_fn or not save_fn:
                raise ValueError(f"No loader/saver for extension {ext}")

            # Try to load if requested
            if config.load_intermediate_files and path.exists():
                logger.info(f"Loading intermediate data from {path}")
                return load_fn(path)

            # Compute result
            logger.info(f"Computing data for {path.name}")
            result = func(config, *args, **kwargs)

            # Save if requested
            if config.save_intermediate_files:
                logger.info(f"Saving intermediate data to {path}")
                path.parent.mkdir(parents=True, exist_ok=True)
                save_fn(result, path)

            return result

        return wrapper

    return decorator
