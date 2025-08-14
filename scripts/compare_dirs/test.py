"""Script that compares two directories recursively and lists the differences of files.

File comparisons
- *.nc with xarray.testing
- *.npy with numpy.testing
- *.json with json.loads
- others are skipped

Run with:

pytest -vvv --confcutdir=$PWD/scripts/compare_dirs \
    scripts/compare_dirs/test.py \
    --reference-dir ./data.jul25/intermediate/  \
    --current-dir ./data/intermediate/ \
    --atol 1e-9 --rtol 1e-5
"""

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr


def compare_npy_files(ref_file, curr_file, rtol=None, atol=None):
    ref_array = np.load(ref_file)
    curr_array = np.load(curr_file)
    np.testing.assert_allclose(ref_array, curr_array, rtol=rtol, atol=atol)


def compare_json_files(ref_file, curr_file):
    with open(ref_file) as f:
        ref_data = json.load(f)
    with open(curr_file) as f:
        curr_data = json.load(f)
    assert ref_data == curr_data


def compare_nc_files(ref_file, curr_file, rtol=None, atol=None):
    ref_ds = xr.open_dataset(ref_file)
    curr_ds = xr.open_dataset(curr_file)
    try:
        xr.testing.assert_allclose(ref_ds, curr_ds, rtol=rtol, atol=atol)
    finally:
        ref_ds.close()


def compare_files(ref_file: Path, curr_file: Path, rtol: float, atol: float) -> None:
    """Compare files based on their extension."""
    suffix = ref_file.suffix.lower()

    if suffix == ".nc":
        compare_nc_files(ref_file, curr_file, rtol=rtol, atol=atol)
    elif suffix == ".npy":
        compare_npy_files(ref_file, curr_file, rtol=rtol, atol=atol)
    elif suffix == ".json":
        compare_json_files(ref_file, curr_file)
    else:
        pytest.skip(f"Unsupported file type: {suffix}")


# And finally the test itself


def test_file_content_is_identical(ref_file: Path, curr_file: Path, rtol: float, atol: float):
    compare_files(ref_file, curr_file, rtol=rtol, atol=atol)
