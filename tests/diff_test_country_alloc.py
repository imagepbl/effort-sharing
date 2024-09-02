import xarray as xr

def compare_netcdf_files(file1, file2):
    """
    Compare two NetCDF files and print the differences.

    Parameters:
    file1 (str): Path to the first NetCDF file.
    file2 (str): Path to the second NetCDF file.
    """
    # Load the datasets
    ds1 = xr.open_dataset(file1)
    ds2 = xr.open_dataset(file2)

    # Compare the datasets
    try:
        xr.testing.assert_equal(ds1, ds2)
        print("The NetCDF files are identical.")
    except AssertionError as e:
        print("The NetCDF files are different.")
        print(e)

    # Close the datasets
    ds1.close()
    ds2.close()

# Example usage
file_benchmark = r"K:\data\DataUpdate_08_2024\xr_alloc_USA.nc"
# file2 = r"K:\data\Data_old\xr_alloc_USA.nc"
# file_new = r"K:\data\DataUpdate_ongoing\Allocations\xr_alloc_USA.nc"
file_new = r"K:\data\DataUpdate_ongoing\Allocations\xr_alloc_USA_CO2.nc"
compare_netcdf_files(file_benchmark, file_new)
