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

if __name__ == "__main__":
    # file_benchmark = r"K:\data\DataUpdate_08_2024\xr_alloc_USA.nc"
    benchmark_file = r"K:\Data\Data_effortsharing\DataUpdate_ongoing\\startyear_2021\Allocations_GHG_incl_benchmark\xr_alloc_USA_benchmark.nc"
    # file2 = r"K:\data\Data_old\xr_alloc_USA.nc"
    # file_new = r"K:\data\DataUpdate_ongoing\Allocations\xr_alloc_USA.nc"
    file_new = r"K:\Data\Data_effortsharing\DataUpdate_ongoing\startyear_2021\Allocations_GHG_incl\xr_alloc_USA.nc"

    compare_netcdf_files(benchmark_file, file_new)
