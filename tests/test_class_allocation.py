import unittest
import xarray as xr
from EffortSharingTools.class_allocation import allocation
import numpy as np


class TestAllocation(unittest.TestCase):

    def compare_netcdf_files(self, file1, file2):
        """
        Compare two NetCDF files and assert if they are different.

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
        except AssertionError as e:
            self.fail(f"The NetCDF files are different: {e}")

        # Close the datasets
        ds1.close()
        ds2.close()

    def test_compare_netcdf_files(self):
        """
        When making changes to the code that are not expected to change the output,
        this test should pass.
        """

        benchmark_file = r"K:\Data\Data_effortsharing\DataUpdate_ongoing\startyear_2021\Allocations_GHG_incl_benchmark\xr_alloc_USA_benchmark.nc"
        file_new = r"K:\Data\Data_effortsharing\DataUpdate_ongoing\startyear_2021\Allocations_GHG_incl\xr_alloc_USA.nc"
        self.compare_netcdf_files(benchmark_file, file_new)

    def setUp(self):
        # Set up test data and objects
        self.region = 'USA'
        self.allocation = allocation(self.region)
        self.allocation.xr_total = xr.Dataset({
            'Population': xr.DataArray(np.random.rand(10, 5), dims=['Time', 'Region']),
            'GDP': xr.DataArray(np.random.rand(10, 5), dims=['Time', 'Region']),
            'GHG_globe': xr.DataArray(np.random.rand(10), dims=['Time']),
            'GHG_hist': xr.DataArray(np.random.rand(10, 5), dims=['Time', 'Region']),
            'emis_hist': xr.DataArray(np.random.rand(10, 5), dims=['Time', 'Region']),
            'emis_base': xr.DataArray(np.random.rand(10, 5), dims=['Time', 'Region']),
            'emis_fut': xr.DataArray(np.random.rand(10, 5), dims=['Time', 'Region']),
            'rbw': xr.DataArray(np.random.rand(10), dims=['Time'])
        })
        self.allocation.analysis_timeframe = slice(0, 10)
        self.allocation.start_year_analysis = 2020
        self.allocation.dim_histstartyear = [2000, 2005, 2010]
        self.allocation.dim_discountrates = [0.0, 1.6, 2.0, 2.8]
        self.allocation.dim_convyears = [2030, 2040, 2050]
        self.allocation.focus_region = 'USA'

    def test_ap(self):
        # Test the ap function
        self.allocation.ap()
        self.assertIn('AP', self.allocation.xr_total)
        ap = self.allocation.xr_total['AP']
        self.assertEqual(ap.dims, ('Time', 'Region'))
        self.assertEqual(ap.shape, (10, 5))

        # Additional checks can be added here to verify the correctness of the calculations
        # For example, you can check specific values or ranges



if __name__ == "__main__":
    unittest.main()
