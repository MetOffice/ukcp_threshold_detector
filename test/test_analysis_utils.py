import os
import pytest
import tempfile
import numpy as np
import xarray as xr

"""
Unit tests for functions in analysis_utils.py.

Synthetic NetCDF files are created in temporary directories to test
function behaviour under different scenarios. Temporary files are
cleaned up after the tests run.
"""

from analysis_utils import find_years_to_analyse, extract_year_data

# Dummy class to mimic ThresholdDetector
class DummyDetection:
    def __init__(self, indata, var="var"):
        self.indata = indata
        self.var = var

def create_test_file(path, start_year, end_year):
    time = np.arange(f"{start_year}-01-01", f"{end_year}-12-31", dtype="datetime64[D]")
    ds = xr.Dataset({"var": ("time", np.zeros(len(time)))}, coords={"time": time})
    ds.to_netcdf(path)


#######################################
# Tests for find_years_to_analyse     #
#######################################

# Test basic case
def test_find_years_basic():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create fake NetCDF files
        file1 = os.path.join(tmpdir, "file1.nc")
        file2 = os.path.join(tmpdir, "file2.nc")

        create_test_file(file1, 2000, 2002)
        create_test_file(file2, 2003, 2005)

        # Dummy detection object
        det = DummyDetection(indata=tmpdir + "/")

        file_years, select_years = find_years_to_analyse(
            det, select_years=None, years_from_dec=False)

        # --- Assertions ---
        assert len(file_years) == 2
        assert min(select_years) == 2000
        assert max(select_years) == 2005

# Test invalid input
def test_select_years_type_error():
    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = os.path.join(tmpdir, "file.nc")
        create_test_file(file1, 2000, 2001)

        det = DummyDetection(indata=tmpdir + "/")
        with pytest.raises(TypeError):
            find_years_to_analyse(det, select_years="not_a_list", years_from_dec=False)

# Test years_from_dec=True
def test_years_from_dec_shift():
    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = os.path.join(tmpdir, "file.nc")
        create_test_file(file1, 2000, 2002)

        det = DummyDetection(indata=tmpdir + "/")
        _, select_years = find_years_to_analyse(
            det, select_years=[2001, 2002], years_from_dec=True)

        assert list(select_years) == [2000, 2001]


#######################################
# Tests for extract_year_data         #
#######################################

# Test basic case
def test_extract_year_basic():
    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = os.path.join(tmpdir, "file.nc")
        create_test_file(file1, 2000, 2002)
        det = DummyDetection(indata=tmpdir + "/")

        file_years = {file1: (2000, 2002)}
        year_data = extract_year_data(det, 2001, file_years, years_from_dec=False)

        # Check result is not empty and only contains 2001
        assert year_data.sizes["time"] > 0
        assert all(year_data.time.dt.year.values == 2001)

# Test multiple files
def test_extract_year_multiple_files():
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = os.path.join(tmpdir, "file1.nc")
        file2 = os.path.join(tmpdir, "file2.nc")

        create_test_file(file1, 2000, 2001)
        create_test_file(file2, 2001, 2002)

        det = DummyDetection(indata=tmpdir + "/")

        file_years = {
            file1: (2000, 2001),
            file2: (2001, 2002),}
        year_data = extract_year_data(det, 2001, file_years, years_from_dec=False)

        assert year_data.sizes["time"] > 0
        assert all(year_data.time.dt.year.values == 2001)

# Test December-based year
def test_extract_year_from_dec():
    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = os.path.join(tmpdir, "file.nc")
        create_test_file(file1, 2000, 2002)

        det = DummyDetection(indata=tmpdir + "/")

        file_years = {file1: (2000, 2002)}
        year_data = extract_year_data(det, 2001, file_years, years_from_dec=True)

        # Should include Dec 2001 and Jan–Nov 2002
        years = year_data.time.dt.year.values

        assert 2001 in years
        assert 2002 in years
