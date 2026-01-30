import numpy as np
import xarray as xr
import pyproj
from matplotlib import cm
from matplotlib.colors import ListedColormap

'''
############################
# Function make_color_cmap #
############################

Function make_color_cmap makes a white+Oranges colour map

Example: 
mycmap = make_color_cmap(nval)

Input:
nval: number of total values (1 white & nval-1 Oranges)

Output:
mycmap: white+Oranges colour map

'''

def make_color_map(nval):

    corng = cm.get_cmap('Oranges',nval)
    newcolours = corng(np.linspace(0,1,nval))
    newcolours[0] = [1, 1, 1, 1]  # RGBA for white
    my_whiteoranges = ListedColormap(newcolours)

    return my_whiteoranges


'''
#######################################
# Function cumulative_runlength       #
#######################################

Function cumulative_runlength computes the run length of consecutive
threshold exceedances along the time dimension. The run length increases
by one for each consecutive exceedance day and resets to zero when the
threshold is not exceeded.

This function is designed to be used with xarray.apply_ufunc, where the
time dimension is passed as the leading dimension of the input array.

Example:
runlen = xr.apply_ufunc(
    cumulative_runlength,
    crossing_int,
    input_core_dims=[['time']],
    output_core_dims=[['time']],
    vectorize=True,
    dask='parallelized',
    output_dtypes=[crossing_int.dtype])

Input:
arr: NumPy array with shape (time, ...), where values are 1 for
     threshold exceedance and 0 otherwise

Output:
out: NumPy array of the same shape as arr, containing the run length
     of consecutive exceedances at each time step
'''

def cumulative_runlength(arr):

    # arr is a NumPy array with shape (time, ...)
    out = np.zeros_like(arr)
    out[0] = arr[0]
    for t in range(1, arr.shape[0]):
        out[t] = arr[t] * (out[t-1] + 1)

    return out


'''
##############################
# Function make_spatial_mean #
##############################

Function make_spatial_mean gets a UKCP18 field from 
xarray and  calculates the weigthed spatial mean for 
each time point 

Example: 
fldm = make_spatial_mean(data)

Input:
data: xarray data with coordinates defined either 
      as (e.g. 12 km UKCP18 data):
         data.projection_x_coordinate
         data.projection_y_coordinate
      or (e.g. 2.2 km UKCP18 data):
         data.grid_longitude
         data.grid_longitude

Output:
fldm: array with weigthed mean values

*NOTE: Data must be a single variable with coords [time, x, y]

'''

def make_spatial_mean(data):

    # Detect the grid coordinate names
    if "projection_x_coordinate" in data.coords:
        x = data.projection_x_coordinate
        y = data.projection_y_coordinate
        # Input CRS
        crs_native = pyproj.CRS.from_epsg(27700)
    else:
        # High-res 2.2km rotated grid
        x = data.grid_longitude
        y = data.grid_latitude
        # Rotated pole lon/lat
        pole_lat = 37.5
        pole_lon = 177.5
        # Input CRS
        crs_native = pyproj.CRS.from_cf({
            "grid_mapping_name": "rotated_latitude_longitude",
            "grid_north_pole_latitude": pole_lat,
            "grid_north_pole_longitude": pole_lon })

    # Output CRS = WGS84 (lon/lat)
    wgs84 = pyproj.CRS.from_epsg(4326)

    # Build transformer
    transformer = pyproj.Transformer.from_crs(crs_native, wgs84, always_xy=True)

    # If 2D arrays, flatten and then reshape
    lon2d, lat2d = np.meshgrid(x, y)  # shape (y, x)
    lon, lat = transformer.transform(lon2d, lat2d)

    # Cos(lat) weights
    weights = np.cos(np.deg2rad(lat))
    weights = xr.DataArray(weights, dims=(y.dims[0], x.dims[0]))

    #  Only keep weights where data is valid
    valid = xr.where(np.isnan(data), np.nan, 1)
    weights_valid = weights * valid
 
    # Weighted mean
    wmean = (data * weights_valid).sum(dim=[y.dims[0], x.dims[0]]) / \
            weights_valid.sum(dim=[y.dims[0], x.dims[0]])

    return wmean


'''
#######################
# Function gwl_ukcp18 #
#######################

Function gwl_ukcp18 takes as inputs a) a DataArray of a threshold metric
created with UKCP18 data, b) the number of the ensemble member and c) a
global warming level (GWL) and creates the metric field for that level. 
Each level corresponds to pre-computed  20-year time slice. The first year
of these time slices is specified in a dictionary within the function.
The output is a metric field for the GWL requested, computed as the mean
value of the 20-year time slice.

Examples: 
data_gwl = gwl_ukcp18(data, ens, gwl)
data_gwl = gwl_ukcp18(data, ens, gwl, output_file = 'output_file.nc')

Input:
data: xarray data with spatio-temporal coordinates. The years in the data
      must overlap with the time slice of the requested GWL
ens: the number of the UKCP18 ensemble member
gwl: the GWL - must be one of 1., 1.5, 2, 2.5, 3., 3.5, 4.

Output:
data_gwl: DataArray with the field for the requested GWL
output_file(optional): name of a netcdf file to save the output   

'''

def gwl_ukcp18(data, ens, gwl, output_file = None):

    # Start year of the 20-year time slice for each GWL
    gwl_period = {
        1:{1:1995, 1.5:2008, 2:2018, 2.5:2029, 3:2037, 4:2052},
        4:{1:1994, 1.5:2005, 2:2015, 2.5:2025, 3:2034, 4:2048},
        5:{1:1996, 1.5:2008, 2:2020, 2.5:2031, 3:2040, 4:2055},
        6:{1:1994, 1.5:2007, 2:2017, 2.5:2027, 3:2036, 4:2052},
        7:{1:1994, 1.5:2006, 2:2019, 2.5:2030, 3:2038, 4:2052},
        8:{1:1995, 1.5:2007, 2:2020, 2.5:2031, 3:2041, 4:2058},
        9:{1:1992, 1.5:2003, 2:2016, 2.5:2025, 3:2032, 4:2045},
       10:{1:1995, 1.5:2009, 2:2020, 2.5:2030, 3:2038, 4:2055},
       11:{1:1993, 1.5:2006, 2:2017, 2.5:2027, 3:2036, 4:2052},
       12:{1:1998, 1.5:2012, 2:2023, 2.5:2033, 3:2042, 4:2055},
       13:{1:1993, 1.5:2005, 2:2017, 2.5:2027, 3:2037, 4:2052},
       15:{1:1993, 1.5:2007, 2:2020, 2.5:2032, 3:2041, 4:2056},
       23:{1:1997, 1.5:2012, 2:2028, 2.5:2036, 3:2044, 4:2064},
       25:{1:1993, 1.5:2009, 2:2024, 2.5:2035, 3:2043, 4:2060},
       27:{1:2008, 1.5:2027, 2:2038, 2.5:2049, 3:2060, 4:None},
       29:{1:1999, 1.5:2017, 2:2031, 2.5:2044, 3:2056, 4:2075}}

    # Check if inputs are correct and find the 1st year of the GWL
    if ens not in [1,4,5,6,7,8,9,10,11,12,13,15,23,25,27,29]:
        raise ValueError(f'Invalid ensemble member number: {ens}.')
    if gwl not in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]:
        raise ValueError(f'Invalid GWL specification: {gwl}.')
    if gwl_period[ens][gwl] is None:
        raise ValueError(f'Year(s) of the GWL {gwl} time slice missing.')
    else:
        # Start year for selected ensemble member and GWL
        y1 = gwl_period[ens][gwl]
    if not (data.year.min() <= y1 and data.year.max() >= y1+19):
        raise ValueError(f'Year(s) of the GWL {gwl} time slice missing.')

    # Extract time-slice and return the mean
    data_gwl = data.sel(year = slice(y1, y1+19)).mean(dim = 'year')

    # Save to netCDF if requested
    if output_file is not None:
        data_gwl.to_netcdf(output_file)

    return data_gwl
