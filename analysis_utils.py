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

    import numpy as np
    from matplotlib import cm
    from matplotlib.colors import ListedColormap

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

    import numpy as np

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

    import xarray as xr
    import numpy as np
    import pyproj

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
