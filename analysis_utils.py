'''
############################
# MODULE analysis_utils.py #
############################

Utility functions used by the Threshold Detector

   . Function make_color_cmap: creates a custom colour map
   . Function cumulative_runlength: computes the run length of consecutive threshold crossings
   . Function make_spatial_mean: computes the weigthed spatial mean of spatial fields
   . Function gwl_ukcp18: computes a selected Global Warming Level (GWL)
   . Function nc2tif_ukcp18: converts netcdf files tif
   . Function apply_spatial_smoothing: spatialy smooths the ThresholdDetector output metric

'''

import numpy as np
import xarray as xr
import pyproj
import rasterio
from rasterio.transform import from_origin
import cartopy.crs as ccrs
from matplotlib import cm
from matplotlib.colors import ListedColormap


def make_color_map(nval):
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

    corng = cm.get_cmap('Oranges',nval)
    newcolours = corng(np.linspace(0,1,nval))
    newcolours[0] = [1, 1, 1, 1]  # RGBA for white
    my_whiteoranges = ListedColormap(newcolours)

    return my_whiteoranges


def cumulative_runlength(arr):
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

    # arr is a NumPy array with shape (time, ...)
    out = np.zeros_like(arr)
    out[0] = arr[0]
    for t in range(1, arr.shape[0]):
        out[t] = arr[t] * (out[t-1] + 1)

    return out


def make_spatial_mean(data):
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

    *NOTE 1: Data must be a single variable with coords [time, x, y]
    *NOTE 2: The function depends on the coordinate system and was developed
    for UKCP18 gridded data (but may also be applied to HadUKGrid). The
    code may not work when applied to datasets with different coordinate 
    systems, or coordinate names.

    '''

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
    weights_valid = weights * xr.where(np.isnan(data), np.nan, 1)

    # Weighted mean
    wmean = (data * weights_valid).sum(dim=[y.dims[0], x.dims[0]]) / \
            weights_valid.sum(dim=[y.dims[0], x.dims[0]])

    return wmean


def gwl_ukcp18(data, ens, gwl, output_file = None):
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
    # Start year for selected ensemble member and GWL
    y1 = gwl_period[ens][gwl]
    if not (data.year.min() <= y1 and data.year.max() >= y1+19):
        raise ValueError(f'Year(s) of the GWL {gwl} time slice missing.')

    # Extract time-slice and return the mean
    data_gwl = data.sel(year = slice(y1, y1+19)).mean(dim = 'year')
    data_gwl.attrs.update(data.attrs)

    # Save to netCDF if requested
    if output_file is not None:
        data_gwl.to_netcdf(output_file)

    return data_gwl


def nc2tif_ukcp18(ncfile):
    '''
    ##########################
    # Function nc2tif_ukcp18 #
    ##########################

    Function nc2tif_ukcp18 converts netcdf output (.nc) from the Threshold 
    Detector to GeoTiff (.tif) for GIS applications. The input is the name
    of the netcdf file, which the function converts to GeoTiff and saves the 
    converted output into the working directory. If the input file has several
    years of spatial data (3D output), then the function creates separate
    GeoTiff files, one for each year. Output files keep the same name as the 
    input file, but with extension .tif instead of .nc, and if several years
    are transformed, then the corresponding files also have the year 
    information in the name (as "_YYYY.tif").

    Example: 
    nc2tif_ukcp18(ncfile)

    Input:
    ncfile: the netcdf file with the input data (e.g. 'input_data.nc')

    Output:
    The function transforms the input file to .tif and saves the output
    in the working directory.

    *NOTE: The function depends on the coordinate system and was developed
    for threshold detections created with UKCP18 data (but may also be used
    for HadUKGrid based detctions). The code may not work when applied to
    datasets with different coordinate systems, or coordinate names.

    '''

    # Open dataset
    dataset = xr.open_dataset(ncfile)
    data = dataset[list(dataset.data_vars)[0]]

    # Select ensemble member if in dims
    if 'ensemble_member' in data.dims:
        data = data.sel(ensemble_member = data.ensemble_member.item())

    # Determine years to export
    if 'year' in data.dims:
        years = list(data.year.values)
    else:
        years = [None]

    # Determine coordinate type
    coords = data.coords
    if 'projection_x_coordinate' in coords and 'projection_y_coordinate' in coords:
        # Case 1: standard UKCP projection coordinates
        x = data.projection_x_coordinate.values
        y = data.projection_y_coordinate.values
        crs = pyproj.CRS.from_epsg(27700)  # Already in British National Grid
    elif 'grid_latitude' in coords and 'grid_longitude' in coords:
        # Case 2: rotated-pole high-res UKCP
        rot_lat = data.grid_latitude.values
        rot_lon = data.grid_longitude.values
        pole_lat = 37.5
        pole_lon = 177.5
        rot_crs = ccrs.RotatedPole(pole_latitude=pole_lat, pole_longitude=pole_lon)
        rot_crs_proj = pyproj.CRS.from_wkt(rot_crs.to_wkt())
        # Target CRS: British National Grid
        target_crs = pyproj.CRS.from_epsg(27700)
        # Create transformer
        transformer = pyproj.Transformer.from_crs(rot_crs_proj, target_crs, always_xy=True)
        # Make 2D meshgrid of rotated coordinates
        lon2d, lat2d = np.meshgrid(rot_lon, rot_lat)
        # Transform rotated coordinates to target CRS
        x2d, y2d = transformer.transform(lon2d, lat2d)
        # Take first row/column to define 1D x and y arrays for raster writing
        x = x2d[0, :]      # x along columns
        y = y2d[:, 0]      # y along rows
        crs = target_crs
    else:
        raise ValueError("Unknown coordinate system in data")

    # Perform the transformation. If many years produce one file per year
    for yr in years:
        # Select year (for 3D fields)
        if yr is None:
            da = data.squeeze()
        else:
            da = data.sel(year=yr).squeeze()
        assert len(da.dims) == 2, 'DataArray must be 2D for GeoTIFF output'
        # Reorder y-coord if necessary to be north-up
        if y[0] < y[-1]:
            data_vals = da.values[::-1, :]
            y_flipped = y[::-1]
        else:
            data_vals = da.values
            y_flipped = y
        # Pixel sizes (always positive)
        xsize = abs(float(x[1] - x[0]))
        ysize = abs(float(y_flipped[1] - y_flipped[0]))
        west = float(x.min())
        north = float(y_flipped.max())
        transform = from_origin(west, north, xsize, ysize)
        # Output filename
        if yr is None:
            output_file = ncfile.rsplit('.',1)[0]+'.tif'
        else:
            output_file = ncfile.rsplit('.',1)[0]+'_'+str(yr)+'.tif'
        # Write GeoTIFF
        with rasterio.open(
            output_file,
            'w',
            driver='GTiff',
            height=data_vals.shape[0],
            width=data_vals.shape[1],
            count=1,
            dtype=data_vals.dtype,
            crs=crs,
            transform=transform,
            nodata=np.nan
        ) as dst:
            dst.write(data_vals, 1)
            dst.update_tags(
                variable=data.name,
                long_name=data.attrs.get("long_name", ""),
                units=data.attrs.get("units", ""),
                standard_name=data.attrs.get("standard_name", ""))


def apply_spatial_smoothing(data, box_size=3, output_file = None):
    '''
    ####################################
    # Function apply_spatial_smoothing #
    ####################################

    Function apply_spatial_smoothing takes as input a DataArrary of a threshold
    metric created by the ThresholdDetector and applies spatial smoothing to all
    fields by replacing each grid-point value by the mean of the NxN grid boxes
    around it. Default box-size N is set to 3 (i.e. smoothing uses the mean of
    3x3=9 values around the grid-point).

    Examples: 
    smoothed_data = apply_spatial_smoothing(data)
    smoothed_data = apply_spatial_smoothing(data, box_size=5, 
                    output_file = 'smoothed_data.nc')

    Input:
    data: a DataArray with the threshold-crossing metric
    box_size (optional): the size of the box used for smoothing (default = 3)

    Output:
    smoothed_data: a DataArray with the smoothed metric
    output_file(optional): name of a netcdf file to save the output 

    *NOTE: The function depends on the coordinate names of the input field and
     was developed for UKCP18 data (but may also be used with HadUKGrid). The
     code will not work when applied to datasets with different coordinate names.

    '''

    # Smooth input field
    if 'projection_x_coordinate' in data.coords and \
       'projection_y_coordinate' in data.coords:
        data_smoothed = (data.rolling(projection_y_coordinate = box_size,
                                      projection_x_coordinate = box_size,
                                      center = True, min_periods = 1).mean())
    elif 'grid_latitude' in data.coords and \
         'grid_longitude' in data.coords:
        data_smoothed = (data.rolling(grid_latitude = box_size,
                                      grid_longitude = box_size,
                                      center = True, min_periods = 1).mean())
    else:
        raise ValueError("Unknown coordinate system in data")

    # Mask by the input field
    data_smoothed = data_smoothed.where(data.notnull())

    # Save to NetCDF if requested
    if output_file is not None:
        data_smoothed.to_netcdf(output_file)

    return data_smoothed
