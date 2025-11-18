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

    # Weighted mean
    wmean = (data * weights).sum(dim=[y.dims[0], x.dims[0]]) / weights.sum()

    return wmean
