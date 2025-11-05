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
data: xarray data with coordinates defined as
      data.projection_x_coordinate
      data.projection_y_coordinate

Output:
fldm: array with weigthed mean values

*NOTE: Data must be a single variable with shape [time, x, y]

'''

def make_spatial_mean(data):

    import xarray as xr
    import numpy as np
    import pyproj

    # UKCP18 12km projection
    tm_proj = pyproj.CRS.from_epsg(27700)  # OSGB36 / British National Grid
    wgs84 = pyproj.CRS.from_epsg(4326)     # lat/lon

    # Build transformer
    transformer = pyproj.Transformer.from_crs(tm_proj, wgs84, always_xy=True)

    # Convert grid coordinates to lon/lat
    x = data.projection_x_coordinate
    y = data.projection_y_coordinate

    # If 2D arrays, flatten and then reshape
    lon2d, lat2d = np.meshgrid(x, y)  # shape (y, x)
    lon, lat = transformer.transform(lon2d, lat2d)

    # Cosine weighting
    weights = np.cos(np.deg2rad(lat))
    weights = xr.DataArray(weights, dims=('projection_y_coordinate', 'projection_x_coordinate'))

    # Multiply by weights and normalize
    weighted_mean = (data * weights).sum(dim=['projection_y_coordinate','projection_x_coordinate']) / weights.sum()
    weigthed_mean = weighted_mean.compute()

    return weighted_mean
