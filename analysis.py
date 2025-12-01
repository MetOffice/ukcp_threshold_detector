import numpy as np
import xarray as xr
import glob
import matplotlib.pyplot as plt
from pyproj import CRS, Transformer
import cartopy.crs as ccrs
from analysis_utils import make_color_map, make_spatial_mean
from input_datapaths import inputs

'''
###########################
# Class ThresholdDetector #
###########################

Class ThresholdDetector holds information about the detection to be 
performed and provides methods for computing the threshold crossings
and producing simple plots of the results

Instances of the class are created as follows:
       my_detection = ThresholdDetector(var, threshold, ens=ens, obs=False, method=method)

Inputs:
       var: input variable - can be one of 'tasmax', 'tasmin', 'tas', 'pr'
       threshold: threshold value
       ens (optional): an integer indicating the ensemble member (default = 1)
       obs(optional): True if analysing HadUK-Grid observations (default = False)
       method (optional): threshold crossing method - can be 'above' (default) or 'below' 

Attributes:
       .var: input variable
       .threshold: threshold value
       .ens: ensemble member(s)
       .method: threshold crossing method
       .indata: directory where the input UKCP18 files (netcdf - daily data) are stored
                NOTE 1: the directory should only contain files for one variable and ensemble member
                NOTE 2: if obs = True then this is the directory with the HadUK-Grid data

Methods(details and examples of use are given at the header of each method):
       .detect: creates a DataArray of threshold crossings
       .plot_temporal_mean: plots a map of the mean threshold crossings over a perio
       .plot_spatial_mean: plots the timeseries of annual mean threshold crossings over an area

'''

class ThresholdDetector:

    # Initialise the class
    def __init__(self, var, threshold, ens=1, obs=False, method='above'):

        # Create attributes that hold general information
        self.threshold = threshold
        self.ens = ens
        self.method = method
        if obs and var == 'pr':
            self.var = 'rainfall'
        else:
            self.var = var
        
        # Find the data folder for the requested detection
        # and keep the paths in attribute self.indata
        if obs:
            self.indata = inputs[f'{var}_obs']
        else:    
            if ens < 10:
                self.indata = inputs[f'{var}_0{ens}']
            else:
                self.indata = inputs[f'{var}_{ens}']

        # Check in input variable is correct
        if var not in ['tasmax', 'tasmin', 'tas', 'pr']:
            raise ValueError(f'Invalid variable name: {var}.')

        # Check if input method is correct
        if method not in ['above', 'below']:
            raise ValueError(f'Invalid method name: {method}.')


    '''
    #################
    # Method detect #
    #################

    Method detect creates an xarray DataArray of threshold crossings over 
    ensemble member, time, latitude, and longitude

    ---------
    Examples:
    ---------

    * output_array = my_detection.detect()
    * output_array = my_detection.detect(select_years = [y1, y2, ..., yn])
    * output_array = my_detection.detect(output_file = 'threshold_crossings.nc')

    my_detection: an instance of the class ThresholdDetector

    Inputs:
           self
           select_years (optional): a list of years to analyse. This is useful
                                    for high-res data, where analysing smaller 
                                    segements (rather than all available years)
                                    helps avoid running out of memory
           output_file(optional): name of a netcdf file to save the output   
    Outputs:
           output_array: a DataArray with the threshold exceedances

    '''

    def detect(self, select_years = None, output_file = None):

        print("Processing data in " + self.indata)

        # -------------------------------------------------------------------
        # List files and determine which years each file contains
        # -------------------------------------------------------------------
        files = sorted(glob.glob(f"{self.indata}*.nc"))
        file_years = {}   # mapping: filename → (start_year, end_year)
        for f in files:
            data = xr.open_dataset(f, decode_times=True)
            start_year = data.time.min().dt.year.item()
            end_year   = data.time.max().dt.year.item()
            file_years[f] = (start_year, end_year)
            data.close()

        # Determine the full year range
        all_start_years = [yrs[0] for yrs in file_years.values()]
        all_end_years   = [yrs[1] for yrs in file_years.values()]
        first_year = min(all_start_years)
        last_year  = max(all_end_years)

        # Select years to analyse (default: all available years)
        if select_years is not None:
            if not isinstance(select_years, list):
                raise TypeError("Invalid input: select_years must be a list")
        else:
            select_years = range(first_year, last_year+1)    
        
        # -------------------------------------------------------------------
        # Loop over years, load only the slices needed
        # -------------------------------------------------------------------
        annual_results = []

        for year in select_years:
            print(f"  Processing year: {year}")

            # Identify files that contain this year
            relevant_files = [
                f for f, (y0, y1) in file_years.items()
                if (y0 <= year <= y1) ]

            # Load only the time slices for this year
            parts = []
            for f in relevant_files:
                data = xr.open_dataset(f, decode_times=True)
                data_year = data[self.var].where(data.time.dt.year == year, drop=True)
                parts.append(data_year)
                data.close()

            # Combine the parts (1 or 2 files depending on split years)
            year_data = xr.concat(parts, dim="time")

            # If there are not enough days in this year, then skip it
            if year_data.sizes['time'] < 360:
                continue
            
            # ---------------------------------------------------------------
            # Compute annual exceedances
            # ---------------------------------------------------------------
            if self.method == "above":
                exceed = (year_data > self.threshold).sum(dim="time")
            else:
                exceed = (year_data < self.threshold).sum(dim="time")

            # Mask missing points
            exceed = exceed.where(year_data.notnull().all(dim="time"))

            # Add the year coordinate properly
            exceed = exceed.assign_coords(year=year).expand_dims("year")

            annual_results.append(exceed)

        # -------------------------------------------------------------------
        # Concatenate all years into final DataArray
        # -------------------------------------------------------------------
        var_counts = xr.concat(annual_results, dim="year")

        # Keep original attributes (use the attributes from the last loaded year data)
        var_counts.attrs.update(year_data.attrs)

        # Save to netCDF if requested
        if output_file is not None:
            var_counts.to_netcdf(output_file)

        return var_counts


    '''
    #############################
    # Method plot_temporal_mean #
    #############################

    Method plot_temporal_mean plots a map of the mean threshold crossings
    over a period starting in y1 and ending in y2

    ---------
    Examples:
    ---------

    * my_detection.plot_temporal_mean(var_counts, y1, y2)
    * my_detection.plot_temporal_mean(var_counts, y1, y2, output_file = 'plot.png')

    my_detection: an instance of the class ThresholdDetector

    Inputs:
           self
           var_counts: DataArray with threshold crossings (created by method detect)
           y1, y2: the first and last years of the selected period
           output_file(optional): name of a png file to save the plot

    '''

    def plot_temporal_mean(self, var_counts, y1, y2, output_file = None):

        # Check if period is correctly specified
        if y1 < var_counts.year.min() or y2 > var_counts.year.max():
            raise ValueError('Selected year(s) outside expected range')
        if y1 > y2:
            raise ValueError('Incorrect period: y2 must be >= y1')

        # Select period and compute temporal mean
        varmean = var_counts.sel(year = slice(str(y1), str(y2))).mean(dim='year')

        # If there is an 'ensemble_member' coord, then select it
        if 'ensemble_member' in varmean.coords:
            varmean = varmean['ensemble_member' == self.ens]

        # Plot
        plt.ion()
        mycmap = make_color_map(361)
        fig = plt.figure()
        ax = plt.subplot2grid((1,1), (0,0), projection=ccrs.epsg(27700))
        if "projection_x_coordinate" in var_counts.coords:
            plt.pcolormesh(varmean.projection_x_coordinate,
                           varmean.projection_y_coordinate,
                           varmean.data, cmap=mycmap,
                           vmin=0, vmax=varmean.quantile(0.98))
        else:
            pole_lat = 37.5
            pole_lon = 177.5
            rpole = ccrs.RotatedPole(pole_longitude=float(pole_lon),
                                     pole_latitude=float(pole_lat))
            plt.pcolormesh(varmean.grid_longitude,
                        varmean.grid_latitude,
                        varmean.data, transform=rpole, cmap=mycmap,
                        vmin=0, vmax=varmean.quantile(0.98))
        ax.coastlines()
        ax.set_title(f'{y1} - {y2} / Threshold = {self.threshold}')
        plt.colorbar(orientation='horizontal', label='Threshold Crossings / Year', fraction=0.03, pad=0.03)

        # Save plot if output_file is given 
        if output_file is not None:
            plt.savefig(output_file)


    '''
    #############################
    # Method plot_spatial_mean #
    #############################

    Method plot_spatial_mean plots the timeseries of annual mean 
    threshold crossings over an area

    ---------
    Examples:
    ---------

    * my_detection.plot_spatial_mean(var_counts)
    * my_detection.plot_spatial_mean(var_counts, output_file = 'plot.png')
    * my_detection.plot_spatial_mean(var_counts, mylon = [lon1, lon2], mylat = [lat1, lat2])

    my_detection: an instance of the class ThresholdDetector

    Inputs:
           self
           var_counts: DataArray with threshold crossings (created by method detect)
           mylon, mylat (optional): 2-dimensional lists with the coordinates of an area to extract.
                                    If not given, the mean is computed over the entire area
           output_file(optional): name of a png file to save the plot

    '''

    def plot_spatial_mean(self, var_counts, mylon = None, mylat = None, output_file = None):

        # Extract area, if needed
        if (mylon is not None) and (mylat is not None):
            if "projection_x_coordinate" in var_counts.coords:
                transformer = Transformer.from_crs('EPSG:4326', 'EPSG:27700', always_xy=True)
                x, y = transformer.transform(mylon, mylat)
                var_counts_myarea = var_counts.sel(projection_x_coordinate = slice(x[0], x[1]),
                                                   projection_y_coordinate = slice(y[0], y[1])) 
            else:
                # Rotated pole lon/lat
                pole_lat = 37.5
                pole_lon = 177.5
                rot = ccrs.RotatedPole(pole_latitude=pole_lat, pole_longitude=pole_lon)
                geo = ccrs.PlateCarree()
                rot_crs = CRS.from_wkt(rot.to_wkt())
                geo_crs = CRS.from_wkt(geo.to_wkt())
                transformer = Transformer.from_crs(geo_crs, rot_crs, always_xy=True)               
                # Convert lon/lat → rotated coords
                x0, y0 = transformer.transform(mylon[0], mylat[0])
                x1, y1 = transformer.transform(mylon[1], mylat[1])
                # Align to dataset coordinate space
                grid_min = float(var_counts.grid_longitude.min())
                grid_max = float(var_counts.grid_longitude.max())
                def align_rotlon(x, gmin=grid_min):
                    while x < gmin:
                        x += 360
                        return x
                x0 = align_rotlon(x0)
                x1 = align_rotlon(x1)
                # Handle seam crossing
                if x0 <= x1:
                    var_counts_myarea = var_counts.sel(grid_longitude=slice(x0, x1),
                                                       grid_latitude=slice(y0, y1))
                else:
                # x0 > x1 → crossing the 360→0 seam
                    part1 = var_counts.sel(grid_longitude=slice(x0, grid_max),
                                           grid_latitude=slice(y0, y1))
                    part2 = var_counts.sel(grid_longitude=slice(grid_min, x1),
                                           grid_latitude=slice(y0, y1))
                    var_counts_myarea = xr.concat([part1, part2], dim="grid_longitude")
        else:
            var_counts_myarea = var_counts.copy()
            
        # Compute the spatial mean
        if 'ensemble_member' in var_counts_myarea.dims:
            varmean = make_spatial_mean(var_counts_myarea.sel(ensemble_member = self.ens))
        else:
            varmean = make_spatial_mean(var_counts_myarea)
            
        # Plot
        plt.ion()
        fig = plt.figure()
        ax = plt.subplot2grid((1,1), (0,0))
        ax.plot(varmean.year, varmean, color='black')
        ax.set_title(f'Spatial Mean Timeseries / Var: {self.var} / Threshold = {self.threshold}')
        ax.set_xlabel('Year')
        ax.set_ylabel('No of Crossings Per Year')

        # Save plot if output_file is given 
        if output_file is not None:
            plt.savefig(output_file)
