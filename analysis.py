import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pyproj import Transformer
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
       my_detection = ThresholdDetector(var, threshold, ens=ens, method=method)

Inputs:
       var: input variable - can be one of 'tasmax', 'tasmin', 'tas', 'pr'
       threshold: threshold value
       ens (optional): an integer indicating the ensemble member (default = 1)
       method (optional): threshold crossing method - can be 'above' (default) or 'below' 

Attributes:
       .var: input variable
       .threshold: threshold value
       .ens: ensemble member
       .method: threshold crossing method
       .indata: directory where the input UKCP18 files (netcdf - daily data) are stored
                NOTE: the directory should only contain files for one variable and ensemble member

Methods(details and examples of use are given at the header of each method):
       .detect: creates a DataArray of threshold crossings
       .plot_temporal_mean: plots a map of the mean threshold crossings over a perio
       .plot_spatial_mean: plots the timeseries of annual mean threshold crossings over an area

'''

class ThresholdDetector:

    # Initialise the class
    def __init__(self, var, threshold, ens=1, method='above'):

        # Create attributes that hold general information
        self.var = var
        self.threshold = threshold
        self.ens = ens
        self.method = method

        # Find the data folder for the requested detection
        if ens < 10:
            self.indata = inputs[f'{var}_0{ens}']
        else:
            self.indata = inputs[f'{var}_{ens}']

        # Check in input variable is correct
        if var not in ['tasmax', 'tasmin', 'tas', 'pr']:
            raise ValueError(f'Invalid variable name: {var}.')

        # Check if input ensemble member variable is an integer
        if not isinstance(ens, int):
            raise ValueError(f'Ensemble member {ens} is not an integer')

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
    * output_array = my_detection.detect(output_file = 'threshold_crossings.nc')

    my_detection: an instance of the class ThresholdDetector

    Inputs:
           self
           output_file(optional): name of a netcdf file to save the output   
    Outputs:
           output_array: a DataArray with the threshold exceedances

    '''

    def detect(self, output_file = None):

        # Read input files
        data = xr.open_mfdataset(f'{self.indata}*.nc', combine="by_coords")

        # Omit first and last years that don't include all the days
        y1 = data.time.min().dt.year.item(0)+1
        y2 = data.time.max().dt.year.item(0)-1
        data = data.sel(time = slice(str(y1), str(y2)))

        # Compute threshold crossings in each year
        if self.method == 'above':
            var_select = data[self.var] > self.threshold
        else:
            var_select = data[self.var] < self.threshold
        var_counts = var_select.groupby('time.year').sum(dim='time')

        # Save to netcdf if output_file is given 
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
           var_counts: DataArray with threshold crossings (created by method detections)
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
        varmean = var_counts.sel(year = slice(str(y1), str(y2))).mean(dim='year').load()

        # Plot
        plt.ion()
        mycmap = make_color_map(361)
        fig = plt.figure()
        ax = plt.subplot2grid((1,1), (0,0), projection=ccrs.epsg(27700))
        plt.pcolormesh(varmean.projection_x_coordinate,
                       varmean.projection_y_coordinate,
                       varmean.data[0,:,:], cmap=mycmap,
                       vmin=0, vmax=varmean.max())
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
           var_counts: DataArray with threshold crossings (created by method detections)
           mylon, mylat (optional): 2-dimensional lists with the coordinates of an area to extract.
                                    If not given, the mean is computed over the entire area
           output_file(optional): name of a png file to save the plot

    '''

    def plot_spatial_mean(self, var_counts, mylon = None, mylat = None, output_file = None):

        # Extract area, if needed
        if (mylon is not None) and (mylat is not None):
            transformer = Transformer.from_crs('EPSG:4326', 'EPSG:27700', always_xy=True)
            x, y = transformer.transform(mylon, mylat)
            var_counts_myarea = var_counts.sel(projection_x_coordinate = slice(x[0], x[1]),
                                               projection_y_coordinate = slice(y[0], y[1])) 
        else:
            var_counts_myarea = var_counts.copy()

        # Compute the spatial mean
        varmean = make_spatial_mean(var_counts_myarea[0, :])

        # Plot
        plt.ion()
        fig = plt.figure()
        ax = plt.subplot2grid((1,1), (0,0))
        ax.plot(varmean.year, varmean)
        ax.set_title(f'Spatial Mean Timeseries / Var: {self.var} / Threshold = {self.threshold}')
        ax.set_xlabel('Year')
        ax.set_ylabel('No of Crossings Per Year')

        # Save plot if output_file is given 
        if output_file is not None:
            plt.savefig(output_file)
