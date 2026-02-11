import numpy as np
import xarray as xr
import glob
import matplotlib.pyplot as plt
from pyproj import CRS, Transformer
import cartopy.crs as ccrs
from analysis_utils import make_color_map, make_spatial_mean, cumulative_runlength
from input_datapaths import inputs

'''
####################################
# Class ThresholdDetectorCoumpound #
####################################

This is a version of the ThresholdDetector developed to examine compound
events with two variables crossing simultaneously (on the same day) their
specified thresholds.

Class ThresholdDetectorCoumpound holds information about the detection 
to be performed and provides methods for computing the threshold 
crossings and producing simple plots of the results.

Instances of the class are created as follows:
       my_detection = ThresholdDetectorCompund([var1, var2], 
                                               [threshold1, threshold2], 
                                               [method1, method2], 
                                               ens=ens, obs=False)

Inputs:
       var: input variables - a list of the two variables that define the compound
                              event. Acceptable variables are:
                              'tasmax', 'tasmin', 'tas', 'pr', 'uas',
                              'vas', 'sfcWind', 'hurs', 'huss', 'prsn'
       threshold: a list of the two threshold values
       method : a list of the two threshold crossing method - can be 'above' or 'below' 
       ens (optional): an integer indicating the ensemble member (default = 1)
       obs(optional): True if analysing HadUK-Grid observations (default = False)

Attributes:
       .var: input variables
       .threshold: threshold values
       .ens: ensemble member
       .method: threshold crossing methods
       .indata: directories where the input UKCP18 files (netcdf - daily data) are stored
                One directory per variable (indata is a list of the two directories).
                NOTE 1: each directory should only contain files for one variable and ensemble member
                NOTE 2: if obs = True then the input directories have the HadUK-Grid data

Methods(details and examples of use are given at the header of each method):
       .detect: creates a DataArray of threshold crossings
       .detect_maxlength: creates a DataArray of the length of the longest spell
       .detect_spells: creates a DataArray of threshold-crossing spells
       .plot_temporal_mean: plots a map of the mean threshold crossings over a perio
       .plot_spatial_mean: plots the timeseries of annual mean threshold crossings over an area

'''

class ThresholdDetectorCompound:

    # Initialise the class
    def __init__(self, var, threshold, method, ens=1, obs=False):

        # Check all mandatory inputs are lists of length 2
        if not all(isinstance(x, list) and len(x) == 2 for x in [var, threshold, method]):
            raise ValueError("Inputs var, threshold and method must all be lists of length 2")

        # Check in input variable is correct
        if any(v not in ['tasmax', 'tasmin', 'tas', 'pr', 'uas', 'vas', 'sfcWind',
                         'hurs', 'huss', 'prsn'] for v in var):
            raise ValueError(f'Invalid name for some input variable(s): {var}.')
        
        # Check if input method is correct
        if any(m not in ['above', 'below'] for m in method):
            raise ValueError(f'Invalid name for some input method(s): {method}.')
       
        # Create attributes that hold general information
        self.threshold = threshold
        self.method = method
        self.ens = ens
        if obs and var == 'pr':
            self.var = ['rainfall' if v == 'pr' else v for v in var]
        else:
            self.var = var

        # Find the data folder for the requested detection
        # and keep the paths in attribute self.indata
        if obs:
            self.indata = [inputs[f'{var[0]}_obs'], inputs[f'{var[1]}_obs']]
        else:    
            if ens < 10:
                self.indata = [inputs[f'{var[0]}_0{ens}'], inputs[f'{var[1]}_0{ens}']]
            else:
                self.indata = [inputs[f'{var[0]}_{ens}'], inputs[f'{var[1]}_{ens}']]


    '''
    #################
    # Method detect #
    #################

    Method detect creates an xarray DataArray of threshold crossings 
    over time, latitude, and longitude

    ---------
    Examples:
    ---------

    * output_array = my_detection.detect()
    * output_array = my_detection.detect(select_years = [y1, y2, ..., yn])
    * output_array = my_detection.detect(select_months = [m1, m2, ..., mn])
    * output_array = my_detection.detect(year_from_dec = True)
    * output_array = my_detection.detect(output_file = 'output_file.nc')

    my_detection: an instance of the class ThresholdDetector

    Inputs:
           self
           select_years (optional): a list of years to analyse. This is useful
                                    for high-res data, where analysing smaller 
                                    segements (rather than all available years)
                                    helps avoid running out of memory
           select_months (optional): a list of months to analyse. If None 
                                    (default) then annual exceedances are computed.
                                    If a list of months is provided, then the
                                    exceedances are computed only for the selected
                                    months. This is a useful option if, for example,
                                    users want to calculate exceedances in a season
           years_from_dec (optional): if True, then years run from Dec to Nov instead
                                      of the default (Jan to Dec)
           output_file(optional): name of a netcdf file to save the output   
    Outputs:
           output_array: a DataArray with the threshold exceedances

    '''

    def detect(self, select_years = None, select_months = None,
               years_from_dec = False, output_file = None):

        # -------------------------------------------------------------------
        # List files and determine which years each file contains
        # -------------------------------------------------------------------
        # VARIABLE 1 
        print("Processing data in " + self.indata[0])
        files = sorted(glob.glob(f"{self.indata[0]}*.nc"))
        file_years1 = {}   # mapping: filename → (start_year, end_year)
        for f in files:
            data = xr.open_dataset(f, decode_times=True)
            start_year = data.time.min().dt.year.item()
            end_year   = data.time.max().dt.year.item()
            file_years1[f] = (start_year, end_year)
            data.close()

        # Determine the full year range
        all_start_years = [yrs[0] for yrs in file_years1.values()]
        all_end_years   = [yrs[1] for yrs in file_years1.values()]
        first_year = min(all_start_years)
        last_year  = max(all_end_years)

        # VARIABLE 2
        print("Processing data in " + self.indata[1])
        files = sorted(glob.glob(f"{self.indata[1]}*.nc"))
        file_years2 = {}   # mapping: filename → (start_year, end_year)
        for f in files:
            data = xr.open_dataset(f, decode_times=True)
            start_year = data.time.min().dt.year.item()
            end_year   = data.time.max().dt.year.item()
            file_years2[f] = (start_year, end_year)
            data.close()

        # Determine the full year range
        all_start_years = [yrs[0] for yrs in file_years2.values()]
        all_end_years   = [yrs[1] for yrs in file_years2.values()]
        first_year = max([first_year, min(all_start_years)])
        last_year = min([last_year, max(all_end_years)])
        
        # Select years to analyse (default: all available years)
        if select_years is not None:
            if not isinstance(select_years, list):
                raise TypeError("Invalid input: select_years must be a list")
            if years_from_dec:
                # Start from December of the previous year
                select_years = [iyr-1 for iyr in select_years]
        else:
            if years_from_dec:
                select_years = range(first_year, last_year)
            else:
                select_years = range(first_year, last_year+1)
        
        # -------------------------------------------------------------------
        # Loop over years, load only the slices needed
        # -------------------------------------------------------------------
        annual_results = []

        for year in select_years:
            print(f"  Processing year: {year}")

            # Identify files that contain this year
            if years_from_dec:
                relevant_files1 = [
                    f for f, (y0, y1) in file_years1.items()
                    if (y0 <= year <= y1) or (y0 <= year+1 <=y1)]
                relevant_files2 = [
                    f for f, (y0, y1) in file_years2.items()
                    if (y0 <= year <= y1) or (y0 <= year+1 <=y1)]
            else:
                relevant_files1 = [
                    f for f, (y0, y1) in file_years1.items()
                    if (y0 <= year <= y1) ]
                relevant_files2 = [
                    f for f, (y0, y1) in file_years2.items()
                    if (y0 <= year <= y1) ]

            # Load only the time slices for this year - Variable 1
            parts = []
            for f in relevant_files1:
                data = xr.open_dataset(f, decode_times=True)
                if years_from_dec:
                    data_year = data[self.var[0]].sel(time = 
                        ((data.time.dt.year == year) & (data.time.dt.month == 12)) |
                        ((data.time.dt.year == year+1) & (data.time.dt.month < 12)))
                else:
                    data_year = data[self.var[0]].where(data.time.dt.year == year, drop=True)
                if data_year.sizes['time'] > 0:
                    parts.append(data_year)
                data.close()

            # Combine the parts - Variable 1
            year_data1 = xr.concat(parts, dim="time")

            # Load only the time slices for this year - Variable 2
            parts = []
            for f in relevant_files2:
                data = xr.open_dataset(f, decode_times=True)
                if years_from_dec:
                    data_year = data[self.var[1]].sel(time = 
                        ((data.time.dt.year == year) & (data.time.dt.month == 12)) |
                        ((data.time.dt.year == year+1) & (data.time.dt.month < 12)))
                else:
                    data_year = data[self.var[1]].where(data.time.dt.year == year, drop=True)
                if data_year.sizes['time'] > 0:
                    parts.append(data_year)
                data.close()

            # Combine the parts - Variable 2
            year_data2 = xr.concat(parts, dim="time")

            # If there are not enough days in this year, then skip it
            if year_data1.sizes['time'] < 360 or year_data2.sizes['time'] < 360:
                continue

            # If only some months are required (e.g. a season) then extract them
            if select_months is not None:
                year_data1 = year_data1.sel(time=(year_data1.time.dt.month.isin(select_months)))
                year_data2 = year_data2.sel(time=(year_data2.time.dt.month.isin(select_months)))

            # ---------------------------------------------------------------
            # Compute annual exceedances
            # ---------------------------------------------------------------
            if self.method == ['above', 'above']:
                exceed = ((year_data1 > self.threshold[0]) &
                          (year_data2 > self.threshold[1])).sum(dim="time")
            if self.method == ['above', 'below']:
                exceed = ((year_data1 > self.threshold[0]) &
                          (year_data2 < self.threshold[1])).sum(dim="time")
            if self.method == ['below', 'above']:
                exceed = ((year_data1 < self.threshold[0]) &
                          (year_data2 > self.threshold[1])).sum(dim="time")
            if self.method == ['below', 'below']:
                exceed = ((year_data1 < self.threshold[0]) &
                          (year_data2 < self.threshold[1])).sum(dim="time")

            # Mask missing points
            exceed = exceed.where(year_data1.notnull().all(dim="time"))
            exceed = exceed.where(year_data2.notnull().all(dim="time"))

            # Add the year coordinate properly
            if years_from_dec:
                exceed = exceed.assign_coords(year=year+1).expand_dims("year")
            else:
                exceed = exceed.assign_coords(year=year).expand_dims("year")

            annual_results.append(exceed)
                
        # -------------------------------------------------------------------
        # Concatenate all years into final DataArray
        # -------------------------------------------------------------------
        var_counts = xr.concat(annual_results, dim="year")
        var_counts.name = 'threshold_crossings'
        # Copy original attributes and update them
        var_counts.attrs.update(year_data1.attrs)
        var_counts.attrs.update({
            "standard_name": "theshold_crossings",
            "long_name": "Threshold Detector Output",
            "units": "number of days per year",
            "description": "Threshold Detector Output",
            "label_units": "number of days per year",
            "plot_label": "Theshold Crossings" })

        # Save to netCDF if requested
        if output_file is not None:
            var_counts.to_netcdf(output_file)

        return var_counts


    '''
    ###########################
    # Method detect_maxlength #
    ###########################

    Method detect_maxlength creates an xarray DataArray of the maximum spell length
    over time, latitude, and longitude. Spells are defined as consecutive days when
    the thresholds of the two variables are crossed simultaneoulsy. The method computes
    the number of days associated with the longest spell of the year.

    ---------
    Examples:
    ---------

    * output_array = my_detection.detect_maxlength()
    * output_array = my_detection.detect_maxlength(select_years = [y1, y2, ..., yn])
    * output_array = my_detection.detect_maxlength(select_months = [m1, m2, ..., mn])
    * output_array = my_detection.detect_maxlength(year_from_dec = True)
    * output_array = my_detection.detect_maxlength(output_file = 'output_file.nc')

    my_detection: an instance of the class ThresholdDetector

    Inputs:
           self
           select_years (optional): a list of years to analyse. This is useful
                                    for high-res data, where analysing smaller 
                                    segements (rather than all available years)
                                    helps avoid running out of memory
           select_months (optional): a list of months to analyse. If None 
                                    (default) then annual exceedances are computed.
                                    If a list of months is provided, then the
                                    exceedances are computed only for the selected
                                    months. This is a useful option if, for example,
                                    users want to calculate exceedances in a season
           years_from_dec (optional): if True, then years run from Dec to Nov instead
                                      of the default (Jan to Dec)
           output_file(optional): name of a netcdf file to save the output   
    Outputs:
           output_array: a DataArray with the threshold exceedances

    '''

    def detect_maxlength(self, select_years = None, select_months = None,
                         years_from_dec = False, output_file = None):

        # -------------------------------------------------------------------
        # List files and determine which years each file contains
        # -------------------------------------------------------------------
        # VARIABLE 1 
        print("Processing data in " + self.indata[0])
        files = sorted(glob.glob(f"{self.indata[0]}*.nc"))
        file_years1 = {}   # mapping: filename → (start_year, end_year)
        for f in files:
            data = xr.open_dataset(f, decode_times=True)
            start_year = data.time.min().dt.year.item()
            end_year   = data.time.max().dt.year.item()
            file_years1[f] = (start_year, end_year)
            data.close()

        # Determine the full year range
        all_start_years = [yrs[0] for yrs in file_years1.values()]
        all_end_years   = [yrs[1] for yrs in file_years1.values()]
        first_year = min(all_start_years)
        last_year  = max(all_end_years)

        # VARIABLE 2
        print("Processing data in " + self.indata[1])
        files = sorted(glob.glob(f"{self.indata[1]}*.nc"))
        file_years2 = {}   # mapping: filename → (start_year, end_year)
        for f in files:
            data = xr.open_dataset(f, decode_times=True)
            start_year = data.time.min().dt.year.item()
            end_year   = data.time.max().dt.year.item()
            file_years2[f] = (start_year, end_year)
            data.close()

        # Determine the full year range
        all_start_years = [yrs[0] for yrs in file_years2.values()]
        all_end_years   = [yrs[1] for yrs in file_years2.values()]
        first_year = max([first_year, min(all_start_years)])
        last_year = min([last_year, max(all_end_years)])
        
        # Select years to analyse (default: all available years)
        if select_years is not None:
            if not isinstance(select_years, list):
                raise TypeError("Invalid input: select_years must be a list")
            if years_from_dec:
                # Start from December of the previous year
                select_years = [iyr-1 for iyr in select_years]
        else:
            if years_from_dec:
                select_years = range(first_year, last_year)
            else:
                select_years = range(first_year, last_year+1)

        # -------------------------------------------------------------------
        # Loop over years, load only the slices needed
        # -------------------------------------------------------------------
        annual_results = []

        for year in select_years:
            print(f"  Processing year: {year}")

            # Identify files that contain this year
            if years_from_dec:
                relevant_files1 = [
                    f for f, (y0, y1) in file_years1.items()
                    if (y0 <= year <= y1) or (y0 <= year+1 <=y1)]
                relevant_files2 = [
                    f for f, (y0, y1) in file_years2.items()
                    if (y0 <= year <= y1) or (y0 <= year+1 <=y1)]              
            else:
                relevant_files1 = [
                    f for f, (y0, y1) in file_years1.items()
                    if (y0 <= year <= y1) ]
                relevant_files2 = [
                    f for f, (y0, y1) in file_years2.items()
                    if (y0 <= year <= y1) ]

            # Load only the time slices for this year - Variable 1
            parts = []
            for f in relevant_files1:
                data = xr.open_dataset(f, decode_times=True)
                if years_from_dec:
                    data_year = data[self.var[0]].sel(time = 
                        ((data.time.dt.year == year) & (data.time.dt.month == 12)) |
                        ((data.time.dt.year == year+1) & (data.time.dt.month < 12)))
                else:
                    data_year = data[self.var[0]].where(data.time.dt.year == year, drop=True)
                if data_year.sizes['time'] > 0:
                    parts.append(data_year)
                data.close()

            # Combine the parts - Variable 1
            year_data1 = xr.concat(parts, dim="time")

            # Load only the time slices for this year - Variable 2
            parts = []
            for f in relevant_files2:
                data = xr.open_dataset(f, decode_times=True)
                if years_from_dec:
                    data_year = data[self.var[1]].sel(time = 
                        ((data.time.dt.year == year) & (data.time.dt.month == 12)) |
                        ((data.time.dt.year == year+1) & (data.time.dt.month < 12)))
                else:
                    data_year = data[self.var[1]].where(data.time.dt.year == year, drop=True)
                if data_year.sizes['time'] > 0:
                    parts.append(data_year)
                data.close()

            # Combine the parts - Variable 2
            year_data2 = xr.concat(parts, dim="time")

             # If there are not enough days in this year, then skip it
            if year_data1.sizes['time'] < 360 or year_data2.sizes['time'] < 360:
                continue

            # If only some months are required (e.g. a season) then extract them
            if select_months is not None:
                year_data1 = year_data1.sel(time=(year_data1.time.dt.month.isin(select_months)))
                year_data2 = year_data2.sel(time=(year_data2.time.dt.month.isin(select_months)))

            # ---------------------------------------------------------------
            # Compute max length spells
            # ---------------------------------------------------------------
            if self.method == ['above', 'above']:
                crossing = (year_data1 > self.threshold[0]) & (year_data2 > self.threshold[1])
            if self.method == ['above', 'below']:
                crossing = (year_data1 > self.threshold[0]) & (year_data2 < self.threshold[1])
            if self.method == ['below', 'above']:
                crossing = (year_data1 < self.threshold[0]) & (year_data2 > self.threshold[1])
            if self.method == ['below', 'below']:
                crossing = (year_data1 < self.threshold[0]) & (year_data2 < self.threshold[1])

            # Count consecutive True values within each spell
            spell_length = crossing.cumsum(dim="time") - \
            crossing.cumsum(dim="time").where(~crossing).ffill(dim="time").fillna(0)

            #  Maximum spell length per grid point
            max_spell_length = spell_length.max(dim="time")

            # Mask missing points
            max_spell_length = max_spell_length.where(year_data1.notnull().all(dim="time"))
            max_spell_length = max_spell_length.where(year_data2.notnull().all(dim="time"))

            # Add the year coordinate properly
            if years_from_dec:
                max_spell_length = max_spell_length.assign_coords(year=year+1).expand_dims("year")
            else:
                max_spell_length = max_spell_length.assign_coords(year=year).expand_dims("year")

            annual_results.append(max_spell_length)
                
        # -------------------------------------------------------------------
        # Concatenate all years into final DataArray
        # -------------------------------------------------------------------
        var_counts = xr.concat(annual_results, dim="year")
        var_counts.name = 'max_spell_length'
        # Copy original attributes and update them
        var_counts.attrs.update(year_data1.attrs)
        var_counts.attrs.update({
            "standard_name": "max_spell_length",
            "long_name": "Threshold Detector Output",
            "units": "days",
            "description": "Threshold Detector Output",
            "label_units": "days",
            "plot_label": "Max Spell Length" })

        # Save to netCDF if requested
        if output_file is not None:
            var_counts.to_netcdf(output_file)

        return var_counts


    '''
    ########################
    # Method detect_spells #
    ########################

    Method detect_spells creates an xarray DataArray of threshold-crossing spells
    over time, latitude, and longitude. Spells are defined as consecutive days when
    the thresholds of the two variables are crossed simultaneoulsy. Spells of a 
    minimumn length (min_length) may be specified. Also, spells may be considered 
    separate only if there are at least X no-crossing days (decluster_days) 
    between them. Method detect_spells counts how many such events occur in the year.

    ---------
    Examples:
    ---------

    * output_array = my_detection.detect_spells()                   # count all spells of any length
    * output_array = my_detection.detect_spells(min_length = 5)     # only count spells >= 5 days
    * output_array = my_detection.detect_spells(decluster_days = 3) # spells separated by at least 3 days
    * output_array = my_detection.detect_spells(select_years = [y1, y2, ..., yn])
    * output_array = my_detection.detect_spells(select_months = [m1, m2, ..., mn])
    * output_array = my_detection.detect_spells(years_from_dec = True)
    * output_array = my_detection.detect_spells(output_file = 'output_file.nc')

    my_detection: an instance of the class ThresholdDetector

    Inputs:
           self
           min_length (optional) : if specified, only spells with at least min_length
                                   days are counted 
           decluster_days(optional): minimum number of days without a threshold 
                                     crossing required between spells. If set 
                                     to 1 (default), all spells are counted, even
                                     if only separated by 1 day
           select_years (optional): a list of years to analyse. This is useful
                                    for high-res data, where analysing smaller 
                                    segements (rather than all available years)
                                    helps avoid running out of memory
           select_months (optional): a list of months to analyse. If None 
                                    (default) then annual exceedances are computed.
                                    If a list of months is provided, then the
                                    exceedances are computed only for the selected
                                    months. This is a useful option if, for example,
                                    users want to calculate exceedances in a season
           years_from_dec (optional): if True, then years run from Dec to Nov instead
                                      of the default (Jan to Dec)
           output_file(optional): name of a netcdf file to save the output   
    Outputs:
           output_array: a DataArray with the threshold exceedances

    '''

    def detect_spells(self, min_length = False, decluster_days = 1, select_years = None,
                      select_months = None, years_from_dec = False, output_file = None):

        # Check if decluster_days is correct (must be >= 1)
        if decluster_days < 1:
            raise ValueError(f'Invalid value for decluster_days: {decluster_days}. Must be >=1')
        
        # -------------------------------------------------------------------
        # List files and determine which years each file contains
        # -------------------------------------------------------------------
        # VARIABLE 1 
        print("Processing data in " + self.indata[0])
        files = sorted(glob.glob(f"{self.indata[0]}*.nc"))
        file_years1 = {}   # mapping: filename → (start_year, end_year)
        for f in files:
            data = xr.open_dataset(f, decode_times=True)
            start_year = data.time.min().dt.year.item()
            end_year   = data.time.max().dt.year.item()
            file_years1[f] = (start_year, end_year)
            data.close()

        # Determine the full year range
        all_start_years = [yrs[0] for yrs in file_years1.values()]
        all_end_years   = [yrs[1] for yrs in file_years1.values()]
        first_year = min(all_start_years)
        last_year  = max(all_end_years)

        # VARIABLE 2
        print("Processing data in " + self.indata[1])
        files = sorted(glob.glob(f"{self.indata[1]}*.nc"))
        file_years2 = {}   # mapping: filename → (start_year, end_year)
        for f in files:
            data = xr.open_dataset(f, decode_times=True)
            start_year = data.time.min().dt.year.item()
            end_year   = data.time.max().dt.year.item()
            file_years2[f] = (start_year, end_year)
            data.close()

        # Determine the full year range
        all_start_years = [yrs[0] for yrs in file_years2.values()]
        all_end_years   = [yrs[1] for yrs in file_years2.values()]
        first_year = max([first_year, min(all_start_years)])
        last_year = min([last_year, max(all_end_years)])
        
        # Select years to analyse (default: all available years)
        if select_years is not None:
            if not isinstance(select_years, list):
                raise TypeError("Invalid input: select_years must be a list")
            if years_from_dec:
                # Start from December of the previous year
                select_years = [iyr-1 for iyr in select_years]
        else:
            if years_from_dec:
                select_years = range(first_year, last_year)
            else:
                select_years = range(first_year, last_year+1)
        
        # -------------------------------------------------------------------
        # Loop over years, load only the slices needed
        # -------------------------------------------------------------------
        annual_results = []

        for year in select_years:
            print(f"  Processing year: {year}")

            # Identify files that contain this year
            if years_from_dec:
                relevant_files1 = [
                    f for f, (y0, y1) in file_years1.items()
                    if (y0 <= year <= y1) or (y0 <= year+1 <=y1)]
                relevant_files2 = [
                    f for f, (y0, y1) in file_years2.items()
                    if (y0 <= year <= y1) or (y0 <= year+1 <=y1)]
            else:
                relevant_files1 = [
                    f for f, (y0, y1) in file_years1.items()
                    if (y0 <= year <= y1) ]
                relevant_files2 = [
                    f for f, (y0, y1) in file_years2.items()
                    if (y0 <= year <= y1) ]

            # Load only the time slices for this year - Variable 1
            parts = []
            for f in relevant_files1:
                data = xr.open_dataset(f, decode_times=True)
                if years_from_dec:
                    data_year = data[self.var[0]].sel(time = 
                        ((data.time.dt.year == year) & (data.time.dt.month == 12)) |
                        ((data.time.dt.year == year+1) & (data.time.dt.month < 12)))
                else:
                    data_year = data[self.var[0]].where(data.time.dt.year == year, drop=True)
                if data_year.sizes['time'] > 0:
                    parts.append(data_year)
                data.close()

            # Combine the parts - Variable 1
            year_data1 = xr.concat(parts, dim="time")

            # Load only the time slices for this year - Variable 2
            parts = []
            for f in relevant_files2:
                data = xr.open_dataset(f, decode_times=True)
                if years_from_dec:
                    data_year = data[self.var[1]].sel(time = 
                        ((data.time.dt.year == year) & (data.time.dt.month == 12)) |
                        ((data.time.dt.year == year+1) & (data.time.dt.month < 12)))
                else:
                    data_year = data[self.var[1]].where(data.time.dt.year == year, drop=True)
                if data_year.sizes['time'] > 0:
                    parts.append(data_year)
                data.close()

            # Combine the parts - Variable 2
            year_data2 = xr.concat(parts, dim="time")

            # If there are not enough days in this year, then skip it
            if year_data1.sizes['time'] < 360 or year_data2.sizes['time'] < 360:
                continue

            # If only some months are required (e.g. a season) then extract them
            if select_months is not None:
                year_data1 = year_data1.sel(time=(year_data1.time.dt.month.isin(select_months)))
                year_data2 = year_data2.sel(time=(year_data2.time.dt.month.isin(select_months)))

            # ---------------------------------------------------------------
            # Compute number of spells (separated by at least dectuster_days)
            # ---------------------------------------------------------------

            if self.method == ['above', 'above']:
                crossing = (year_data1 > self.threshold[0]) & (year_data2 > self.threshold[1])
            if self.method == ['above', 'below']:
                crossing = (year_data1 > self.threshold[0]) & (year_data2 < self.threshold[1])
            if self.method == ['below', 'above']:
                crossing = (year_data1 < self.threshold[0]) & (year_data2 > self.threshold[1])
            if self.method == ['below', 'below']:
                crossing = (year_data1 < self.threshold[0]) & (year_data2 < self.threshold[1])

            # Apply minumum spell length, if required
            if min_length:
                crossing_int = crossing.astype(int)
                # Apply run-length calculation along time dimension, preserving all other dims,
                # using helper function cumulative_runlength
                runlen = xr.apply_ufunc(
                    cumulative_runlength,
                    crossing_int,
                    input_core_dims=[['time']],
                    output_core_dims=[['time']],
                    vectorize=True,
                    dask='parallelized',
                    output_dtypes=[crossing_int.dtype])
                # Mask events shorter than min_length
                crossing = crossing & (runlen >= min_length)

            # Identify event starts (first day of each spell)
            event_start = crossing & ~crossing.shift(time=1, fill_value=False)

            # Compute gap since last event
            gap = (~crossing).cumsum(dim='time') - \
                (~crossing).cumsum(dim='time').where(crossing).ffill(dim='time').fillna(0)

            # Look at the gap before the event starts
            gap_before = gap.shift(time=1, fill_value=decluster_days)

            # Keep event starts only if the quiet gap is long enough
            new_event = event_start & (gap_before >= decluster_days)

            # Count clustered events
            exceed = new_event.sum(dim="time")

            # Mask missing points
            exceed = exceed.where(year_data1.notnull().all(dim="time"))
            exceed = exceed.where(year_data2.notnull().all(dim="time"))

            # Add the year coordinate properly
            if years_from_dec:
                exceed = exceed.assign_coords(year=year+1).expand_dims("year")
            else:
                exceed = exceed.assign_coords(year=year).expand_dims("year")

            annual_results.append(exceed)

        # -------------------------------------------------------------------
        # Concatenate all years into final DataArray
        # -------------------------------------------------------------------
        var_counts = xr.concat(annual_results, dim="year")
        var_counts.name = 'spells_of_threshold_crossings'
        # Copy original attributes and update them
        var_counts.attrs.update(year_data1.attrs)
        var_counts.attrs.update({
            "standard_name": "spells_of_theshold_crossings",
            "long_name": "Threshold Detector Output",
            "units": "number of spells per year",
            "description": "Threshold Detector Output",
            "label_units": "number of spells per year",
            "plot_label": "Spells of Theshold Crossings" })

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
    * my_detection.plot_temporal_mean(var_counts, y1, y2, set_label = 'Max Spell Length')
    * my_detection.plot_temporal_mean(var_counts, y1, y2, output_file = 'plot.png')

    my_detection: an instance of the class ThresholdDetector

    Inputs:
           self
           var_counts: DataArray with threshold crossings (created by method detect)
           y1, y2: the first and last years of the selected period
           set_label (optional): a customised label for the plot
           output_file(optional): name of a png file to save the plot

    '''

    def plot_temporal_mean(self, var_counts, y1, y2, set_label = 'Threshold Crossings', output_file = None):

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
        plt.colorbar(orientation='horizontal', label=set_label, fraction=0.03, pad=0.03)

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
           set_label (optional): a customised label for the plot
           output_file(optional): name of a png file to save the plot

    '''

    def plot_spatial_mean(self, var_counts, mylon = None, mylat = None,
                          set_label = 'Threshold Crossings', output_file = None):

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
        ax.set_ylabel(set_label)

        # Save plot if output_file is given 
        if output_file is not None:
            plt.savefig(output_file)
