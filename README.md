## ukcp_threshold_detector

- [1. Introduction](#1-introduction)
- [2. Code Structure](#2-code-structure)
- [3. Examples](#3-examples)

### 1. Introduction
Repository *ukcp_threshold_detector* contains Python code for the HCCP project "Creating a UKCP threshold detector to address stakeholder needs for decision-relevant climate information". The project delivers a **threshold detector capability** for the UK from the **UK Climate Projections (UKCP)**.

The code processes **UKCP18 daily gridded data (default resolution 12 km)** to compute threshold crossings for the variables *tasmax*, *tasmin*, *tas*, *pr*, *uas*, *vas*, and *sfcWind*. It can analyse the available **16 UKCP18 ensemble members** covering the period **1981–2079**, which follow a **high-emissions pathway (RCP8.5)** for future years. Users can specify 
- the **variable** of interest,
- the **ensemble member**, and
- the **threshold value** and **detection method** (*above* or *below*).
  
Additional functionality also allows users to apply the detector to **HadUK-Grid** observations instead of UKCP by setting:
- parameter **obs = *True*** (default is *False*)

The tool returns **gridded fields of threshold crossings** in the same **NetCDF format** as the input data and can also produce **simple visual outputs**, such as maps of exceedance frequency or time series of area-mean values. 

> ##### Spatial resolution of input data
> The tool can analyse datasets with different spatial resolutions. The default 12 km resolution allows fast computations, whereas higher-resolution datasets (e.g. 2.2 km or 1 km) increase the computational cost. Users working with high-resolution datasets may require additional computing resources and may wish to analyse the data in segments - for example, processing one decade at a time rather than all available years in a single job. The tool allows users to select and analyse subsets of the available years.

### 2. Code Structure
The tool uses **[xarray](https://docs.xarray.dev/en/stable/)** to read and analyse the input NetCDF files. Users can save the output detection field as a NetCDF file. The output field retains the same attributes as the input and includes a coordinate, *year*, which represents the time dimension and gives the threshold-crossing metric for each analysed year.

> ##### Code logic
> The tool is built around a high-level class called *ThresholdDetector*, with which users can create a detection *instance* by specifying the variable of interest, the threshold, and the crossing method (going above or below the threshold). Optionally, they may also specify the ensemble member, or indicate that observations should be used for the analysis instead of UKCP data. Once an instance is created, which holds all the high-level information, users can apply it to multiple functions (called *methods* in Python) without the need to re-specify the high-level information (variable, threshold, crossing method) each time. The same instance can be used to call different methods as required. The tool currently includes three methods that compute three different detection metrics (threshold-crossing counts, number of spells, and maximum spell length), as well as two additional methods for basic visualisation of the computed metrics.

The code repository includes the Python scripts listed below. Further information about each component is provided through comments and examples in the scripts.

#### analysis.py
This script contains the main code of the tool and defines the class *ThresholdDetector*. The class stores information about the detection setup and provides methods to compute threshold-crossing metrics and generate simple plots of the results. Example uses and notes on its capabilities are provided in the script comments and a brief desctiption of the different components is summarised below:

- **Class ThresholdDetector**
This is the basic building block of the tool and the starting point of all computations, which are performed using a set of methods defined within the class.

  - Class Inputs:
    - **var**: input variable - can be one of 'tasmax', 'tasmin', 'tas', 'pr'
    - **threshold**: threshold value
    - **ens** (optional): an integer indicating the ensemble member (default = 1)
    - **obs**(optional): True if analysing HadUK-Grid observations (default = False)
    - **method** (optional): threshold crossing method - can be 'above' (default) or 'below'
  - Class Attributes (information created and stored in the instance when the class is called):
    - **var**: input variable
    - **threshold**: threshold value
    - **ens**: ensemble member
    - **method**: threshold crossing method
    - **indata**: directory where the input files (NetCDF - daily data) are stored

The methods of *ThersholdDetector* that compute threshold-crossing metrics are:
- **Method detect.**
 This is the most basic threshold-crossing function that computes the number of days the threshold is crossed in each year. It takes an an input an instance of the ThresholdDetector and, optionally, the following parameters:
  - **select_years**: a list of years to analyse. This is useful for high-resolution data, where analysing smaller segements helps avoid  memory limitations.
  - **select_months**: a list of months to analyse. If *None* (default) then annual exceedances are computed. This is a useful option if, for example, users want to calculate exceedances in a season.
  - **years_from_dec**: if set to *True*, then years run from Dec to Nov instead of the default (Jan to Dec).
  - **output_file**: name of a NetCDF file to save the output.

- **Method detect_maxlength.**
  This method computes the maximum spell length in each year, where spells are defined as consecutive days above (or below) the threshold. The method has the same inputs as method *detect*.

- **Method detect_spells.**
   This method computes the total number of threshold-crossing spells in each year. Spells are again defined as consecutive days above (or below) the threshold. Spells of a minimumn length (*min_length*) may be specified.
   Also, spells may be considered separate only if there are at least X non-exceedance days (*decluster_days*) between them. In addition to the inputs of method *detect*, this method also includes the following (optional) parameters:
  - **min_length:** if specified, only spells with at least *min_length* days are counted.
  - **decluster_days:** minimum number of days without a threshold crossing required between spells. If set to 1 (default), all spells are counted, even if only separated by 1 day.

The two methods of *ThersholdDetector* for basic output visualisation are listed below. 

***Important note:*** the plotting methods are only to be applied to the UK region covered by UKCP or HadUK-Grid data and may not work correctly if the input fields have non-stardard co-ordinate names or grid specifications.

- **Method plot_temporal_mean.**
  This method plots a map of the mean threshold-crossing metric over a period starting in y1 and ending in y2. It takes as an input an instance if the ThresholdDetector as well as  the following:
  
  - **var_counts**: a DataArray with the threshold-crossing metric (created by one of the detect methods)
  - **y1, y2**: the first and last years of the selected period
  - **set_label** (optional): a customised label for the plot
  - **output_file**(optional): name of a png file to save the plot

- **Method plot_spatial_mean.**
  This method plots the timeseries of the annual mean threshold-crossing metric over an area. It takes as an input an instance if the ThresholdDetector as well as  the following:
  
  - **var_counts**: a DataArray with the threshold-crossing metric (created by one of the detect methods)
  - **mylon, mylat** (optional): 2-dimensional lists with the coordinates of an area to extract. If not given, the mean is computed over the entire area
  - **set_label** (optional): a customised label for the plot
  - **output_file** (optional): name of a png file to save the plot

#### analysis_utils.py
This script contains supporting functions used in the analysis, including:
- **Function make_color_cmap**: a function that create a custom colour map used for map plotting.
- **Function cumulative_runlength**: a function that computes the run length of consecutive threshold exceedances along the time dimension .
- **Function make_spatial_mean**: a function that computes the weigthed spatial mean of UKCP or HadUK-Grid fields for each time slice.
- **Function gwl_ukcp18**: a function that takes as input the threshold metric created by the detector and returns it on a selected Global Warming Level (GWL). Note that this function is to be used only for UKCP18 esnemble members, as they are the only input for which the detector knows the time slices corresponding to different GWLs. The user must provide the ensemble member and GWL of interest (available levels: 1, 1.5, 2, 2.5, 3, and 4 degrees). An example is provided in Section 3.

#### input_datapaths.py
This script defines the file paths for UKCP18 daily data for each ensemble member (and HadUKGrid-data, if required). Users should edit this file to provide the paths to their local data. 


***
### 3. Examples

- Basic use
``` python
# Import the Detector
from analysis import ThresholdDetector

# Instantiate a detection for tasmax > 27.C
mydetection = ThresholdDetector('tasmax', 27.)

# Compute threshold crossings
thresh_metric = mydetection.detect()

# Plot a map of the mean exceedances in the 2070s
mydetection.plot_temporal_mean(thresh_metric, 2070, 2079)

# Plot the timeseries of annual threshold exceedances averaged over the area of Wales
mydetection.plot_spatial_mean(thresh_metric, mylon = [-5.5, -2.5], mylat = [51.4, 53.5])
```

- Creating an instance
``` python
# Example 1: Max temperature above 30.C
mydetection = ThresholdDetector('tasmax', 30.)

# Example 2: Min temperature below 0.C, ensemble member 8
mydetection = ThresholdDetector('tasmin', 0., method = 'below', ens = 8)

# Example 3: Max temperature above 27.C, computed using observations
mydetection = ThresholdDetector('tasmax', 27., obs = True)
```

- Computing different detection metrics
``` python
# In the following examples an instance has been created for tasmax > 27.
mydetection = ThresholdDetector('tasmax', 27.)

# Example 1: Counts. Computes number of days when the threshold is exceeded.
thresh_metric_1 = mydetection.detect()

# Example 2: Spells. Computes number of spells in a year, i.e. consecutive days when the threshold is exceeded. In this case spells can be of any length (>= 1day).
thresh_metric_2 = mydetection.detect_spells()

# Example 3: Spells. Computes number of spells in a year, but only for spells >= 5 days.
thresh_metric_3 = mydetection.detect_spells(min_length = 5)

# Example 4: Spells. As example 3, but spells separated by 3 days or less are counted as a single event.
thresh_metric_4 = mydetection.detect_spells(min_length = 5, decluster_days = 3)

# Example 5: Max spell length. Computes the length of the longest spell of each year.
thresh_metric_5 = mydetection.detect_maxlength()

# ADDITIONAL FUNCTIONALITY
# Basic detection
thresh_metric = mydetection.detect()
# Analyse only selected years
thresh_metric = mydetection.detect(select_years = [2070, 2071, 2072, 2073, 2073])
# Compute threshold-crossings only in summer months
thresh_metric = mydetection.detect(select_months = [6, 7, 8])
# Start years in December instead of January
thresh_metric = mydetection.detect(years_from_dec = True)
# Save output in a NetCDF file
thresh_metric = mydetection.detect(output_file = 'thresh_metric.nc')
```

- Visualisation
``` python
# Plot map of the metric over period 2020-2029 and save into a png file
mydetection.plot_temporal_mean(thresh_metric, 2020, 2029, output_file = 'plot.png')

# Add a customised label
mydetection.plot_temporal_mean(thresh_metric, 2020, 2029, set_label = 'Max Spell Length')

# Plot timeseries of the metric value averaged over the entire region and save in file
mydetection.plot_spatial_mean(thresh_metric, output_file = 'plot.png')

# Plot timeseries of the metric value averaged over the London area
mydetection.plot_spatial_mean(thresh_metric, mylon = [-0.6, 0.4], mylat = [51.2, 51.8])
```

- Metric on GWLs
``` python
# Import the detector and GWL functions
from analysis import ThresholdDetector
from analysis_utils import gwl_ukcp18

# Create a simple threshold metric, as in the earlier basic use example,
# for counting days with tasmax > 27.
mydetection = ThresholdDetector('tasmax', 27.)
thresh_metric = mydetection.detect()

# Compute the metric for a GWL of 2 degrees and save it in a file
thresh_metric_gwl = gwl_ukcp18(thresh_metric, ens = mydetection.ens, gwl=2.0, output_file = 'thresh_2deg_gwl.nc')


