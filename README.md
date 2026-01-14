## ukcp_threshold_detector

### 1. Introduction
Repository *ukcp_threshold_detector* contains Python code for the HCCP project "Creating a UKCP threshold detector to address stakeholder needs for decision-relevant climate information". The project delivers a **threshold detector capability** for the UK from the **UK Climate Projections (UKCP)**.

The code processes **UKCP18 daily gridded data (default resolution 12 km)** to compute threshold crossings for the variables *tasmax*, *tasmin*, *tas* and *pr*. It can analyse the available **16 UKCP18 ensemble members** covering the period **1981–2079**, which follow a **high-emissions pathway (RCP8.5)** for future years. Users can specify 
- the **variable** of interest,
- the **ensemble member**, and
- the **threshold value** and **detection method** (*above* or *below*).
  
Additional functionality also allows users to apply the detector to ***HadUK-Grid observations*** instead of UKCP by setting:
- parameter **obs = *True*** (default is *False*)

The tool returns **gridded fields of threshold crossings** in the same **NetCDF format** as the input data and can also produce **simple visual outputs**, such as maps of exceedance frequency or time series of area-mean values. 

> ##### Spatial resolution of input data
> The tool can analyse datasets with different spatial resolutions. The default 12 km resolution allows fast computation, whereas higher-resolution datasets (e.g. 2.2 km or 1 km) increase computational cost. Users working with high-resolution datasets may require
 additional computing resources and may wish to analyse the data in segments - for example, processing one decade at a time rather than all available years in a single job.

### 2. Code Structure
The tool uses **[xarray](https://docs.xarray.dev/en/stable/)** to read and analyse the input NetCDF files. Users can save the output detection field as a NetCDF file. The output field retains the same attributes as the input and includes a coordinate, *year*, which represents the time dimension and gives the threshold-crossing metric for each analysed year.

The code repository includes the Python scripts listed below. Further information about each component is provided through comments and examples in the scripts.

#### analysis.py
This script contains the main code of the tool and defines the class *ThresholdDetector*. The class stores information about the detection setup and provides methods to compute threshold-crossing metrics and generate simple plots of the results. Example uses and notes on its capabilities are provided in the script comments and a brief desctiption of the different components is summarised below:

- **Class ThresholdDetector**

- **Method detect.**
  This is the most basic threshold-crossing function that computes the number of days the threshold is crossed in each year. It takes an an input an instance of the ThresholdDetector and, optionally,the following (optional) parameters:
  - **select_years** : a list of years to analyse. This is useful for high-resolution data, where analysing smaller segements helps avoid  memory limitations.
  - **select_months**: a list of months to analyse. If *None* (default) then annual exceedances are computed. This is a useful option if, for example, users want to calculate exceedances in a season.
  - **years_from_dec**: if set to *True*, then years run from Dec to Nov instead of the default (Jan to Dec).
  - **output_file**: name of a NetCDF file to save the output.

- **Method detect_maxlength.**
  This method computes the maximum spell length in each year, where spells are defined as consecutive days above (or below) the threshold. The method has the same inputs as method *detect*.

- **Method detect_spells.**
   This method computes the total number of threshold-crossing spells in each year. Spells are again defined as consecutive days above (or below) the threshold. Spells of a minimumn length (*min_length*) may be specified.
   Also, spells may be considered separate only if there are at least X non-exceedance days (*decluster_days*) between them. In addition to the inputs of method *detect*, this method also include the following (optional) parameters:
  - **min_length:** if specified, only spells with at least *min_length* days are counted.
  - **decluster_days:** minimum number of days without a threshold crossing required between spells. If set to 1 (default), all spells are counted, even if only separated by 1 day.

#### analysis_utils.py
This script contains supporting functions used in the analysis, including:
- A function to create a custom colour map used for plotting.
- A function to compute the (weighted) spatial mean of a UKCP field.

#### input_datapaths.py
This script defines the file paths for UKCP18 daily data for each ensemble member. Users should update this file with the correct paths to their local data.


***
#### Example (basic use)

``` python
# Import the Detector
from analysis import ThresholdDetector

# Instantiate a detection for tasmax > 27.C
mydetection = ThresholdDetector('tasmax', 27.)

# Compute threshold crossings
thresh_exceed = mydetection.detect()

# Plot a map of the mean exceedances in the 2070s
mydetection.plot_temporal_mean(thresh_exceed, 2070, 2079)

# Plot the timeseries of annual threshold exceedances averaged over the area of Wales
mydetection.plot_spatial_mean(thresh_exceed, mylon=[-5.5, -2.5], mylat=[51.4, 53.5])

```
