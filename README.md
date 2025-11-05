## ukcp_threshold_detector

Repository *ukcp_threshold_detector* contains Python code for the HCCP project "Creating a UKCP threshold detector to address stakeholder needs for decision-relevant climate information". The project intends to produce a **threshold detector capability** for the UK from the **UK Cllimate Projections (UKCP)**.

The code processes **UKCP18 daily gridded data (12 km)** to compute threshold exceedances for the variables *tasmax*, *tasmin*, *tas* and *pr*. It can analyse up to **16 UKCP18 ensemble members** covering the period **1981–2079**, following a **high-emissions pathway (RCP8.5)** for future years. Users can specify 
- the **variable** of interest,
- the **ensemble member(s)**, and
- the **threshold value** and **detection method** (*above* or *below*).

The tool returns **gridded fields of threshold crossings** in the same **NetCDF format** as the input data and can also produce **simple visual outputs**, such as maps of exceedance frequency or time series of area-mean values. 

The repository includes the Python scripts listed below. Further information about each component is provided through comments and examples in the scripts.

### analysis.py
This script contains the main code of the tool and defines the class *ThresholdDetector*. The class stores information about the detection setup and provides methods to compute threshold crossings and generate simple plots of the results. Example uses and notes on its capabilities are provided in the script comments.

### analysis_utils.py
This script contains supporting functions used in the analysis, including:
- A function to create a custom colour map used for plotting.
- A function to compute the (weighted) spatial mean of a UKCP field.

### input_datapaths.py
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
