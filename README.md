<div align="center">
<a href="https://www.cemac.leeds.ac.uk/">
  <img src="https://github.com/cemac/cemac_generic/blob/master/Images/cemac.png"></a>
  <br>
</div>

# Storm Tracking Scripts #

[![GitHub release](https://img.shields.io/badge/release-v.2.0-blue.svg)](
https://github.com/cemac/StormTrackingScripts/releases/tag/v.2.0)

## Description ##

Development of Storm Tracking Scripts from ICAS dynamics group.

## Requirements ##

 * [Python](https://www.anaconda.com/download/) (Standard anaconda package)
 * [Iris](https://scitools.org.uk/iris/docs/latest)

A full list of Requirements is listed in the yml file (python 2)

<hr>


# Version 2.0 #

The first major release of developed scripts. Building from Rory's storm tracking scripts these functions have been moved a modular system. Finding storms in an area, extracting information about those storms and calculating some statistics about those storms and producing a set of Standard plots.


## Python 3 ##
Use the python 3 download. Requires [python SkewT](https://github.com/tjlang/SkewT)

## New Features ##

* Removed Hardcoding, can be used on different model runs, easily adapted for different variable etc
* Runs in Parallel - giving a speed improvement
* Python 3 version available
* Reduction in lines of code from 3600 to 1100
* Large increase in comments docstrings

## Usage ##

Keep Python package folder.

Can be used to find a subset of storms from the precip_tracking_12km_hourly
text files and mine the relevant data. Comes with plotting tools.

```Python

from os.path import expanduser
import numpy as np
import pandas as pd
import StormScriptsPy3 as SSP3


# Box dimensions
x1, x2 = [345, 375]
y1, y2 = [10, 18]
# run id
size_of_storm = 5000
# corresponding data_dirs
dataroot = '/path/to/data/'
# Where to store the generated files
stormhome = expanduser("~")+'/storm_project/'
# label these storms
idstring = 'mystorms'
# print some help?
SSP3.S_Box.StormInBox?
# Define your storms
c = SSP3.S_Box.StormInBox(x1, x2, y1, y2, size_of_storm, idstring)
# generate the list of storms
c.gen_storm_box_csv()
# point to data and if you want the full set of calculations done
dmf = SSP3.dm_functions(dataroot, CAPE='Y', TEPHI='Y')
csvname = 'the_file_you_just_generated.csv'
storms_to_keep = pd.read_csv(csvname, sep=',')
dmf.genvarscsv(idstring, storms_to_keep)

```

<hr>

## Installation (recommended) ##

Storm Scripts requires a few non-standard modules (skewt, meteocalc). anaconda or miniconda is the recommended method of Installation.

### Python 2 ###

```bash

# download anaconda installer
wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
bash Miniconda2-latest-Linux-x86_64.sh #skip if you already have anaconda
export PATH="$HOME/miniconda3/bin:$PATH" #skip if you already have anaconda
conda clean -t

```

### Python 3 ###

*SkewT python 3 support is not yet in the anaconda cloud coming soon, for now an
extra step is required*

````bash

# download anaconda installer
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh #skip if you already have anaconda
export PATH="$HOME/miniconda3/bin:$PATH" #skip if you already have anaconda


git clone https://github.com/tjlang/SkewT.git
cd SkewT
python setup.py install
conda clean -t

````


## Issue Templates ##

* Please use our issue templates for feature requests and bug fixes.

## Upcoming Features ##

0. Plotting tools
2. Documentation improvements
4. STASH code variable library
3. Feature requests
4. Switch to CF-python
5. minor speed up
1. Python GUI

<hr>

## Acknowledgements ##
These scripts have been developed based on the work done by the [Institute of Climate and Atmospheric Science (ICAS)](http://www.see.leeds.ac.uk/research/icas/) [dynamics](http://www.see.leeds.ac.uk/research/icas/research-themes/atmosphere/) group and the University of Leeds. This work was originally part of the [African Monsoon Multidisciplinary Analysis](https://www.amma2050.org/) (AMMA-2050) project.
