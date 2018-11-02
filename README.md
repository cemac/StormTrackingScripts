# Storm Tracking Scripts #

## Description ##

Development of Storm Tracking Scripts from ICAS dynamics group.

## Requirements ##

 * [Python](https://www.anaconda.com/download/) (Standard anaconda package)
 * [Iris](https://scitools.org.uk/iris/docs/latest)

A full list of Requirements is listed in pipfile however Iris is not available via pip install.

anaconda python is strongly recommended.

<hr>

# Version 1.0 #

The first major release of developed scripts. Building from Rory's storm tracking scripts these functions have been moved a modular system. Finding storms in an area, extracting information about those storms and calculating some statistics about those storms and producing a set of Standard plots.

## Python 3 ##
Use the python 3 download. Requires [python SkewT](https://github.com/tjlang/SkewT)

## New Features ##

* Removed Hard code, can be used on different model runs
* Serial and Parallel versions available
* Includes a progress bar to long running components.
* Larger runs could be ran on a node on HPC (more details coming soon)
* Python 3 version available


## Usage ##

*To be filled in once completed*

<hr>

## Installation (recommended)##
StormScripts requires a few nonstandard modules (skewt, meteocalc). anaconda or miniconda is recommended method of Installation.

**python 3 (recommended)**

````bash

# download anaconda installer
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh #skip if you already have anaconda
export PATH="$HOME/miniconda3/bin:$PATH" #skip if you already have anaconda
conda create --name Storms python=3.6
source activate Storms
conda install -c conda-forge iris
conda install numba
pip install meteocalc
git clone https://github.com/tjlang/SkewT.git
cd SkewT
python setup.py install
conda clean -t

````


```bash

# download anaconda installer
wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh
bash Miniconda2-latest-Linux-x86_64.sh #skip if you already have anaconda
export PATH="$HOME/miniconda3/bin:$PATH" #skip if you already have anaconda
conda create --name Storms_py2 python=2.7
source activate Storms_py
conda install -c conda-forge iris
conda install numba
pip install meteocalc
pip install skewt
conda clean -t
```

## Issue Templates ##

* Please use our issue templates for feature requests and bug fixes.

## Upcoming Features ##

1. Python GUI
2. Documentation improvements
3. Feature requests

<hr>

## Acknowledgements ##

These scripts have been developed based off the work done by the [Institute of Climate and Atmospheric Science (ICAS)](http://www.see.leeds.ac.uk/research/icas/) [dynamics](http://www.see.leeds.ac.uk/research/icas/research-themes/atmosphere/) group and the University of Leeds. This work was originally part of the [African Monsoon Multidisciplinary Analysis](https://www.amma2050.org/) (AMMA-2050) project.
