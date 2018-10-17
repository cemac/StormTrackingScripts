# -*- coding: utf-8 -*-
"""Storm calculations

This module was developed by CEMAC as part of the AMAMA 2050 Project.
This scripts build on Work done my Rory Fitzpatrick, taking the
information generated by dataminer to cacluate desired diagnostics
about storms over desired area. 

Example:
    To use::
        module load Strom_calculations
        c = storm_calculations(x1,x2,y1,y2,size)

Attributes:
    varslist(list): List of vairables required in dataminer
    fname(str): File to extract lat and lons
    froot(str): Root folder of data

.. CEMAC_stomtracking:
   https://github.com/cemac/StormTrackingScripts
"""

""" right, so in this file we are setting up all the information needed
 for our CAPE calculations.
 Looking at the literature, we need pressure, temperature, dewpoint temperature, height, and specific  (g/kg)
 for all the intense storms. We will use the pressure values 850 hPa upwards in order to remove the iss
ues with 925 hPa T values for some storms.
"""
import iris
import scipy.stats as stat
import numpy as np
from numpy import genfromtxt as gent
import matplotlib.pyplot as plt
from iris.experimental.equalise_cubes import equalise_attributes
import collections
from matplotlib import colors
import glob
import meteocalc
import skewt
from skewt import SkewT as sk
import pandas as pd
import metpy.calc as metcalc
from metpy.plots import add_metpy_logo, SkewT
from metpy.units import units
