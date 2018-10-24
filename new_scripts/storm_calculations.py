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
right, so in this file we are setting up all the information needed
for our CAPE calculations.
Looking at the literature, we need pressure, temperature, dewpoint temperature,
height, and specific  (g/kg) for all the intense storms. We will use the
pressure values 850 hPa upwards in order to remove the issues with 925 hPa T
values for some storms.
"""

import pandas as pd


class stormcalcs(object):
    '''Description
       Stage 1: currently a suit of functions for finding information on
       storms in region and Generating cvs files of that information.

    '''
    def __init__(self, csvname, altcsvname=None):

        fname = self.csvname
        df = pd.read_csv(csvname, sep=',')
        cold = df.mean_T15_1800 - df.mean_T15_1200
        mslp_diff = df.eve_mslp_mean - df.midday_mslp
        u_diff_10m = df.eve_wind_99p - df.midday_wind
        u3_diff_10m = df.eve_wind3_99p - df.midday_wind3
        precip99 = df.precip_99th_perc*3600
        precipvol = df.precip_accum*3600
