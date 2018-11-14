# -*- coding: utf-8 -*-
"""Stormsin a box

This module was developed by CEMAC as part of the AMAMA 2050 Project.
This scripts build on Work done my Rory Fitzpatrick, taking the stroms saved in
the folderroot text files for a certain area and saving the specified variables
into a cvs file to be used in dataminer.py

Example:
    To use::
        module load stromsinabox.py
        c = stromsinabox(x1,x2,y1,y2,size)

Attributes:
    varslist(list): List of vairables required in dataminer
    fname(str): File to extract lat and lons
    froot(str): Root folder of data

.. CEMAC_stomtracking:
   https://github.com/cemac/StormTrackingScripts
"""


import glob
import iris
import pandas as pd
import numpy as np


class storminbox(object):
    '''Description
       Stage 1: currently a suit of functions for finding information on
       storms in region and Generating cvs files of that information.

       Attributes:
        x1(int): longitude index East
        x2(int): longitude index West
        y1(int): latitude index South
        y2(int): latitude index North
        size_of_storm(int): size of storm in km e.g 5000
    '''
    def __init__(self, x1, x2, y1, y2, size_of_storm, idstring):

        self.varslist = ['stormid', 'year', 'month', 'day', 'hour', 'llon',
                         'ulon', 'llat', 'ulat', 'centlon', 'centlat', 'area',
                         'mean_olr']
        stormsdf = pd.DataFrame(columns=self.varslist)
        fname = ('/nfs/a277/IMPALA/data/4km/a03332_12km/a03332_A1hr_mean_' +
                 'ah261_4km_200012070030-200012072330.nc')
        cube = iris.load(fname)[1]
        self.lon = cube.coord('longitude').points
        self.lat = cube.coord('latitude').points
        froot = '/nfs/a277/IMPALA/data/4km/precip_tracking_12km_hourly/'
        df = pd.DataFrame()
        df2 = pd.DataFrame(columns=['file'])
        i = 0
        cols = ['storm', 'no', 'area', 'centroid', 'box', 'life', 'u', 'v',
                'mean', 'min', 'max', 'accreted', 'parent', 'child', 'cell']
        for rw in glob.iglob(froot+'*/a04203*4km*txt'):
            if rw[90:92] in [str(x).zfill(2) for x in range(4, 7)]:
                i += 1
                cfile = rw
                vari = pd.read_csv(cfile, names=cols,  header=None,
                                   delim_whitespace=True)
                # the txt files have stoms and then child cells with surplus info
                var2 = vari[pd.notnull(vari['mean'])]
                size = var2.area.str[5::]
                var2.loc[:, 'area'] = pd.to_numeric(size)*144
                # If it meets our size criteria
                storms = var2[var2.area >= size_of_storm].reset_index(drop=True)
                # And is the centroid in our location
                storms[['centlat', 'centlon']] = storms['centroid'].str.split(',', expand=True)
                # centroid lat and lon are reffering to indicies written by matlab
                # i.e. +1 to the indice in python.
                centlons = self.lon[np.array(pd.to_numeric(storms.centlon)
                                             - 1).astype(int)][0]
                centlats = self.lat[np.array(pd.to_numeric(storms.centlat.str[9::])
                                             - 1).astype(int)][0]
                storms['centlon'] = centlons
                storms['centlat'] = centlats
                # get rid of irrelvant storms
                storms = storms[storms.centlon <= x2].reset_index(drop=True)
                storms = storms[storms.centlon >= x1].reset_index(drop=True)
                storms = storms[storms.centlat <= y2].reset_index(drop=True)
                storms = storms[storms.centlat >= y1].reset_index(drop=True)
                # Any in this file?
                if len(storms.index) == 0:
                    continue
                # lets create a data frame of the varslist components
                # join DataFrame to stormsdf and move on to next file.
                # Make a dataframe to fill this time steps storm data
                stormsdf2 = pd.DataFrame(columns=self.varslist)
                stormsdf2.stormid = storms.no
                datestamp = pd.to_datetime(cfile[86:98])
                stormsdf2.month = datestamp.month
                stormsdf2.day = datestamp.day
                stormsdf2.hour = datestamp.hour
                stormsdf2.year = datestamp.year
                stormsdf2.area = storms.area
                stormsdf2.centlon = storms.centlon
                stormsdf2.centlat = storms.centlat
                stormsdf2.mean_olr = storms['mean'].str[5::]
                # box: defines the rectangle around the storm
                # [minlatix, minlonix, nlats, nlons]
                # llat, llon, ulat, ulon
                storms[['llat', 'llon', 'nlat', 'nlon']] = storms['box'].str.split(',', expand=True)
                stormsdf2.llon = self.lon[np.array([pd.to_numeric(storms.llon)
                                          - 1]).astype(int)][0]
                stormsdf2.llat = self.lat[np.array([pd.to_numeric(storms.llat.str[4::])
                                          - 1]).astype(int)][0]
                stormsdf2.ulon = self.lon[np.array([pd.to_numeric(storms.nlon)
                          + pd.to_numeric(storms.llon) - 1]).astype(int)][0]
                stormsdf2.ulat = self.lat[np.array([pd.to_numeric(storms.nlat) +
                          pd.to_numeric(storms.llat.str[4::]) - 1]).astype(int)][0]
                # Append to whole area
                stormsdf = pd.concat([stormsdf, stormsdf2]).reset_index(drop=True)
        stormsdf.to_csv(idstring + 'storms_over_box_area' + str(size_of_storm)
                        + '_lons_' + str(x1) + '_' + str(x2) + '_lat_' +
                        str(y1) + '_' + str(y2)+'.csv')
