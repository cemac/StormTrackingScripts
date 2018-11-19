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


import sys
import glob
import gc
import pandas as pd
import numpy as np
import iris


class StormInBox(object):
    '''Description
       Stage 1: currently a suit of functions for finding information on
       storms in region and Generating cvs files of that information.

       Attributes:
        x1(int): longitude index East
        x2(int): longitude index West
        y1(int): latitude index South
        y2(int): latitude index North
        size_of_storm(int): size of storm in km e.g 5000
        varslist: variables to be collected
    '''
    def __init__(self, x1, x2, y1, y2, size_of_storm, idstring, run='cc', root=None):

        self.varslist = ['stormid', 'year', 'month', 'day', 'hour', 'llon',
                         'ulon', 'llat', 'ulat', 'centlon', 'centlat', 'area',
                         'mean_olr']
        self.size = size_of_storm
        self.idstring = idstring
        self.x1, self.x2, self.y1, self.y2 = [x1, x2, y1, y2]
        self.m1, self.m2 = [6, 10]
        # Empty dataframe to be populated
        self.stormsdf = pd.DataFrame(columns=self.varslist)
        # Get grid lat lons
        fname = ('/nfs/a277/IMPALA/data/4km/a03332_12km/a03332_A1hr_mean_' +
                 'ah261_4km_200012070030-200012072330.nc')
        cube = iris.load(fname)[1]
        self.lon = cube.coord('longitude').points
        self.lat = cube.coord('latitude').points
        # Data root, will depend on
        if run == 'cc':
            root = '/nfs/a277/IMPALA/data/4km/'
        elif run == 'fc':
            root = '/nfs/a299/IMPALA/data/fc/4km/'
        else:
            sys.exit('Please specify data root eg. /nfs/a299/IMPALA/data/fc/4km/')
        self.froot = root + 'precip_tracking_12km_hourly/'

    def stormsinthebox(self):
        '''stormsinthebox
            Description: search through storm files (iterable glob from file
            pattern), find the files for april to july. Read the file and
            collect the storms in the selected area and size. Dump to csv.
            Attributes:
                froot: where the data is stored for this run
                cols: header names of csv file read

            Returns:
                stormsdf2: csv dump of generated pandas dataframe

        '''
        cols = ['storm', 'no', 'area', 'centroid', 'box', 'life', 'u', 'v',
                'mean', 'min', 'max', 'accreted', 'parent', 'child', 'cell']
        # Only 1 file per day!
        for rw in glob.iglob(self.froot+'*/a04203*4km*0030-*2330_*.txt'):
            if rw[90:92] in [str(x).zfill(2) for x in range(self.m1, self.m2)]:
                cfile = rw
                # read in whole csv
                allvars = pd.read_csv(cfile, names=cols, header=None,
                                      delim_whitespace=True)
                # the txt files have stoms and then child cells with
                # surplus info select parent storms
                parent = allvars[pd.notnull(allvars['mean'])]
                size = parent.area.str[5::]
                parent.loc[:, 'area'] = pd.to_numeric(size)*144
                # If it meets our size criteria
                storms = parent[parent.area >= self.size].reset_index(drop=True)
                # get rid of irrelvant storms
                # box: defines the rectangle around the storm
                # [minlatix, minlonix, nlats, nlons]
                # llat, llon, ulat, ulon
                storms[['llat', 'llon', 'nlat', 'nlon']] = storms['box'].str.split(',', expand=True)
                llats = storms.llat.str[4::]
                llons = storms.llon
                storms.llon = self.lon[np.array([pd.to_numeric(storms.llon)
                                                 - 1]).astype(int)][0]
                storms.llat = self.lat[np.array([pd.to_numeric(llats)
                                                 - 1]).astype(int)][0]
                storms['ulon'] = self.lon[np.array([pd.to_numeric(llons)
                            + pd.to_numeric(storms.nlon) - 1]).astype(int)][0]
                storms['ulat'] = self.lat[np.array([pd.to_numeric(llats) +
                          pd.to_numeric(storms.nlat) - 1]).astype(int)][0]
                storms['centlon'] = (storms.ulon + storms.llon) / 2.0
                storms['centlat'] = (storms.ulat + storms.llat) / 2.0
                storms = storms[self.x1 <= storms.centlon]
                storms = storms[storms.centlon <= self.x2]
                storms = storms[self.y1 <= storms.centlat]
                # Final criteria so reset the indices to 0-N
                storms = storms[storms.centlat <= self.y2].reset_index(drop=True)
                if storms.empty:
                    gc.collect()
                    continue
                # lets create a data frame of the varslist components
                # join DataFrame to stormsdf and move on to next file.
                # Make a dataframe to fill this time steps storm data
                stormsdf2 = pd.DataFrame(columns=self.varslist)
                stormsdf2.stormid = storms.no
                datestamp = pd.to_datetime(cfile[86:98])
                stormsdf2.month = datestamp.month
                stormsdf2.day = datestamp.day.astype(int)
                stormsdf2.hour = datestamp.hour.astype(int)
                stormsdf2.year = datestamp.year.astype(int)
                stormsdf2.area = storms.area
                stormsdf2.llon = storms.llon
                stormsdf2.llat = storms.llat
                stormsdf2.ulon = storms.ulon
                stormsdf2.ulat = storms.ulat
                stormsdf2.centlon = storms.centlon
                stormsdf2.centlat = storms.centlat
                stormsdf2.mean_olr = storms['mean'].str[5::]
                self.stormsdf = pd.concat([self.stormsdf, stormsdf2]).reset_index(drop=True)
                gc.collect()  # Clear cache of unreference memeory
        stormsdf = stormsdf.drop_duplicates(subset='stormid', keep='first',
                                            inplace=False).reset_index(drop=True)
        self.stormsdf.to_csv(self.idstring + 'storms_over_box_area' + str(self.size)
                             + '_lons_' + str(self.x1) + '_' + str(self.x2) +
                             '_lat_' + str(self.y1) + '_' + str(self.y2)+'.csv')
