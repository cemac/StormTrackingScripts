# -*- coding: utf-8 -*-
"""Paralell Stormsin a box

This module was developed by CEMAC as part of the AMAMA 2050 Project.
This scripts build on Work done my Rory Fitzpatrick, taking the stroms saved in
the folderroot text files for a certain area and saving the specified variables
into a cvs file to be used in dataminer.py. This is the Paralell version of
stromsinabox.py

Example:
    To use::
        module load pstromopinabox.py
        c = pstromsinabox.storminbox(x1, x2, y1, y2, size_of_storm, idstring)
        c.genstormboxcsv()

Attributes:
    varslist(list): List of vairables required in dataminer
    fname(str): File to extract lat and lons
    froot(str): Root folder of data

.. CEMAC_stomtracking:
   https://github.com/cemac/StormTrackingScripts
"""


import glob
import copy_reg as copyreg
import warnings
import sys
import types
import gc
import iris
import Pfuncts
import pandas as pd
import numpy as np

if not sys.warnoptions:
    warnings.simplefilter("ignore")

copyreg.pickle(types.MethodType, Pfuncts._pickle_method)


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
    '''
    def __init__(self, x1, x2, y1, y2, size_of_storm, idstring, run='cc', root=None):

        self.size = size_of_storm
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.idstring = idstring
        self.varslist = ['stormid', 'year', 'month', 'day', 'hour', 'llon',
                         'ulon', 'llat', 'ulat', 'centlon', 'centlat', 'area',
                         'mean_olr']
        fname = ('/nfs/a277/IMPALA/data/4km/a03332_12km/a03332_A1hr_mean_' +
                 'ah261_4km_200012070030-200012072330.nc')
        cube = iris.load(fname)[1]
        self.lon = cube.coord('longitude').points
        self.lat = cube.coord('latitude').points
        self.cols = ['storm', 'no', 'area', 'centroid', 'box', 'life', 'u',
                     'v', 'mean', 'min', 'max', 'accreted', 'parent', 'child',
                     'cell']
        # Data root, will depend on
        if run == 'cc':
            root = '/nfs/a277/IMPALA/data/4km/'
        elif run == 'fc':
            root = '/nfs/a299/IMPALA/data/fc/4km/'
        else:
            sys.exit('Please specify data root eg. /nfs/a299/IMPALA/data/fc/4km/')
        self.froot = root + 'precip_tracking_12km_hourly/'

    def create_dataframe(self):
        """Description
       Create a data frame of file names

       Attributes:
       froot(str): file root

       Returns:
       df(DataFrame): A data frame of storm file names
        """
        try:
            df = pd.read_csv('filelist.csv', sep=',')
            return df
        except IOError:
            df2 = pd.DataFrame(columns=['file'])
            i = 0
            for rw in glob.iglob(self.froot+'*/a04203*4km*.txt'):
                if rw.file[90:92] in [str(x).zfill(2) for x in range(4, 7)]:
                    i += 1
                    df2.loc[i] = 0
                    df2['file'].loc[i] = rw
            df = df2.reset_index(drop=True)
            df.to_csv('filelist.csv', sep=',')
        # Storms we want have this infor
        print('generated file list...')
        return df

    def find_the_storms(self, df):
        '''find the storms
            Description:
                Find the storms in box specified and above size specified. A
                timer is added giving the average progess of each processor.

            Attributes:
                df: dataframe of file names

            Returns:
                stormsdf: DataFrame of storms meeting criteria
        '''
        stormsdf = pd.DataFrame(columns=self.varslist)
        for row in df.itertuples():
            cfile = row.file  # current file
            # read in whole csv
            allvars = pd.read_csv(cfile, names=self.cols, header=None,
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
            storms['centlon'] = (storms.ulon + storms.ulon) / 2.0
            storms['centlat'] = (storms.ulat + storms.ulat) / 2.0
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
            stormsdf2.llon = storms.llon
            stormsdf2.llat = storms.llat
            stormsdf2.ulon = storms.ulon
            stormsdf2.ulat = storms.ulat
            stormsdf2.centlon = storms.centlon
            stormsdf2.centlat = storms.centlat
            datestamp = pd.to_datetime(cfile[86:98])
            stormsdf2.month = datestamp.month
            stormsdf2.day = datestamp.day
            stormsdf2.hour = datestamp.hour
            stormsdf2.year = datestamp.year
            stormsdf2.area = storms.area
            stormsdf2.mean_olr = storms['mean'].str[5::]
            stormsdf = pd.concat([stormsdf, stormsdf2]).reset_index(drop=True)
            gc.collect()  # Clear cache of unreference memeory
        return stormsdf

    def genstormboxcsv(self, nice=4, shared='Y'):
        '''generate storms csvs

            Attributes:
                nice(int): niceness 1/nice share of machince.
                shared(str): 'Y' or 'N' using shared resource.

            Returns:
                csvfile: file of stromsinabox
        '''
        if nice == 1 and shared == 'Y':
            print('WARNING: Using a whole machine is not very nice')
            print('Setting to quater machine....')
            print('If you are not using a shared resource please specify')
            nice = 4

        if nice == 2 and shared == 'Y':
            ans = Pfuncts.yes_or_no(('***WARNING***: You are asking to use half a shared computer \
            consider fair use of shared resources, do you wish to continue?\
            Y or N'))

            if not ans:
                print('Please revise nice number to higher value and try again...')
                return
        df = self.create_dataframe()
        pstorms = Pfuncts.parallelize_dataframe(df, self.find_the_storms, nice)
        pstorms.to_csv(self.idstring + 'storms_over_box_area' +
                       str(self.size)
                       + '_lons_' + str(self.x1) + '_' + str(self.x2) +
                       '_lat_' + str(self.y1) + '_' + str(self.y2)+'.csv')
