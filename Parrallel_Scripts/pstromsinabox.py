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
from tqdm import tqdm
import multiprocessing
import copy_reg as copyreg
import pandas as pd
import numpy as np
import warnings
import sys
from numba import autojit
import types
import iris
import Pfuncts

if not sys.warnoptions:
    warnings.simplefilter("ignore")

copyreg.pickle(types.MethodType, Pfuncts._pickle_method)


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

        self.size_of_storm = size_of_storm
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
        self.lon = cube.coord('longitude').points.tolist()
        self.lat = cube.coord('latitude').points.tolist()
        self.cols = ['storm', 'no', 'area', 'centroid', 'box', 'life', 'u',
                     'v', 'mean', 'min', 'max', 'accreted', 'parent', 'child',
                     'cell']
        self.froot = '/nfs/a277/IMPALA/data/4km/precip_tracking_12km_hourly/'

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
            df = pd.DataFrame()
            df2 = pd.DataFrame(columns=['file'], index=[range(0, len(df))])
            df['file'] = (glob.glob(self.froot+'*/a04203*4km*.txt'))
            for rw in df.itertuples():
                if rw.file[90:92] in [str(x).zfill(2) for x in range(6, 10)]:
                    df2.loc[rw[0]] = 0
                    df2['file'].loc[rw[0]] = rw.file
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
        for row in tqdm(df.itertuples(), total=len(df), unit="file"):
            cfile = row.file  # current file
            vari = pd.read_csv(cfile, names=self.cols,  header=None,
                               delim_whitespace=True)
            # the txt files have stoms and other lines with surplus info
            var2 = vari[pd.notnull(vari['mean'])]
            size = var2.area.str[5::]
            var2.loc[:, 'area'] = pd.to_numeric(size)*144
            # If it meets our size criteria
            storms = var2[var2.area >= self.size_of_storm].reset_index(drop=True)
            # And is the centroid in our location
            storms[['centlat', 'centlon']] = storms['centroid'].str.split(',', expand=True)
            # centroid lat and lon are reffering to indicies written by matlab
            # i.e. +1 to the indice in python.
            for rw2 in storms.itertuples():
                storms['centlon'].loc[rw2[0]] = self.lon[int(pd.to_numeric(rw2.centlon)-1)]
                storms['centlat'].loc[rw2[0]] = self.lat[int(pd.to_numeric(rw2.centlat[9::])-1)]
            storms = storms[storms.centlon <= self.x2].reset_index(drop=True)
            storms = storms[storms.centlon >= self.x1].reset_index(drop=True)
            storms = storms[storms.centlat <= self.y2].reset_index(drop=True)
            storms = storms[storms.centlat >= self.y1].reset_index(drop=True)
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
            stormsdf2.mean_olr = storms['mean'].str[5::]
            # box: defines the rectangle around the storm
            # [minlatix, minlonix, nlats, nlons]
            # llat, llon, ulat, ulon
            storms[['llat', 'llon', 'nlat', 'nlon']] = storms['box'].str.split(',', expand=True)
            stormsdf2.llon = pd.to_numeric(storms.llon) - 1
            stormsdf2.llat = pd.to_numeric(storms.llat.str[4::]) - 1
            stormsdf2.ulon = (pd.to_numeric(storms.nlon)
                              + pd.to_numeric(storms.llon) - 1)
            stormsdf2.ulat = (pd.to_numeric(storms.nlat) +
                              pd.to_numeric(storms.llat.str[4::]) - 1)
            # Append to whole area
            stormsdf = pd.concat([stormsdf, stormsdf2]).reset_index(drop=True)
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
                       str(self.size_of_storm)
                       + '_lons_' + str(self.x1) + '_' + str(self.x2) +
                       '_lat_' + str(self.y1) + '_' + str(self.y2)+'.csv')
