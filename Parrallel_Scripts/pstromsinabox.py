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
import iris
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing
import copy_reg as copyreg
import types


def _pickle_method(m):
    '''Taken from
    https://laszukdawid.com/2017/12/13/multiprocessing-in-python-all-about-pickling/
    multiprocessing with in a class requires some adaptation to pickling.
    '''
    class_self = m.im_class if m.im_self is None else m.im_self
    return getattr, (class_self, m.im_func.func_name)


copyreg.pickle(types.MethodType, _pickle_method)


def parallelize_dataframe(df, func, nice):
    '''parallelize a data frame

    Attributes:
        df: dataframe
        func: function
        nice: niceness 1/nice share of machince

    Returns:
        df: DataFrame chucked to different processors
    '''
    nprocs = int(multiprocessing.cpu_count()/nice)
    df_split = np.array_split(df, nprocs)
    pool = multiprocessing.Pool(nprocs)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def yes_or_no(question):
    reply = str(input(question+' (y/n): ')).lower().strip()
    if reply[0] == 'y':
        return True
    if reply[0] == 'n':
        return False
    else:
        return yes_or_no("You did not enter one of 'y' or 'n'. Assumed 'n'.")


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

    def create_dataframe(self):
        """Description
       Create a data frame of file names

       Attributes:
       froot(str): file root

       Returns:
       df(DataFrame): A data frame of storm file names
        """
        froot = '/nfs/a277/IMPALA/data/4km/precip_tracking_12km_hourly/'
        df = pd.DataFrame()
        df2 = pd.DataFrame(columns=['file'], index=[range(0, len(df))])
        df['file'] = (glob.glob(froot+'*/a04203*4km*.txt'))
        for rw in df.itertuples():
            if rw.file[90:92] in [str(x).zfill(2) for x in range(6, 10)]:
                df2.loc[rw[0]] = 0
                df2['file'].loc[rw[0]] = rw.file
        self.df = df2.reset_index(drop=True)
        # Storms we want have this infor
        print('generated file list...')
        return self.df

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
        return sotrmsdf

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
            print('If you not using a shared resource please specify')
            nice = 4

        if nice == 2 and shared == 'Y':
            yes_or_no(('***WARNING***: You are asking to use half a shared computer \
            consider fair use of shared resources, do you wish to continue?\
            Y or N'))

        if not ans:
            print('Please revise nice number to higher value and try again...')
            return

        df = self.create_dataframe()
        stormsalldf = pd.DataFrame(columns=self.varslist)
        pstorms = parallelize_dataframe(df, self.find_the_storms, nice)
        pstorms.to_csv(idstring + 'storms_over_box_area' +
                       str(self.size_of_storm)
                       + '_lons_' + str(self.x1) + '_' + str(self.x2) +
                       '_lat_' + str(self.y1) + '_' + str(self.y2)+'.csv')
        return print('generated idstring + 'storms_over_box_area' +
                       str(self.size_of_storm)
                       + '_lons_' + str(self.x1) + '_' + str(self.x2) +
                       '_lat_' + str(self.y1) + '_' + str(self.y2)+'.csv')
