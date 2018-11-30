# -*- coding: utf-8 -*-
"""Stormsin a box

.. module:: S_Box
    :platform: Unix
    :synopis:

.. moduleauther: CEMAC (UoL)

.. description: This module was developed by CEMAC as part of the AMAMA 2050
   Project. This scripts build on Work done my Rory Fitzpatrick, taking the
   stroms saved in the folderroot text files for a certain area and saving the
   specified variables into a cvs file to be used in dataminer.py. This will
   run in parallel on machines with >4 cores.

   :copyright: © 2018 University of Leeds.
   :license: BSD-2 Clause.

Example:
    To use::
        import StormScriptsPy3 as SSP3
        c = SSP3.S_Box.StormInBox(x1, x2, y1, y2, size_of_storm, idstring, run='fc')
        c.gen_storm_box_csv()

.. CEMAC_stomtracking:
   https://github.com/cemac/StormTrackingScripts
"""


import glob
import copyreg
import warnings
import sys
import types
import gc
import iris
import pandas as pd
import numpy as np
import StormScriptsPy3.Pfuncts as Pfuncts

if not sys.warnoptions:
    warnings.simplefilter("ignore")

copyreg.pickle(types.MethodType, Pfuncts._pickle_method)


class StormInBox():
    '''Find the storms in specified box

    Takes the stroms saved in the folderroot text files for a certain area
    and saves the specified variables into a cvs file to be used in
    S_dataminer.py. This will run in parallel on machines with >4 cores.

    '''
    def __init__(self, x1, x2, y1, y2, size_of_storm, idstring, run='cc',
                 root=None, stormhome='./'):
        """Initialise with storm information

        Note:
            currently if you want to edit the underlying vrariables do this
            here e.g. the grid definition or variable list.

        Args:
            x1 (int): longitude of Western edge of box
            x2 (int): longitude of Eastern edge of box
            y1 (int): longitude of Southern edge of box
            y2 (int): latitude of Nothern edge of box
            size_of_storm (int): size of storm e.g. 5000
            idstring (:obj:`str`, optional): lable to identify these storms -
                this will lable the output.
            run (:obj:`str`, optional): a model run identifier such as cc or fc
                found in the file structure. Default is 'cc'
            root (:obj:`str`, optional): if using not fc or cc you must specify
                the root directory of data.
            stormhome (:obj:`str`, optional): where to store the output.
                Default is current directory './'
        """

        # Initialise variables
        self.size = size_of_storm
        self.x1, self.x2, self.y1, self.y2 = [x1, x2, y1, y2]
        # June to October
        self.m1, self.m2 = [6, 10]
        self.idstring = idstring
        # Variables to output
        self.varslist = ['stormid', 'year', 'month', 'day', 'hour', 'llon',
                         'ulon', 'llat', 'ulat', 'centlon', 'centlat', 'area',
                         'mean_olr']
        # Initialise grid variables (lat and lon)
        fname = ('/nfs/a277/IMPALA/data/4km/a03332_12km/a03332_A1hr_mean_' +
                 'ah261_4km_200012070030-200012072330.nc')
        cube = iris.load(fname)[1]
        self.lon = cube.coord('longitude').points
        self.lat = cube.coord('latitude').points
        # Variables in Julia's output
        self.cols = ['storm', 'no', 'area', 'centroid', 'box', 'life', 'u',
                     'v', 'mean', 'min', 'max', 'accreted', 'parent', 'child',
                     'cell']
        # Data root, will depend on run information
        # To start with we use cc or fc but you can specify your own via
        # class args
        if run == 'cc':
            root = '/nfs/a277/IMPALA/data/4km/'
            # File name contains date information at specific locations
            self.mstr1 = 90
            self.mstr2 = 92
            self.start_yr = '_1999'
        elif run == 'fc':
            root = '/nfs/a299/IMPALA/data/fc/4km/'
            # File name contains date information at specific locations
            self.mstr1 = 95
            self.mstr2 = 97
            self.start_yr = '_1997'
        else:
            if root is None:
                sys.exit('Please specify data root eg. /nfs/a299/IMPALA/data/fc/4km/')
        self.froot = root + 'precip_tracking_12km_hourly/'
        self.H = stormhome
        self.run = run

    def create_dataframe_all(self):
        """Create a daNote:

        Note:
            If no run in spefified the file pattern is unknow and all files
            will be listed not just certain months.

        Returns:
            DataFrame: list of files
        """
        # Check if this has already been done
        # If not iterate over files in file pattern
        try:
            df = pd.read_csv(self.H + self.run + 'filelist.csv', sep=',')
            return df
        except IOError:
            df2 = pd.DataFrame(columns=['file'])
            i = 0
            for rw in glob.iglob(self.froot+'*/a04203*4km*.txt'):
                    i += 1
                    df2.loc[i] = 0
                    modf2['file'].loc[i] = rw
            df = df2.reset_index(drop=True)
            df.to_csv(self.H + self.run + 'filelist.csv', sep=',')
        # find start year for later
        yrloc = df.file[1].find('hourly/')
        strtyr = df.file[1][yrloc+7:yrloc+11]
        self.start_yr = '_'+str(strtyr)
        # Storms we want have this infor
        print('generated file list...')
        return df

    def create_dataframe(self):
        """Create a data frame of file names

       Returns:
        df(DataFrame): A data frame of storm file names
        """
        # Check if this has already been done
        # If not iterate over files in file pattern
        try:
            df = pd.read_csv(self.H + self.run + 'filelist.csv', sep=',')
            return df
        except IOError:
            df2 = pd.DataFrame(columns=['file'])
            i = 0
            for rw in glob.iglob(self.froot+'*/a04203*4km*.txt'):
                if rw[self.mstr1:self.mstr2] in [str(x).zfill(2) for x in range(self.m1, self.m2)]:
                    i += 1
                    df2.loc[i] = 0
                    df2['file'].loc[i] = rw
            df = df2.reset_index(drop=True)
            df.to_csv(self.H + self.run + 'filelist.csv', sep=',')
        # Storms we want have this infor
        print('generated file list...')
        return df

    def find_the_storms(self, df):
        '''From the data in the files find the storms in box specified and
            above size specified.

        Arg:
            df: dataframe of file names

        Returns:
            dataframe: stormsdf a DataFrame of storms meeting criteria
        '''
        # p1 and p2 are place markers for the datestamp in filename
        p2 = self.p1 + 8
        # Create a dataframe wite column headers we want.
        stormsdf = pd.DataFrame(columns=self.varslist)
        # Iterate over every file
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
            datestamp = pd.to_datetime(cfile[self.p1:p2])
            stormsdf2.month = datestamp.month
            stormsdf2.day = datestamp.day
            stormsdf2.hour = datestamp.hour
            stormsdf2.year = datestamp.year
            stormsdf2.area = storms.area
            stormsdf2.mean_olr = storms['mean'].str[5::]
            stormsdf = pd.concat([stormsdf, stormsdf2]).reset_index(drop=True)
            gc.collect()  # Clear cache of unreference memeory
        return stormsdf

    def gen_storm_box_csv(self, altrun='N', nice=4, shared='Y'):
        '''generate storms csvs

        Args:
            altrun (str, optional): Default 'N', if 'Y' then use more liberal
                method to generate file list.
            nice (int, optional): niceness 1/nice share of machince. Default: 4
            shared (str, optional): 'Y' or 'N' if using shared resource.
                Default: 'Y'.

        Returns:
            csvfile: csvfile eg:
             fc_teststorms_over_box_area5000_lons_345_375_lat_10_18.csv
        '''
        # Check parallel settings and fair use.
        # DO NOT BE RUDE ON SHARED RESOURCES - PLAYING NICE IS ENFORCED HERE!
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
        # Call class method create_dataframe
        if altrun == 'N':
            df = self.create_dataframe()
        else:
            df = self.create_dataframe_all()
        # find the date string place holder
        self.p1 = df.file[1].find(self.start_yr) + 1
        # Call the parallel function that chuck up the dataframe.
        pstorms = Pfuncts.parallelize_dataframe(df, self.find_the_storms, nice)
        # Remove any duplicates
        pstormsdf = pstorms.drop_duplicates(subset='stormid', keep='first',
                                            inplace=False).reset_index(drop=True)
        # Write out csv wrt storage dir identifier and parameters.
        pstormsdf.to_csv(self.H + self.idstring + 'storms_over_box_area' +
                         str(self.size) + '_lons_' + str(self.x1) + '_' + str(self.x2) +
                         '_lat_' + str(self.y1) + '_' + str(self.y2)+'.csv')
