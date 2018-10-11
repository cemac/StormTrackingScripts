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
import numpy as np
import iris


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
    def __init__(self, x1, x2, y1, y2, size_of_storm):

        self.varslist = ['stormid', 'year', 'month', 'day', 'hour', 'llon',
                         'ulon', 'llat', 'ulat', 'centlon', 'centlat', 'area',
                         'mean_olr']
        stormsdf = pd.DataFrame(columns=self.varlist)
        fname = ('/nfs/a277/IMPALA/data/4km/a03332_12km/a03332_A1hr_mean_' +
                 'ah261_4km_200012070030-200012072330.nc')
        cube = iris.load(fname)[1]
        lon = cube.coord('longitude').points.tolist()
        lat = cube.coord('latitude').points.tolist()
        froot = '/nfs/a277/IMPALA/data/4km/precip_tracking_12km_hourly/'
        df = pd.DataFrame()
        df['file'] = (glob.glob(froot+'*/a04203*4km*.txt'))
        df2 = pd.DataFrame(columns=['file'], index=[range(0, len(df))])
        # The data frame lists every file but we only want june to october
        for row in df.itertuples():
            if row.file[90:92] in [str(x).zfill(2) for x in range(6, 10)]:
                df2.loc[row[0]] = 0
                df2['file'].loc[row[0]] = row.file
        df = df2.reset_index(drop=True)
        # Storms we want have this infor
        cols = ['storm', 'no', 'area', 'centroid', 'box', 'life', 'u', 'v',
                'mean', 'min', 'max', 'accreted', 'parent', 'child', 'cell']
        for row in df.itertuples():
            vars = pd.read_csv(row.file, names=cols,  header=None,
                               delim_whitespace=True)
            # the txt files have stoms and then child cells with surplus info
            var2 = vars[pd.notnull(vars['mean'])]
            size = var2.area.str[5::]
            var2['area'] = pd.to_numeric(size)*144
            # If it meets our size criteria
            storms = var2[var2.area >= size_of_storm].reset_index(drop=True)
            # And is the centroid in our location
            storms['centroid']
            # lets create a data frame of the varslist components
            # join DataFrame to stormsdf and move on to next file.
            # Make a dataframe to fill this time steps storm data
            stormsdf2 = pd.DataFrame(columns=self.varlist)
            stormsdf2.stormid = storms.no
            datestamp = pd.to_datetime(row.file[86:98])
            stormsdf2.year = datestamp.year
            # Append to whole area
            stormsdf = pd.concat([stormsdf, stormsdf2]).reset_index(drop=True)
        stormsdf.to_csv(idstring + 'storms_over_box_area' + str(size_of_storm)
                        + '_lons_' + str(x1) + '_' + str(x2) + '_lat_' +
                        str(y1) + '_' + str(y2)+'.csv')
