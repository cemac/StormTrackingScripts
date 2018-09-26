""" Python module for working Storm tracking CSV files
    STAGE = 1
    * Stage 1 development create functions for data mining
      reduce code size
    * Stage 2 remove hard coding
    * Stage 3 improve effciency
    * Stage 4 integrate
"""


from numpy import genfromtxt as gent
import numpy as np
import pandas as pd
import glob
import iris
import os

class dmfuctions:
    '''Description
       Stage 1: currently a suit of functions for finding information on
       storms in region and Generating cvs files of that information.
    '''

# Global variables

    def __init__(self, x1, x2, y1, y2, storm_size):

        # Variables
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.dataroot = '/nfs/a299/IMPALA/data/fc/4km/'
        self.storm_size = storm_size
        self.varlist = ['year', 'month', 'day', 'hour', 'llon', 'ulon', 'llat',
                        'ulat', 'stormid', 'mean_olr']
        self.vars = pd.read_csv('vars.csv')

    def gen_storms_to_keep(self, altcsvname):
        dates = gent(altcsvname, delimiter=',', names=['stormid', 'year',
                                                       'month',
                                                       'day', 'hour', 'llon',
                                                       'ulon', 'llat', 'ulat',
                                                       'centlon', 'centlat',
                                                       'area', 'mean_olr'])
        dates = np.sort(dates[:], axis=-1, order=['stormid', 'mean_olr'])
        storms_to_keep = np.zeros((1, 10), float)
        # You need to bear in mind that this code wants to track the point when
        # the storm is at minimum OLR.
        for line in dates:
            strm = line['stormid']
            goodup = 0
        for rw in range(0, storms_to_keep.shape[0]):
            if int(strm) == int(storms_to_keep[rw, 8]):
                goodup = goodup + 1
                continue
        if goodup < 1 and 18 == int(line['hour']):
            if np.sum(storms_to_keep[0, :]) == 0:
                storms_to_keep[0, :] = line['year', 'month', 'day', 'hour',
                                            'llon', 'ulon', 'llat', 'ulat',
                                            'stormid', 'mean_olr']
            else:
                temp = np.zeros((1, 10), float)
                temp[0, :] = line['year', 'month', 'day', 'hour', 'llon',
                                  'ulon', 'llat', 'ulat', 'stormid',
                                  'mean_olr']
                storms_to_keep = np.concatenate((storms_to_keep, temp), axis=0)
        return storms_to_keep

    def gen_flist(self, dataroot, storminfo, varcodes=None):
        '''
        Generate flist - search for files pertaining to that storm
        Args:
        dataroot: path to data
        storminfo: YYYYMMDD
        varcodes: dataframe of var codes required
        Returns: list of files contain variables for that storm
        Pandas dataframe of files
        '''
        if varcodes is not None:
            foldername = varcodes
        else:
            ds = pd.Series(os.listdir(dataroot))
            foldername = ds[ds.str.len() == 6].reset_index(drop=True)
        df = pd.DataFrame(columns=['codes'], index=[range(0, len(foldername))])
        for i in range(0, len(foldername)):
            try:
                df.loc[i] = glob.glob(str(dataroot) + str(foldername[i]) + '/' +
                                      str(foldername[i]) + '*' + str(storminfo) + '*.nc')
            except ValueError:
                pass
        flist = df.dropna()
        return flist

    def gen_var_csvs(self, csvroot, storms_to_keep):
        '''
        Generate variable csv files to show information about the storm
        csv root = file pattern for file to be Written
        '''
        stormsdf = pd.DataFrame(storms_to_keep, columns=self.varlist)
        # Variables
        varlist = pd.read_csv('varlist.csv', header=None)
        OLRmin = 300.0
        ukeep925 = 0.0
        ukeep650 = 0.0
        ukeepsheer = 0.0
        varcodes = self.vars['code']
        # for each row create a small sclice using iris
        allvars = pd.read_csv('all_vars_template.csv')
        for row in stormsdf.itertuples():
            '''
            xysmallslice = iris.Constraint(longitude=lambda cell: row.llon
                                           <= cell <= row.ulon, latitude=lambda
                                           cell: row.llat <= cell <= row.ulat)
            xysmallslice_925 = iris.Constraint(pressure=lambda cell: 925 ==
                                               cell, longitude=lambda cell:
                                               row.llon <= cell <= row.ulon,
                                               latitude=lambda cell: row.llat
                                               <= cell <= row.ulat)
            xysmallslice_650 = iris.Constraint(pressure=lambda cell: 650 ==
                                               cell, longitude=lambda cell:
                                               row.llon <= cell <= row.ulon,
                                               latitude=lambda cell: row.llat
                                               <= cell <= row.ulat)
            xysmallslice_850 = iris.Constraint(pressure=lambda cell: 850 ==
                                               cell, longitude=lambda cell:
                                               row.llon <= cell <= row.ulon,
                                               latitude=lambda cell: row.llat
                                               <= cell <= row.ulat)
            xysmallslice_500 = iris.Constraint(pressure=lambda cell: 500 ==
                                               cell, longitude=lambda cell:
                                               row.llon <= cell <= row.ulon,
                                               latitude=lambda cell: row.llat
                                               <= cell <= row.ulat)
            xysmallslice_600plus = iris.Constraint(pressure=lambda cell: 600 >=
                                                   cell >= 300,
                                                   longitude=lambda cell:
                                                   row.llon <= cell <=
                                                   row.ulon, latitude=lambda
                                                   cell: row.llat <= cell
                                                   <= row.ulat)
            xysmallslice_800_925 = iris.Constraint(pressure=lambda cell:
                                                   925 >= cell >= 800,
                                                   longitude=lambda cell:
                                                   row.llon <= cell <=
                                                   row.ulon, latitude=lambda
                                                   cell: row.llat <= cell <=
                                                   row.ulat)
            xysmallslice_500_800 = iris.Constraint(pressure=lambda cell: 500
                                                   <= cell <= 800 or cell ==
                                                   60, longitude=lambda cell:
                                                   row.llon <= cell <=
                                                   row.ulon, latitude=lambda
                                                   cell: row.llat <= cell
                                                   <= row.ulat)
            '''
            storminfo = (str(int(row.year)) + str(int(row.month)).zfill(2) +
                         str(int(row.day)).zfill(2))
            # If any files with structure exsit
            flist = self.gen_flist(self, storminfo, varcodes=self.vars['code'])
            # initialise row
            allvars.loc[row[0]] = 0
            allvars['storms_to_keep'].loc[row[0]] = row.stormid
            allvars['OLRs'].loc[row[0]] = row.mean_olr
        allvars.to_csv('test.csv')
        return

    def calc_newvar(self, newvar_name):
        """
        """

        return newvar
