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
from tqdm import tqdm


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

    def gen_flist(self, storminfo, varcodes=None):
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
        df = pd.DataFrame(columns=['file', 'codes'],
                          index=[range(0, len(foldername))])
        for i in range(0, len(foldername)):
            try:
                df.loc[i] = [glob.glob(str(self.dataroot) + str(foldername[i])
                                       + '/' + str(foldername[i]) + '*_' +
                                       str(storminfo) + '*-*.nc'),
                             foldername[i]]
            except ValueError:
                pass
        return df

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
        # tqdm is a progress wrapper
        for row in tqdm(stormsdf.itertuples(), total=len(stormsdf),
                        unit="storm"):
            storminfo = (str(int(row.year)) + str(int(row.month)).zfill(2) +
                         str(int(row.day)).zfill(2))
            # If any files with structure exsit
            flist = self.gen_flist(storminfo, varcodes=self.vars['code'])
            if flist['file'].isnull().sum() > 0:
                continue
            # Initialise row
            allvars.loc[row[0]] = 0
            allvars['storms_to_keep'].loc[row[0]] = row.stormid
            allvars['OLRs'].loc[row[0]] = row.mean_olr
            xy = self.genslice(row.llon, row.llat, row.ulat, row.ulon)
            evemid = [11, 17]
            lvl = [0.5, 3]
            for rw in flist.itertuples():
                if rw.codes == ('c00409'):
                    allvari = iris.load_cube(rw.file, xy)
                    for num in evemid:
                        tvari = allvari[num, :, :]
                        tvarimean = tvari.collapsed(['latitude', 'longitude'],
                                                    iris.analysis.MEAN).data
                        if num == 11:
                            allvars['midday_mslp'].loc[row[0]] = tvarimean
                            continue
                        else:
                            allvars['eve_mslp_mean'].loc[row[0]] = tvarimean
                            tvari1p = tvari.collapsed(['latitude',
                                                       'longitude'],
                                                      iris.analysis.PERCENTILE,
                                                      percent=99).data
                            allvars['eve_mslp_1p'].loc[row[0]] = tvari1p
                elif rw.codes in ('c03225'):
                    u = iris.load_cube(rw.file, xy)
                elif rw.codes in ('c03226'):
                    v = iris.load_cube(rw.file, xy)
                    for num in evemid:
                        mwind = (iris.analysis.maths.exponentiate(u[num, :, :],
                                                                  2) +
                                 iris.analysis.maths.exponentiate(v[num, :, :],
                                                                  2))
                        for lex in lvl:
                            mwind2 = iris.analysis.maths.exponentiate(mwind,
                                                                      lex)
                            mwind1p = mwind2.collapsed(['latitude',
                                                        'longitude'],
                                                       iris.analysis.PERCENTILE,
                                                       percent=99)
                            mwind2 = mwind2.collapsed(['latitude',
                                                       'longitude'],
                                                      iris.analysis.MEAN).data
                            if lex == 0.5 and num == 11:
                                allvars['midday_wind'].loc[row[0]] = mwind2
                            elif num == 11 and lex == 3:
                                allvars['midday_wind3'].loc[row[0]] = mwind2
                            elif lex == 0.5 and num == 17:
                                allvars['eve_wind_mean'].loc[row[0]] = mwind1p
                                allvars['eve_wind_99p'].loc[row[0]] = mwind2
                            elif lex == 3 and num == 17:
                                allvars['eve_wind3_mean'].loc[row[0]] = mwind2
                                allvars['eve_wind3_99p'].loc[row[0]] = mwind1p
        allvars.to_csv('test.csv')
        return

    def calc_newvar(self, newvar_name):
        """
        """

        return newvar

    def genslice(self, llon, llat, ulat, ulon, n1=None, n2=None):
        """
        """
        if n1 is None and n2 is None:
            xysmallslice = iris.Constraint(longitude=lambda cell: float(llon)
                                           <= cell <= float(ulon),
                                           latitude=lambda cell: float(llat) <=
                                           cell <= float(ulat))
        elif n1 is not None and n2 is None:
            xysmallslice = iris.Constraint(pressure=lambda cell: n1 ==
                                           cell, longitude=lambda cell:
                                           float(llon) <= cell <= float(ulon),
                                           latitude=lambda cell: float(llat)
                                           <= cell <= float(ulat))
        elif n1 == 500 and n2 == 800:
            xysmallslice = iris.Constraint(pressure=lambda cell: n1 <= cell <=
                                           n2 or cell == 60,
                                           longitude=lambda cell:
                                           float(llon) <= cell <=
                                           float(ulon), latitude=lambda
                                           cell: float(llat) <= cell
                                           <= float(ulat))
        else:
            xysmallslice = iris.Constraint(pressure=lambda cell: n1 >=
                                           cell >= n2, longitude=lambda cell:
                                           float(llon) <= cell <= float(ulon),
                                           latitude=lambda cell: float(llat) <=
                                           cell <= float(ulat))
        return xysmallslice
