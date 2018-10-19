# -*- coding: utf-8 -*-
"""Dataminer functions
This module was developed by CEMAC as part of the AMAMA 2050 Project.
This scripts build on Work done my Rory Fitzpatrick, taking the stroms from
stromsinabox csv files

Example:
    To use::
        from dataminer_functions import dmfuctions
        dmf = dmfuctions(x1, x2, y1, y2, size_of_storm, dataroot[0])
        dmf.gen_var_csvs(csvroot[0], storms_to_keep)

Attributes:
    varslist(list): List of vairables required in dataminer
    fname(str): File to extract lat and lons
    froot(str): Root folder of data

.. CEMAC_stomtracking:
   https://github.com/cemac/StormTrackingScripts
"""

import numpy as np
import pandas as pd
import glob
from numpy import genfromtxt as gent
import iris
import os
from tqdm import tqdm


def cubemean(var):
    return var.collapsed(['latitude', 'longitude'], iris.analysis.MEAN)


def cube99(var, p=99):
    return var.collapsed(['latitude', 'longitude'], iris.analysis.PERCENTILE,
                         percent=p).data


class dmfuctions(object):
    '''Description
       Stage 1: currently a suit of functions for finding information on
       storms in region and Generating cvs files of that information.

       Attributes:
        dataroot(str): Root folder of data
    '''
    def __init__(self, dataroot):

        # Variables
        self.dataroot = dataroot
        self.varlist = ['year', 'month', 'day', 'hour', 'llon', 'ulon', 'llat',
                        'ulat', 'stormid', 'mean_olr']
        self.varlistex = ['stormid', 'year', 'month', 'day', 'hour', 'llon',
                          'ulon', 'llat', 'ulat', 'centlon', 'centlat', 'area',
                          'mean_olr']
        self.vars = pd.read_csv('vars.csv')

    def gen_storms_to_keep(self, altcsvname):
        """Generate storms to keep csvs

        Args:
            altcsvname (str): csv.

        Returns: storms to keep
        """
        dates = gent(altcsvname, delimiter=',', names=[self.varlistex])
        dates = np.sort(dates[:], axis=-1, order=['stormid', 'mean_olr'])
        storms_to_keep = np.zeros((1, 10), float)
        for line in dates:
            strm = line['stormid']
            goodup = 0
            for rw in range(0, storms_to_keep.shape[0]):
                if int(strm) == int(storms_to_keep[rw, 8]):
                    goodup += 1
                    continue
            if goodup < 1 and int(line['hour']) == 18:
                if np.sum(storms_to_keep[0, :]) == 0:
                    storms_to_keep[0, :] = line[self.varlist]
                else:
                    temp = np.zeros((1, 10), float)
                    temp[0, :] = line[self.varlist]
                    storms_to_keep = np.concatenate((storms_to_keep, temp),
                                                    axis=0)
        # wirte out a csv file?
        return storms_to_keep

    def gen_flist(self, storminfo, varcodes=None):
        """Generate flist

        Args:
            altcsvname(str): csv.
            dataroot(str): path to data
            storminfo(str): YYYYMMDD
            varcodes(dataframe): dataframe of var codes required

        Returns:
            df(dataframe): dataframe of files
        """
        if varcodes is not None:
            foldername = varcodes
        else:
            ds = pd.Series(os.listdir(self.dataroot))
            foldername = ds[ds.str.len() == 6].reset_index(drop=True)
        df = pd.DataFrame(columns=['file', 'codes'],
                          index=[range(0, len(foldername))])
        for i in range(0, len(foldername)):
            try:
                df.loc[i] = [glob.glob(str(self.dataroot) + str(foldername[i])
                                       + '/' + str(foldername[i]) + '*_' +
                                       str(storminfo) + '*-*.nc')[0],
                             foldername[i]]
            except IndexError:
                pass
        return df

    def meann99(var, str1200, strmean, str99p, p=99, idx):
        for num in self.evemid:
            varn = var[num, :, :]
            varmean = cubemean(varn).data
            if num == 11:
                allvars[str1200].loc[idx] = varmean
            else:
                varn99p = cube99(varn, p=1).data
                allvars[strmean].loc[idx] = varmean
                allvars[str99].loc[idx] = varn99p

    def calcT15(var, idx):
        self.meann99(var, 'mean_T15_1200', 'mean_T15_1800', '1perc_T15_1800',
                     p=1, idx)

    def calcmslp(fname, slice, idx):
            allvari = iris.load_cube(fname, slice)
            self.meann99(allvari, 'midday_mslp', 'eve_mslp_mean',
                         'eve_mslp_1p', idx)

    def calcwinds(u, v, idx):
        for num in evemid:
            mwind = (iris.analysis.maths.exponentiate(u[num, :, :], 2) +
                     iris.analysis.maths.exponentiate(v[num, :, :], 2))
            for lex in [0.5, 3]:
                mwind2 = iris.analysis.maths.exponentiate(mwind, lex)
                mwind1p = cube99(mwind2).data
                mwind2 = cubemean(mwind2).data
                if lex == 0.5 and num == 11:
                    allvars['midday_wind'].loc[idx] = mwind2
                elif num == 11 and lex == 3:
                    allvars['midday_wind3'].loc[idx] = mwind2
                elif lex == 0.5 and num == 17:
                    allvars['eve_wind_mean'].loc[idx] = mwind2
                    allvars['eve_wind_99p'].loc[idx] = mwind1p
                elif lex == 3 and num == 17:
                    allvars['eve_wind3_mean'].loc[idx] = mwind2
                    allvars['eve_wind3_99p'].loc[idx] = mwind1p

    def calcTOW(T, omega, idx):
        rgas = 287.058
        g = 9.80665
        for no in [3, 5]:
            Tnum = T[no, :, :, :]
            Onum = omega[no, :, :, :]
            Onum_1p = Onum.collapsed(['pressure', 'latitude', 'longitude'],
                                     iris.analysis.MIN).data
            Onum_1p = np.ndarray.tolist(Onum_1p)
            Onum_holdr = Onum.data
            ps = Onum.coord('pressure').points
            for p in range(0, Onum_holdr.shape[0]):
                for y in range(0, Onum_holdr.shape[1]):
                    for x in range(0, Onum_holdr.shape[2]):
                        if Onum_holdr[p, y, x] == Onum_1p:
                            T_min = Tnum[p, y, x].data
                            plev = p
            rho = (100.*ps[plev])/(rgas*T_min)
            wnum = -1*Onum_1p/(rho*g)
            Onumn = cubemean(Onum[plev, :, :]).data
            Tnum = cubemean(Tnum[plev, :, :])
            B1p = T_min - Tnum.data
            if no == 3:
                allvars['omega_1200_1p'].loc[idx] = Onum_1p
                allvars['omega_1200_mean'].loc[idx] = Onumn
                allvars['buoyancy_1200_1p'].loc[idx] = B1p
                allvars['max_w_1200'].loc[idx] = wnum
            else:
                allvars['omega_1800_1p'].loc[idx] = Onum_1p
                allvars['omega_1800_mean'].loc[idx] = Onumn
                allvars['buoyancy_1800_1p'].loc[idx] = B1p
                allvars['max_w_1800'].loc[idx] = wnum

    def calcshear(lowu, highu, idx):
        lowup = lowu.collapsed(['pressure'], iris.analysis.MAX)
        hiup = highu.collapsed(['pressure'], iris.analysis.MIN)
        mshear = lowup.data - hiup.data
        allvars['max_shear'].loc[idx] = mshear
        maxcheck = 0
        for p1 in lowu.coord('pressure').points:
            for p2 in highu.coord('pressure').points:
                lowslice = iris.Constraint(pressure=lambda cell: cell == p1)
                highslice = iris.Constraint(pressure=lambda cell: cell == p2)
                lowup2 = lowu.extract(lowslice).data
                lowvp2 = lowv.extract(lowslice).data
                highup2 = highu.extract(highslice).data
                highvp2 = highv.extract(highslice).data
                shearval = ((lowup2 - highup2)**2 + (lowvp2 - highvp2)**2)**0.5
                if shearval > maxcheck:
                    maxcheck = shearval
        allvars['hor_shear'].loc[idx] = maxcheck

    def calcmass(wet, dry, idx):
        mass = wet.data + dry.data
        for num in evemid:
            mass = mass[num, :, :]
            massm = cubemean(mass).data
            if num == 11:
                allvars['mass_mean_1200'].loc[idx] = massm
            else:
                allvars['mass_mean_1800'].loc[idx] = massm

    def gen_var_csvs(self, csvroot, storms_to_keep):
        '''
        Generate variable csv files to show information about the storm
        csv root = file pattern for file to be Written
        '''
        stormsdf = pd.DataFrame(storms_to_keep, columns=self.varlist)
        vels = 0
        vels2 = 0
        ot = 0
        wd = 0
        self.evemid = [11, 17]
        # for each row create a small sclice using iris
        allvars = pd.read_csv('all_vars_template.csv')
        # tqdm is a progress wrapper
        for row in tqdm(stormsdf.itertuples(), total=len(stormsdf),
                        unit="storm"):
            storminfo = (str(int(row.year)) + str(int(row.month)).zfill(2) +
                         str(int(row.day)).zfill(2))
            idx = row[0]
            # If any files with structure exsit
            flist = self.gen_flist(storminfo, varcodes=self.vars['code'])
            if flist['file'].isnull().sum() > 0:
                print('WARNING: Storm missing data files, skipping...')
                continue
            xy = genslice(row.llon, row.llat, row.ulat, row.ulon)
            xyhi = genslice(row.llon, row.llat, row.ulat, row.ulon,
                            n1=500, n2=800)
            xylw = genslice(row.llon, row.llat, row.ulat, row.ulon,
                            n1=925, n2=800)
            xy600 = genslice(row.llon, row.llat, row.ulat, row.ulon,
                             n1=600, n2=300)
            # Find if precip is meets criteria
            precipfile = flist[flist.codes == 'a04203'].file
            precip = iris.load_cube(precipfile).extract(xy)
            precipm = precip[11:15, :, :]
            precip = precip[17, :, :]
            precip = cube99(precip)
            precip = np.ndarray.tolist(precip)
            precipm = precipm.collapsed(['time', 'latitude', 'longitude'],
                                        iris.analysis.MEAN).data
            if precipm >= 0.1/3600. and precip <= 1.0/3600.:
                continue
            # Initialise row
            allvars.loc[idx] = 0
            allvars['storms_to_keep'].loc[idx] = row.stormid
            allvars['OLRs'].loc[idx] = row.mean_olr
            allvars['precip_99th_perc'].loc[idx] = precip
            pvol = (precipm * (float(row.ulon) - float(row.llon)) *
                              (float(row.ulat) - float(row.llat)))
            allvars['precip_accum'].loc[idx] = pvol
            allvars['area'].loc[idx] = pvol / precipm
            for rw in flist.itertuples():
                # we've already looked at precipitation so skip
                if rw.codes == 'a04203':
                    continue
                if rw.codes == ('c00409'):
                    self.calcmslp(rw.file, xy, idx)
                    continue
                elif rw.codes in 'c03225':
                    u = iris.load_cube(rw.file, xy)
                    vels += 1
                elif rw.codes in 'c03226':
                    v = iris.load_cube(rw.file, xy)
                    vels += 1
                if vels == 2:
                    vels = 0
                    self.calcwinds(u, v, idx)
                    continue
                var = iris.load_cube(rw.file)
                if rw.codes in 'f30201':
                    highu = var.extract(xyhi)[3, :, :, :]
                    highu = cubemean(highu)
                    lowu = var.extract(xylw)[3, :, :, :]
                    lowu = cubemean(lowu)
                    vels2 += 1
                    continue
                if rw.codes in ('f30202'):
                    highv = var.extract(xyhi)[3, :, :, :]
                    highv = cubemean(highv)
                    lowv = var.extract(xylw)[3, :, :, :]
                    lowv = cubemean(lowv)
                    vels2 += 1
                    continue
                if vels2 == 2:
                    vels2 = 0
                    self.calcshear(lowu, highu, idx)
                    continue
                if rw.codes == 'f30208':
                    omega = var.extract(xy600)
                    ot += 1
                    continue
                if rw.codes in 'f30204':
                    T = var.extract(xy600)
                    ot += 1
                    continue
                if ot == 2:
                    ot = 0
                    self.calcTOW(T, omega, idx)
                    continue
                var = var.extract(xy)
                if rw.codes == 'a03332':
                    OLR = var[17, :, :]
                    OLR_10p = cube99(OLR, p=10).data
                    OLR_1p = cube99(OLR, p=1).data
                    allvars['OLR_10_perc'].loc[idx] = OLR_10p
                    allvars['OLR_1_perc'].loc[idx] = OLR_1p
                    continue
                if rw.codes == 'a30439':
                    varn = var[5, :, :, :]
                    varmean = cubemean(varn).data
                    varn99p = cube99(varn).data
                    allvars['col_w_mean'].loc[idx] = varmean
                    allvars['col_w_p99'].loc[idx] = varn99p
                    continue
                if rw.codes == 'c03236':
                    self.calcT15(var)
                    continue
                if rw.codes == 'c30403':
                    wd += 1
                    wet = var
                    continue
                if rw.codes == 'c30404':
                    wd += 1
                    dry = var
                    continue
                if wd == 2:
                    wd = 0
                    self.calcmass(wet, dry, idx)
                    continue
            break
        allvars.to_csv(csvroot+'test4.csv')
        return


def genslice(llon, llat, ulat, ulon, n1=None, n2=None):
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
