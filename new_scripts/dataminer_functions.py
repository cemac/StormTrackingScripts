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

import glob
import os
import iris
import numpy as np
from numpy import genfromtxt as gent
import pandas as pd
from tqdm import tqdm


# Stand alone methods
def cubemean(var):
    '''Description:
        Find the mean of an iris cube variable
       Attributes:
        var: iris cube variable
    '''
    return var.collapsed(['latitude', 'longitude'], iris.analysis.MEAN)


def cube99(var, per=99):
    '''Description:
        Find the Nth PERCENTILE of an iris cube variable
       Attributes:
        var: iris cube variable
        p(int): PERCENTILE normally 1 or 99
    '''
    return var.collapsed(['latitude', 'longitude'], iris.analysis.PERCENTILE,
                         percent=per).data


class dm_functions(object):
    '''Description:
            A suite of functions to Calculate a standard set of variables
            for specified storms.
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
        self.evemid = [11, 17]
        self.allvars = pd.read_csv('all_vars_template.csv')

    def gen_storms_to_keep(self, altcsvname):
        """Generate storms to keep csvs
        Attributes:
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
        """Generate filelist
        Attributes:
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

    def mean99(self, var, str1200, strmean, str99p, idx, p=99):
        '''Description:
            Find the mean for the midday slice and the Nth PERCENTILE of an
            iris and mean of cube variable for the eveing slice

            Attributes:
                var: iris cube variable
                str1200(str): variable name str for midday mean
                strmean(str): variable name str for evening mean
                str99(str): variable name str for evening 99th PERCENTILE
                idx(int): row index
                p(int): PERCENTILE normally 1 or 99
        '''
        for num in self.evemid:
            varn = var[num, :, :]
            varmean = cubemean(varn).data
            if num == 11:
                self.allvars[str1200].loc[idx] = varmean
            else:
                var99p = cube99(varn, per=p)
                self.allvars[strmean].loc[idx] = varmean
                self.allvars[str99p].loc[idx] = var99p

    def calc_t15(self, var, idx):
        '''Description: mean99 for T15 variable
            Attributes:
                var: iris cube variable
                idx: row index
        '''
        self.mean99(var, 'mean_T15_1200', 'mean_T15_1800', '1perc_T15_1800',
                    idx, p=1)

    def calc_mslp(self, fname, smslice, idx):
        '''Description:
        Find the mean of an iris cube variable
        Attributes:
            fname(str): file to load cube from
            smslice: cube slice
            idx(int): index
        '''
        allvari = iris.load_cube(fname, smslice)
        self.mean99(allvari, 'midday_mslp', 'eve_mslp_mean',
                    'eve_mslp_1p', idx)

    def calc_winds(self, u, v, idx):
        '''Description: Calculate winds for eveing and midday using u and v.
           Attributes:
                u, v: iris cube variables u and v
                idx: row index
    '''
        for num in self.evemid:
            mwind = (iris.analysis.maths.exponentiate(u[num, :, :], 2) +
                     iris.analysis.maths.exponentiate(v[num, :, :], 2))
            for lex in [0.5, 3]:
                mwind2 = iris.analysis.maths.exponentiate(mwind, lex)
                mwind1p = cube99(mwind2)
                mwind2 = cubemean(mwind2).data
                if lex == 0.5 and num == 11:
                    self.allvars['midday_wind'].loc[idx] = mwind2
                elif num == 11 and lex == 3:
                    self.allvars['midday_wind3'].loc[idx] = mwind2
                elif lex == 0.5 and num == 17:
                    self.allvars['eve_wind_mean'].loc[idx] = mwind2
                    self.allvars['eve_wind_99p'].loc[idx] = mwind1p
                elif lex == 3 and num == 17:
                    self.allvars['eve_wind3_mean'].loc[idx] = mwind2
                    self.allvars['eve_wind3_99p'].loc[idx] = mwind1p

    def calc_tow(self, T, omega, idx):
        '''Description: Calculate bouyancy, omega and max for eveing midday etc
            Attributes:
                T: iris cube variable T
                omega: iris cube variable omega
                idx(int): row index
        '''
        R_GAS = 287.058
        G = 9.80665
        for no in [3, 5]:
            t_num = T[no, :, :, :]
            o_num = omega[no, :, :, :]
            o_num_1p = o_num.collapsed(['pressure', 'latitude', 'longitude'],
                                       iris.analysis.MIN).data
            o_num_1p = np.ndarray.tolist(o_num_1p)
            o_num_holdr = o_num.data
            ps = o_num.coord('pressure').points
            for p in range(0, o_num_holdr.shape[0]):
                for y in range(0, o_num_holdr.shape[1]):
                    for x in range(0, o_num_holdr.shape[2]):
                        if o_num_holdr[p, y, x] == o_num_1p:
                            t_min = t_num[p, y, x].data
                            plev = p
            rho = (100.*ps[plev])/(R_GAS*t_min)
            wnum = -1*o_num_1p/(rho*G)
            o_num_n = cubemean(o_num[plev, :, :]).data
            t_num = cubemean(t_num[plev, :, :])
            b_1p = t_min - t_num.data
            if no == 3:
                self.allvars['omega_1200_1p'].loc[idx] = o_num_1p
                self.allvars['omega_1200_mean'].loc[idx] = o_num_n
                self.allvars['buoyancy_1200_1p'].loc[idx] = b_1p
                self.allvars['max_w_1200'].loc[idx] = wnum
            else:
                self.allvars['omega_1800_1p'].loc[idx] = o_num_1p
                self.allvars['omega_1800_mean'].loc[idx] = o_num_n
                self.allvars['buoyancy_1800_1p'].loc[idx] = b_1p
                self.allvars['max_w_1800'].loc[idx] = wnum

    def calc_shear(self, lowu, highu, lowv, highv, idx):
        '''Description: Find max horizontal shear
            Attributes:
                lowu, low v: iris cube variable low pressure velocities
                high u, hugh v: iris cube variable high pressure velocities
                idx(int): row index
        '''
        lowup = lowu.collapsed(['pressure'], iris.analysis.MAX)
        hiup = highu.collapsed(['pressure'], iris.analysis.MIN)
        mshear = lowup.data - hiup.data
        self.allvars['max_shear'].loc[idx] = mshear
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
        self.allvars['hor_shear'].loc[idx] = maxcheck

    def calc_mass(self, wet, dry, idx, smslice):
        '''Description:
            Calculate midday and eveing mass
            Attributes:
        wet: iris cube variable
        dry: iris cube variable
        idx: row index
        '''
        mass = wet
        mass.data = wet.data + dry.data
        mass.extract(smslice)
        for num in self.evemid:
            massn = mass[num, :, :]
            massm = cubemean(massn).data
            if num == 11:
                self.allvars['mass_mean_1200'].loc[idx] = massm
            else:
                self.allvars['mass_mean_1800'].loc[idx] = massm

    def gen_var_csvs(self, csvroot, storms_to_keep):
        '''Description: The meat of this module, from the storms listed find if
           they meet the criteria of no midday rain and 99p rainfall above 1mm.
           Then write out a dataframe of storm variabels that do meet this.
            Attributes:
                storms_to_keep: list of storms
                csvroot: file name to write to.

        '''
        stormsdf = pd.DataFrame(storms_to_keep, columns=self.varlist)
        VELS = 0
        VELS2 = 0
        OT = 0
        WD = 0
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
            self.allvars.loc[idx] = 0
            self.allvars['storms_to_keep'].loc[idx] = row.stormid
            self.allvars['OLRs'].loc[idx] = row.mean_olr
            self.allvars['precip_99th_perc'].loc[idx] = precip
            pvol = (precipm * (float(row.ulon) - float(row.llon)) *
                    (float(row.ulat) - float(row.llat)))
            self.allvars['precip_accum'].loc[idx] = pvol
            self.allvars['area'].loc[idx] = pvol / precipm
            for rw in flist.itertuples():
                # we've already looked at precipitation so skip
                if rw.codes == 'a04203':
                    continue
                if rw.codes == ('c00409'):
                    self.calc_mslp(rw.file, xy, idx)
                    continue
                elif rw.codes in 'c03225':
                    u = iris.load_cube(rw.file, xy)
                    VELS += 1
                elif rw.codes in 'c03226':
                    v = iris.load_cube(rw.file, xy)
                    VELS += 1
                if VELS == 2:
                    VELS = 0
                    self.calc_winds(u, v, idx)
                    continue
                var = iris.load_cube(rw.file)
                if rw.codes in 'f30201':
                    highu = var.extract(xyhi)[3, :, :, :]
                    highu = cubemean(highu)
                    lowu = var.extract(xylw)[3, :, :, :]
                    lowu = cubemean(lowu)
                    VELS2 += 1
                    continue
                if rw.codes in 'f30202':
                    highv = var.extract(xyhi)[3, :, :, :]
                    highv = cubemean(highv)
                    lowv = var.extract(xylw)[3, :, :, :]
                    lowv = cubemean(lowv)
                    VELS2 += 1
                    continue
                if VELS2 == 2:
                    VELS2 = 0
                    self.calc_shear(lowu, highu, lowv, highv, idx)
                    continue
                if rw.codes == 'f30208':
                    omega = var.extract(xy600)
                    OT += 1
                    continue
                if rw.codes in 'f30204':
                    T = var.extract(xy600)
                    OT += 1
                    continue
                if OT == 2:
                    OT = 0
                    self.calc_tow(T, omega, idx)
                    continue
                var = var.extract(xy)
                if rw.codes == 'a03332':
                    OLR = var[17, :, :]
                    olr_10p = cube99(OLR, per=10)
                    olr_1p = cube99(OLR, per=1)
                    self.allvars['OLR_10_perc'].loc[idx] = olr_10p
                    self.allvars['OLR_1_perc'].loc[idx] = olr_1p
                    continue
                if rw.codes == 'a30439':
                    varn = var[5, :, :, :]
                    varmean = cubemean(varn).data
                    varn99p = cube99(varn)
                    self.allvars['col_w_mean'].loc[idx] = varmean
                    self.allvars['col_w_p99'].loc[idx] = varn99p
                    continue
                if rw.codes == 'c03236':
                    self.calc_t15(var, idx)
                    continue
                if rw.codes == 'c30403':
                    WD += 1
                    wet = var
                    continue
                if rw.codes == 'c30404':
                    WD += 1
                    dry = var
                    continue
                if WD == 2:
                    WD = 0
                    self.calc_mass(wet, dry, idx, xy)
                    continue
            break
        self.allvars.to_csv(csvroot+'test4.csv')
        return


def genslice(llon, llat, ulat, ulon, n1=None, n2=None):
    '''Description:
        Extrac iris cube slices of a variable
       Attributes:
        llon: lower longitude
        llat: lower latitude
        ulat: upper latitude
        ulon: upper longitude
        n1: pressure low
        n2: pressure high
    '''
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
                                       n2 or cell == 60, longitude=lambda cell:
                                       float(llon) <= cell <= float(ulon),
                                       latitude=lambda cell: float(llat) <=
                                       cell <= float(ulat))
    else:
        xysmallslice = iris.Constraint(pressure=lambda cell: n1 >=
                                       cell >= n2, longitude=lambda cell:
                                       float(llon) <= cell <= float(ulon),
                                       latitude=lambda cell: float(llat) <=
                                       cell <= float(ulat))
    return xysmallslice
