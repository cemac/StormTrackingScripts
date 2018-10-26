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
import meteocalc
from skewt import SkewT as sk


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
        vari = ['pressure', 'T' 'dewpT', 'height', 'Q', 'z', 'RH650']
        self.tephidf = pd.DataFrame(columns=vari)
        vari = ['P_lcl', 'P_lfc', 'P_el', 'CAPE', 'CIN']
        self.capedf = pd.DataFrame(columns=vari)

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

    def gen_flist(self, storminfo, varnames, varcodes=None):
        """Generate filelist
        Attributes:
            storminfo(str): YYYYMMDD
            varnames(DataFrame):  dataframe of corresponding varnames
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
                             foldername[i], varnames[i]]
            except IndexError:
                pass
        return df

    def mean99(self, var, strings, idx, p=99):
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
        str1200, strmean, str99p = strings
        for num in self.evemid:
            varn = var[num, :, :]
            varmean = cubemean(varn).data
            if num == 11:
                self.allvars[str1200].loc[idx] = varmean
            else:
                var99p = cube99(varn, per=p)
                self.allvars[strmean].loc[idx] = varmean
                self.allvars[str99p].loc[idx] = var99p

    def calc_t15(self, t15f, xy, idx):
        '''Description: mean99 for T15 variable
            Attributes:
                var: iris cube variable
                idx: row index
        '''
        t15 = iris.load_cube(t15f).extract(xy)
        strings = ['mean_T15_1200', 'mean_T15_1800', '1perc_T15_1800']
        self.mean99(t15, strings, idx, p=1)

    def calc_mslp(self, fname, xy, idx):
        '''Description:
        Find the mean of an iris cube variable
        Attributes:
            fname(str): file to load cube from
            idx(int): index
        '''
        allvari = iris.load_cube(fname, xy)
        strings = ['midday_mslp', 'eve_mslp_mean', 'eve_mslp_1p']
        self.mean99(allvari, strings, idx)

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

    def calc_tow(self, of, Tf, xy600, idx):
        '''Description: Calculate bouyancy, omega and max for eveing midday etc
            Attributes:
                T: iris cube variable T
                omega: iris cube variable omega
                idx(int): row index
        '''
        R_GAS = 287.058
        G = 9.80665
        omega = iris.load_cube(of).extract(xy600)
        T = iris.load_cube(Tf).extract(xy600)
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

    def calc_shear(self, filenames, xylw, xyhi, idx):
        '''Description: Find max horizontal shear
            Attributes:
                lowu, low v: iris cube variable low pressure velocities
                high u, hugh v: iris cube variable high pressure velocities
                idx(int): row index
        '''
        uf, vf = filenames
        u10 = iris.load_cube(uf)
        v10 = iris.load_cube(vf)
        highu = cubemean(u10.extract(xyhi)[3, :, :, :])
        lowu = cubemean(u10.extract(xylw)[3, :, :, :])
        highv = cubemean(v10.extract(xyhi)[3, :, :, :])
        lowv = cubemean(v10.extract(xylw)[3, :, :, :])
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

    def calc_mass(self, wetf, dryf, xy, idx):
        '''Description:
            Calculate midday and eveing mass
            Attributes:
        wet: iris cube variable
        dry: iris cube variable
        idx: row index
        '''
        wet = iris.load_cube(wetf).extract(xy)
        dry = iris.load_cube(dryf).extract(xy)
        mass = wet
        mass.data = wet.data + dry.data
        mass.extract(xy)
        for num in self.evemid:
            massn = mass[num, :, :]
            massm = cubemean(massn).data
            if num == 11:
                self.allvars['mass_mean_1200'].loc[idx] = massm
            else:
                self.allvars['mass_mean_1800'].loc[idx] = massm

    def calc_cape(self, fnamelist, u, v, precip99, xy, idx, latlons):
        r'''Description:
            Variable required to calculate Convective Available Potential
            Energy: $\int_{zf}^{zn} g
            \left( \frac{T_{v, parcel} - T_{v,env}}{T_{v,env}}\right)
            \mathrm{d}z$
            If > 0 storms possible if enough moisture
            Attributes:
                idx: row index
            Returns:
            CAPE(dataframe): dataframe containing Vars for cape calculations
            TEPHI(dataframe): dataframe containg vars for tephigrams
        '''
        Qf, Q15f, Tf, T15f = fnamelist
        q15 = iris.load_cube(Q15f)
        q15 = q15[11, :, :].extract(xy)
        T15 = cubemean(iris.load_cube(T15f)[11, :, :].extract(xy))
        xwind = cubemean(u[3, :, :])
        ywind = cubemean(v[3, :, :])
        P = self.allvars['eve_mslp_mean'].loc[idx]/100
        # ? 975 is a max value ?
        T = iris.load_cube(Tf)
        q = iris.load_cube(Qf)
        if P > 975:
            P = 975
        xy850 = genslice(latlons, n1=P, n2=100)
        T = T[3, :, :].extract(xy850)
        q = q[3, :, :].extract(xy850)
        Tp = T.data
        Qp = q.data
        # get rid of values sub 100?
        Tp[Tp < 100] = np.nan
        Qp[Qp < 100] = np.nan
        Tcol = np.zeros(Tp.shape[0])
        Qcol = np.zeros(Tp.shape[0])
        for p in range(0, Tp.shape[0]):
            Tcol[p] = np.nanmean(Tp[p, :, :])
            Qcol[p] = np.nanmean(Qp[p, :, :])
        Tcol = cubemean(Tcol)
        Qcol = cubemean(Qcol)
        T.data = Tcol
        q.data = Qcol
        pressure = T.coord('pressure').points
        pressure = pressure.extend([P])
        pval = T.data.shape[0] + 1
        f_T = T[3, : pval, 1, 1]
        f_q = q[3, : pval, 1, 1]
        f_T.data[:pval] = Tcol
        f_T.data[pval] = T15.data
        f_q.data[:pval] = Qcol
        f_q.data[pval] = cubemean(q15).data
        t = f_T
        pres = f_q
        f_q.data = pressure*100
        pressures = f_q.data
        height = np.zeros((len(pressures)))
        dwpt = np.zeros((len(pressures)))
        if len(pressure == 18):
            for p in range(0, len(pressures)):
                if 710. >= pressures[p] > 690.:
                    RH_650 = ([(0.263 * pres.data[p] * pressures.data[p])
                               / 2.714**((17.67*(t.data[p] - 273.16)) /
                                         (t.data[p] - 29.65))])
                T = (0.263 * pres.data[p] *
                     pressures.data[p])/2.714**((17.67*(t.data[p] -
                                                        273.16))/(t.data[p]
                                                                  - 29.65))
                dwpt[p] = meteocalc.dew_point(temperature=t[p].data - 273.16,
                                              humidity=T)
                if p < len(pressures)-1:
                    height[p] = (t[p].data*((100*P /
                                             pressures.data[p])**(1./5.257)
                                            - 1)
                                 / 0.0065)
                else:
                    height[p] = 1.5

        self.tephidf['pressure'].loc[idx] = np.average(pressure, axis=0)
        self.tephidf['T'].loc[idx] = np.average(T, axis=0)
        self.tephidf['dewpT'].loc[idx] = np.average(dwpt, axis=0)
        self.tephidf['height'].loc[idx] = np.average(height, axis=0)
        self.tephidf['Q'].loc[idx] = np.average(q, axis=0)
        self.tephidf['p99'].loc[idx] = np.average(precip99, axis=0)
        self.tephidf['xwind'].loc[idx] = np.average(xwind, axis=0)
        self.tephidf['ywind'].loc[idx] = np.average(ywind, axis=0)
        self.tephidf['RH650'].loc[idx] = RH_650
        mydata = dict(zip(('hght', 'pres', 'temp', 'dwpt'),
                          (height, pressure, T, dwpt)))
        S = sk.Sounding(soundingdata=mydata)
        parcel = S.get_parcel('mu')
        self.capedf.loc[idx] = S.get_cape(*parcel)

    def gen_var_csvs(self, csvroot, storms_to_keep, CAPE=None, TEPHI=None):
        '''Description: The meat of this module, from the storms listed find if
           they meet the criteria of no midday rain and 99p rainfall above 1mm.
           Then write out a dataframe of storm variabels that do meet this.
            Attributes:
                storms_to_keep: list of storms
                csvroot: file name to write to.

        '''
        stormsdf = pd.DataFrame(storms_to_keep, columns=self.varlist)
        # tqdm is a progress wrapper
        for row in tqdm(stormsdf.itertuples(), total=len(stormsdf),
                        unit="storm"):
            storminfo = (str(int(row.year)) + str(int(row.month)).zfill(2) +
                         str(int(row.day)).zfill(2))
            idx = row[0]
            # If any files with structure exsit
            flist = self.gen_flist(storminfo, self.vars['varname'],
                                   varcodes=self.vars['code'])
            if flist['file'].isnull().sum() > 0:
                print('WARNING: Storm missing data files, skipping...')
                continue
            latlons = [row.llon, row.llat, row.ulat, row.ulon]
            xy = genslice(latlons)
            xyhi = genslice(latlons, n1=500, n2=800)
            xylw = genslice(latlons, n1=925, n2=800)
            xy600 = genslice(latlons, n1=600, n2=300)
            # Find if precip is meets criteria
            precipfile = flist[flist.varname == 'precip'].file
            precip = iris.load_cube(precipfile).extract(xy)
            precipm = precip[11:15, :, :]
            precip = precip[17, :, :]
            precip99 = cube99(precip)
            precip = np.ndarray.tolist(precip99)
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
            self.calc_mslp(flist[flist.varname == 'ms1p'].file, xy, idx)
            uf = flist[flist.varname == 'u10'].file
            u = iris.load_cube(uf, xy)
            vf = flist[flist.varname == 'v10'].file
            v = iris.load_cube(vf, xy)
            self.calc_winds(u, v, idx)
            uf = flist[flist.varname == 'u'].file
            uf = flist[flist.varname == 'v'].file
            self.calc_shear([uf, vf], xylw, xyhi, idx)
            of = flist[flist.varname == 'omega'].file
            Tf = flist[flist.varname == 'T'].file
            self.calc_tow(of, Tf, xy600, idx)
            olr_f = flist[flist.varname == 'olr'].file
            OLR = iris.load_cube(olr_f).extract(xy)
            OLR = OLR[17, :, :]
            olr_10p = cube99(OLR, per=10)
            olr_1p = cube99(OLR, per=1)
            self.allvars['OLR_10_perc'].loc[idx] = olr_10p
            self.allvars['OLR_1_perc'].loc[idx] = olr_1p
            colwf = flist[flist.varname == 'col_w'].file
            colw = iris.load_cube(colwf).extract(xy)
            varn = colw[5, :, :, :]
            varmean = cubemean(varn).data
            varn99p = cube99(varn)
            self.allvars['col_w_mean'].loc[idx] = varmean
            self.allvars['col_w_p99'].loc[idx] = varn99p
            t15f = flist[flist.varname == 'T15'].file
            self.calc_t15(t15f, xy, idx)
            dryf = flist[flist.varname == 'dry_mass'].file
            wetf = flist[flist.varname == 'wet_mass'].file
            self.calc_mass(wetf, dryf, xy, idx)
            if CAPE == 'Y' or TEPHI == 'Y':
                q15f = flist[flist.varname == 'Q15'].file
                qf = flist[flist.varname == 'Q'].file
                fnamelist = [qf, q15f, Tf, t15f]
                self.calc_cape(fnamelist, u, v, precip99, xy, idx, latlons)
            break
        self.allvars.to_csv(csvroot+'_standard.csv')
        if CAPE == 'Y':
            self.capedf.to_csv(csvroot+'_cape.csv')
        if TEPHI == 'Y':
            self.tephidf.to_csv(csvroot+'_tephi.csv')
        return


def genslice(latlons, n1=None, n2=None):
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
    llon, llat, ulat, ulon = latlons
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
