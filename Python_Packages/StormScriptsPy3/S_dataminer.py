# -*- coding: utf-8 -*-
"""pdataminer
This module was developed by CEMAC as part of the AMAMA 2050 Project.
This scripts build on Work done my Rory Fitzpatrick, taking the stroms from
stromsinabox csv files. This is the parallelized version.

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
import sys
import os
import copyreg
import warnings
import types
import pandas as pd
import numpy as np
import iris
import meteocalc
from skewt import SkewT as sk
from tqdm import tqdm
from StormScriptsPy3.Pfuncts import *
import StormScriptsPy3.Pfuncts as Pf

if not sys.warnoptions:
    warnings.simplefilter("ignore")
copyreg.pickle(types.MethodType, Pf._pickle_method)


class dm_functions():
    '''Description:
            A suite of functions to Calculate a standard set of variables
            for specified storms.
       Attributes:
        dataroot(str): Root folder of data
    '''
    def __init__(self, dataroot, CAPE=None, TEPHI=None):
        # Variables
        self.CAPE = CAPE
        self.TEPHI = TEPHI
        self.dataroot = dataroot
        self.varlist = ['year', 'month', 'day', 'hour', 'llon', 'ulon', 'llat',
                        'ulat', 'stormid', 'mean_olr']
        self.varlistex = ['stormid', 'year', 'month', 'day', 'hour', 'llon',
                          'ulon', 'llat', 'ulat', 'centlon', 'centlat', 'area',
                          'mean_olr']
        self.vars = pd.read_csv('data/stash_vars.csv')
        self.novars = len(self.vars)
        self.evemid = [11, 17]
        self.allvars = pd.read_csv('data/all_vars_template.csv')

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
        df = pd.DataFrame(columns=['file', 'codes', 'varname'])
        for i in range(self.novars):
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
        qf, q15f, Tf, t15f = fnamelist
        # ? 975 is a max value ?
        T = iris.load_cube(Tf)
        q = iris.load_cube(qf)
        mslp = self.allvars['eve_mslp_mean'].loc[idx]/100
        if mslp > 975:
            mslp = 975
        xy850 = genslice(latlons, n1=mslp, n2=100)
        T850 = T[3, :, :].extract(xy850)
        q850 = q[3, :, :].extract(xy850)
        Tp = T850.data
        Qp = q850.data
        # get rid of values sub 100?
        Tp[Tp < 100] = np.nan
        Tcol = np.zeros(Tp.shape[0])
        Qcol = np.zeros(Tp.shape[0])
        for p in range(0, Tp.shape[0]):
            Tcol[p] = np.nanmean(Tp[p, :, :])
            Qcol[p] = np.nanmean(Qp[p, :, :])
        Tcube = cubemean(T850)
        Qcube = cubemean(q850)
        Tcube.data = Tcol
        Qcube.data = Qcol
        pval = Tcube.data.shape[0] + 1
        f_T = T[3, : pval, 1, 1]
        f_q = q[3, : pval, 1, 1]
        f_T.data[:pval-1] = Tcol
        f_q.data[:pval-1] = Qcol
        t15 = cubemean(iris.load_cube(t15f)[11, :, :].extract(xy))
        P = self.allvars['eve_mslp_mean'].loc[idx]/100
        f_T.data[pval-1] = t15.data
        q15 = iris.load_cube(q15f)
        q15 = q15[11, :, :].extract(xy)
        f_q.data[pval-1] = cubemean(q15).data
        T = f_T.data
        hum = f_q.data
        Tkel = T - 273.16
        pressures = Tcube.coord('pressure').points
        pressures = np.append(pressures, mslp)
        P = pressures * 100
        height = np.zeros((len(pressures)))
        dwpt = np.zeros((len(pressures)))
        humity = np.zeros((len(pressures)))
        RH_650 = np.zeros((len(pressures)))
        if len(pressures) == 18:
            for p in range(0, len(pressures)):
                if 710. >= P[p] > 690.:
                    RH_650[p] = ([(0.263 * hum[p] * P[p]) /
                                  2.714**((17.67*(Tkel[p])) /
                                  (T[p] - 29.65))])

                humity[p] = ((0.263 * hum[p] * P[p]) /
                             2.714**((17.67*(Tkel[p]))/(T[p] - 29.65)))
                try:
                    dwpt[p] = meteocalc.dew_point(temperature=Tkel[p],
                                                  humidity=humity[p])
                except ValueError:
                    dwpt[p] = np.nan
                if p < len(pressures)-1:
                    height[p] = T[p]*((mslp*100/P[p])**(1./5.257) - 1)/0.0065
                else:
                    height[p] = 1.5
        xwind = cubemean(u[3, :, :])
        ywind = cubemean(v[3, :, :])
        self.allvars['Tephi_pressure'].loc[idx] = np.average(pressures, axis=0)
        self.allvars['Tephi_T'].loc[idx] = np.average(T, axis=0)
        self.allvars['Tephi_dewpT'].loc[idx] = np.average(dwpt, axis=0)
        self.allvars['Tephi_height'].loc[idx] = np.average(height, axis=0)
        self.allvars['Tephi_Q'].loc[idx] = np.average(humity, axis=0)
        self.allvars['Tephi_p99'].loc[idx] = precip99*3600.
        self.allvars['Tephi_xwind'].loc[idx] = xwind.data
        self.allvars['Tephi_ywind'].loc[idx] = ywind.data
        self.allvars['Tephi_RH650'].loc[idx] = np.average(RH_650, axis=0)
        mydata = dict(zip(('hght', 'pres', 'temp', 'dwpt'),
                          (height[::-1], pressures[::-1], T.data[::-1],
                           dwpt[:: -1])))
        S = sk.Sounding(soundingdata=mydata)
        parcel = S.get_parcel('mu')
        P_lcl, P_lfc, P_el, CAPE, CIN = S.get_cape(*parcel)
        self.allvars['CAPE_P_lcl'].loc[idx] = P_lcl
        self.allvars['CAPE_P_lfc'].loc[idx] = P_lfc
        self.allvars['CAPE_P_el'].loc[idx] = P_el
        self.allvars['CAPE_CAPE'].loc[idx] = CAPE
        self.allvars['CAPE_CIN'].loc[idx] = CIN

    def gen_vars(self, stormsdf):
        '''
        Generate variable csv files to show information about the storm
        csv root = file pattern for file to be Written
        '''
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
            precipm, precip99, precip = precips(flist, xy)
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
            u, v = vels(flist, xy)
            self.calc_winds(u, v, idx)
            uf = flist[flist.varname == 'u'].file
            vf = flist[flist.varname == 'v'].file
            self.calc_shear([uf, vf], xylw, xyhi, idx)
            of = flist[flist.varname == 'omega'].file
            Tf = flist[flist.varname == 'T'].file
            self.calc_tow(of, Tf, xy600, idx)
            olr_10p, olr_1p = olrs(flist, xy)
            self.allvars['OLR_10_perc'].loc[idx] = olr_10p
            self.allvars['OLR_1_perc'].loc[idx] = olr_1p
            varn99p, varmean = colws(flist, xy)
            self.allvars['col_w_mean'].loc[idx] = varmean
            self.allvars['col_w_p99'].loc[idx] = varn99p
            t15f = flist[flist.varname == 'T15'].file
            self.calc_t15(t15f, xy, idx)
            dryf = flist[flist.varname == 'dry_mass'].file
            wetf = flist[flist.varname == 'wet_mass'].file
            self.calc_mass(wetf, dryf, xy, idx)
            if self.CAPE == 'Y' or self.TEPHI == 'Y':
                q15f = flist[flist.varname == 'Q15'].file
                qf = flist[flist.varname == 'Q'].file
                fnamelist = [qf, q15f, Tf, t15f]
                self.calc_cape(fnamelist, u, v, precip99, xy, idx, latlons)
        return self.allvars

    def genvarscsv(self, csvroot, storms_to_keep, nice=4, shared='Y'):
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
            ans = yes_or_no(('***WARNING***: You are asking to use '
                             'half a shared computer \n consider fair '
                             'use of shared resources, do you wish to '
                             'continue? \n Y or N'))

            if not ans:
                print('Please revise nice number to higher value and try '
                      'again...')
                return
        df = storms_to_keep
        pstorms = parallelize_dataframe(df, self.gen_vars, nice)
        pstorms.to_csv(csvroot+'_standard.csv', sep=',')
