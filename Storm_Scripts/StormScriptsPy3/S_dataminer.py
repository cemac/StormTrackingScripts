# -*- coding: utf-8 -*-
"""Data miner

.. module:: S_dataminer
    :platform: Unix
    :synopis:

.. moduleauthor: CEMAC (UoL)

.. description: This module was developed by CEMAC as part of the AMAMA 2050
   Project. This script builds on Work done my Rory Fitzpatrick. From the csv
   generated by S_Box check find storms of certain criteria and collect data.
   This will run in parallel on machines with >4 cores.

   :copyright: © 2018 University of Leeds.
   :license: BSD-2 Clause.

Example:
    To use::
        import StormScriptsPy3 as SSP3
        dmf = SSP3.dm_functions(dataroot, CAPE='Y', TEPHI='Y')
        storms_to_keep = pd.read_csv('fc_teststorms_over_box_area5000_lons_'+
                                     '345_375_lat_10_18.csv', sep=',')
        dmf.genvarscsv('fc_test', storms_to_keep)

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
from StormScriptsPy3.Pfuncts import *
import StormScriptsPy3.Pfuncts as Pf

# Over ride pickle GIL (Allows the parallelization)
if not sys.warnoptions:
    warnings.simplefilter("ignore")
copyreg.pickle(types.MethodType, Pf._pickle_method)


class dm_functions():
    '''Data miner suite of functions.

    Constists of 3 generation functions and 8 calculation functions
    members:
        gen_flist: generates file lists of variables
        gen_vars: mines the data required for all the storms
        genvarscsv: calls gen_vars in parallel mannor
        mean99: some vaibles simply want just the 99th percentile
        calc_t15: air_temperature
        calc_mslp: mean sea level pressure variables
        calc_winds: wind variables
        calc_tow: Temp, Omega, col_w. tendency_of_air_pressure, bouyancy and col_w
        calc_shear: Find max horizontal shear
        calc_mass: wet and dry mass
        calc_cape: CAPE variables + Tephi

    Note:
    This has been done if more variables are required they could be inserted by
    additional calc methods called in gen vars.
    '''
    def __init__(self, dataroot, CAPE=None, TEPHI=None):
        """Initialise with storm information

        Note:
            currently, if you want to edit the underlying variables do this
            here e.g. the grid definition or variable list.

        Args:
            dataroot (str): file path to data dir e.g.
                '/nfs/a299/IMPALA/data/fc/4km/'
            CAPE (:obj:`str`, optional): Include CAPE variables e.g. 'Y'.
                Defaults to None
            TEPHI (:obj:`str`, optional): Include TEPHI variables e.g. 'Y'.
                Defaults to None
        """

        # Variables
        self.CAPE = CAPE
        self.TEPHI = TEPHI
        self.dataroot = dataroot
        # Varibales from configuration file.
        self.vars = pd.read_csv('data/stash_vars.csv')
        self.varnos = len(self.vars)
        # For efficiency len is not called in a for loop
        self.novars = len(self.vars)
        # 1200 and 1800 time variables
        self.evemid = [11, 17]
        # Column headers. Messy to list.
        self.allvars = pd.read_csv('data/all_vars_template.csv')
        self.mcdp = np.vectorize(meteocalc.dew_point)

    def dataise(self, var): return var.data

    def nodata(self, var): return var

    def gen_flist(self, storminfo, varnames, varcodes=None):
        """Generate filelist
        Args:
            storminfo (str): YYYYMMDD
            varnames (DataFrame):  dataframe of corresponding varnames
            varcodes (:obj:dataframe, optional): dataframe of var codes
                required. Defaults to None
        Returns:
            df (dataframe): dataframe of files
        """
        if varcodes is not None:
            foldername = varcodes
        else:
            ds = pd.Series(os.listdir(self.dataroot))
            foldername = ds[ds.str.len() == 6].reset_index(drop=True)
        df = pd.DataFrame(columns=['file', 'codes', 'varname'])
        for i, folder in enumerate(foldername):
            try:
                df.loc[i] = [glob.glob(str(self.dataroot) + str(folder)
                                       + '/' + str(folder) + '*_' +
                                       str(storminfo) + '*-*.nc')[0],
                             folder, varnames[i]]
            except IndexError:
                pass
        return df

    def gen_vars(self, stormsdf):
        '''Generate variable csv files to show information about the storm

        Iterates over a data frame checking each storm. Finds corresponding
        netcdf files and checks for missing data and checks for the criteria:
        rainfall_above_1mm_no_midday_rain. If that's all fine and dandy then
        populates a dataframe with an entry for that storm.

        Note:
            Calls all the calculation functions below
        Args:
            stormsdf (DataFrame): dataframe of storms list generated by S_Box
        Returns:
            allvars (DataFrame): dataframe of mined data
        '''
        # Interate over list of storms
        storminfo = 00000000
        for row in stormsdf.itertuples():
            # Get the datestamp make sure its YYYYMMDD
            storminfo1 = storminfo
            storminfo = (str(int(row.year)) + str(int(row.month)).zfill(2) +
                         str(int(row.day)).zfill(2))
            idx = row[0]  # Initialise
            # Generate corresponding netCDF list for storm, if new day!
            if storminfo != storminfo1:
                flist = self.gen_flist(storminfo, self.vars['varname'],
                                       varcodes=self.vars['code'])
                if len(flist) < self.varnos:
                    # reset storminfo else missing file won't get caught...
                    storminfo = 00000000
                    print('WARNING: Storm missing data files, skipping...')
                    continue
            # location of storm
            latlons = [row.llon, row.llat, row.ulat, row.ulon]
            # stand cube slice
            xy = genslice(latlons)
            # Find if precip is meets criteria
            precipm, precip99, precip = precips(flist, xy)
            precipm = precipm.collapsed(['time', 'latitude', 'longitude'],
                                        iris.analysis.MEAN).data
            # if rainfall below 1mm and midday_rain the skip
            if precipm >= 0.1/3600. and precip <= 1.0/3600.:
                continue
            # generate rest of slices
            xyhi = genslice(latlons, n1=500, n2=800)
            xylw = genslice(latlons, n1=925, n2=800)
            xy600 = genslice(latlons, n1=600, n2=300)
            # Initialise row
            self.allvars.loc[idx] = 0
            self.allvars['storms_to_keep'].loc[idx] = row.stormid
            self.allvars['OLRs'].loc[idx] = row.mean_olr
            self.allvars['precip_99th_perc'].loc[idx] = precip
            pvol = (precipm * (float(row.ulon) - float(row.llon)) *
                    (float(row.ulat) - float(row.llat)))
            self.allvars['precip_accum'].loc[idx] = pvol
            self.allvars['area'].loc[idx] = pvol / precipm
            # MSLP
            self.calc_mslp(flist[flist.varname == 'ms1p'].file, xy, idx)
            # WINDS
            u, v = vels(flist, xy)
            self.calc_winds(u, v, idx)
            uf = flist[flist.varname == 'u'].file
            vf = flist[flist.varname == 'v'].file
            # SHEAR
            self.calc_shear([uf, vf], xylw, xyhi, idx)
            of = flist[flist.varname == 'omega'].file
            Tf = flist[flist.varname == 'T'].file
            # bouyancy, temperature tendency_of_air_pressure and col_W
            self.calc_tow(of, Tf, xy600, idx)
            olr_10p, olr_1p = olrs(flist, xy)
            self.allvars['OLR_10_perc'].loc[idx] = olr_10p
            self.allvars['OLR_1_perc'].loc[idx] = olr_1p
            varn99p, varmean = colws(flist, xy)
            self.allvars['col_w_mean'].loc[idx] = varmean
            self.allvars['col_w_p99'].loc[idx] = varn99p
            # AIR TEMP
            t15f = flist[flist.varname == 'T15'].file
            self.calc_t15(t15f, xy, idx)
            # MASS
            dryf = flist[flist.varname == 'dry_mass'].file
            wetf = flist[flist.varname == 'wet_mass'].file
            self.calc_mass(wetf, dryf, xy, idx)
            # CAPE AND TEHPHI CALCS
            if self.CAPE == 'Y' or self.TEPHI == 'Y':
                q15f = flist[flist.varname == 'Q15'].file
                qf = flist[flist.varname == 'Q'].file
                fnamelist = [qf, q15f, Tf, t15f]
                self.calc_cape(fnamelist, u, v, precip99, xy, idx, latlons)
        return self.allvars

    def genvarscsv(self, csvroot, storms_to_keep, nice=4, shared='Y'):
        '''generate storms csvs

        Args:
            csvroot (str): file identifier e.g. fc_test
            storms_to_keep (dataframe): from S_Box
            nice (int, optional): niceness 1/nice share of machine. Default: 4
            shared (str, optional): 'Y' or 'N' if using a shared resource.
                Default: 'Y'.

        Returns:
            csvfile: csvfile eg:
                fc_standard.csv
        '''
        # Check parallel settings and fair use.
        # DO NOT BE RUDE ON SHARED RESOURCES - PLAYING NICE IS ENFORCED HERE!
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
        # Run datamine in parallel
        pstorms = parallelize_dataframe(df, self.gen_vars, nice)
        # Write out the file
        pstorms.to_csv(csvroot+'_standard.csv', sep=',')

    def mean99(self, var, strings, idx, p=99):
        '''Description:
            Find the mean for the midday slice and the Nth PERCENTILE of an
            iris and mean of cube variable for the eveing slice

        Args:
            var (iris.Cube): iris cube variable
            strings (:obj:`list` of :obj:`str`): list of strings
                [str1200, strmean, str99p] corresponding to column name
            idx (int): row index
            p (obj:`int`, optional): PERCENTILE normally 1 or 99.
                Defaults to 99.
        Returns:
            Fills data frame value
        '''
        eve = [11, 17, 17]
        for i in range(2):
            vari = var[eve[i], :, :]
            varn = cubemean(vari)
            self.allvars[strings[i]].loc[idx] = varn.data
        var99p = cube99(vari, per=p)
        self.allvars[strings[2]].loc[idx] = var99p

    def calc_t15(self, t15f, xy, idx):
        '''Description: mean99 for T15 (air_temperature) variable
        Args:
            t15f (str): file name for t15 variable
            xy (iris.Constraint): from genslice
            idx (int): row index
        Returns:
            Fills data frame values:
             'mean_T15_1200', 'mean_T15_1800', '1perc_T15_1800'
        '''
        t15 = iris.load_cube(t15f).extract(xy)
        # Column names in DataFrame
        strings = ['mean_T15_1200', 'mean_T15_1800', '1perc_T15_1800']
        self.mean99(t15, strings, idx, p=1)

    def calc_mslp(self, fname, xy, idx):
        '''mean99 for mslp (mean sea level pressure)
        Args:
            fname (str): file to load cube from
            xy (iris.Constraint): from genslice
            idx (int): index

        Returns:
            Fills data frame values:
                'midday_mslp', 'eve_mslp_mean', 'eve_mslp_1p'
        '''
        allvari = iris.load_cube(fname, xy)
        # Column names in DataFrame
        strings = ['midday_mslp', 'eve_mslp_mean', 'eve_mslp_1p']
        self.mean99(allvari, strings, idx)

    def calc_winds(self, u, v, idx):
        '''Calculate winds for eveing and midday using u and v.
        Args:
            u (iris.cube): x_wind
            v (iris.cube): y_wind
            idx (int): index

        Returns:
            Fills wind related data frame values
        '''
        # Loop for 1200 and 1800
        strings = ['midday_wind', 'midday_wind3', 'eve_wind_mean',
                   'eve_wind3_mean', 'eve_wind_99p', 'eve_wind3_99p']
        eve = [11, 11, 17, 17, 17, 17]
        lex = [0.5, 3, 0.5, 3, 0.5, 3]
        func = cubemean, cubemean, cubemean, cube99, cube99
        func2 = self.dataise, self.dataise, self.dataise, self.nodata, self.nodata
        for x, y, z, f, g in zip(eve, lex, strings, func, func2):
            mwind = (iris.analysis.maths.exponentiate(u[x, :, :], 2) +
                     iris.analysis.maths.exponentiate(v[x, :, :], 2))
            mwind2 = iris.analysis.maths.exponentiate(mwind, y)
            var = f(mwind2)
            self.allvars[z].loc[idx] = g(var)

    def calc_tow(self, of, Tf, xy600, idx):
        '''Description: Calculate bouyancy, omega and max for eveing midday etc
        Args:
            of (str): omega file to load cube from
            Tf (str): temperature file to load cube from
            xy600 (iris.Constraint): from genslice)
            idx (int): index

        Returns:
            Fills wind related data frame values
        '''
        # Constants
        R_GAS = 287.058
        G = 9.80665
        omega = iris.load_cube(of).extract(xy600)
        T = iris.load_cube(Tf).extract(xy600)
        # loop for 1200 and 1800
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
        Args:
            filenames (:obj:`list` of :obj:`str`): uf, vf 10m Velocity files
            xyhi (iris.Constraint): high pressure slice from genslice
            xylw (iris.Constraint): low pressure slice from genslice
            idx (int): index

        Returns:
            Fills wind related data frame values
        '''
        uf, vf = filenames
        u10 = iris.load_cube(uf)
        v10 = iris.load_cube(vf)
        # Only uses 1200 data
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
        '''Calculate midday and eveing mass
        Args:
            wetf (str): wet file to load cube from
            dryf (str): dry file to load cube from
            xy (iris.Constraint): slice from genslice
            idx (int): index

        Returns:
            Fills wind related data frame values
        '''
        # load cubes
        wet = iris.load_cube(wetf).extract(xy)
        dry = iris.load_cube(dryf).extract(xy)
        mass = wet
        mass.data = wet.data + dry.data
        mass.extract(xy)
        eve = [11, 17]
        strings = ['mass_mean_1200', 'mass_mean_1800']
        for x, y, in zip(eve, strings):
            massn = mass[x, :, :]
            massm = cubemean(massn).data
            self.allvars[y].loc[idx] = massm

    def calc_cape(self, fnamelist, u, v, precip99, xy, idx, latlons):
        r'''Description:
            Variable required to calculate Convective Available Potential
            Energy: $\int_{zf}^{zn} g
            \left( \frac{T_{v, parcel} - T_{v,env}}{T_{v,env}}\right)
            \mathrm{d}z$
            If > 0 storms possible if enough moisture
            Notes:
                Blindly follows Rory's meathod
            Args:
                fnamelist (:obj:`list` of :obj:`str`): list of filenames:
                    [qf, q15f, Tf, t15f]
                u (iris.cube): x_wind
                v (iris.cube): y_wind
                precip99 (float): 99th percentile precp
                xy (iris.Constraint): slice from genslice
                idx (int): index
                latlons (:obj:`list` of :obj:`int`): list of lat-lons:
                    [llon, llat, ulat, ulon]
                idx: row index
            Returns:
            CAPE(dataframe): dataframe containing Vars for cape calculations
            TEPHI(dataframe): dataframe containg vars for tephigrams
        '''
        qf, q15f, Tf, t15f = fnamelist
        # ? 975 is a max value ?
        T = iris.load_cube(Tf)
        q = iris.load_cube(qf)
        # Find mslp but keep it greater that 975
        mslp = self.allvars['eve_mslp_mean'].loc[idx]/100
        if mslp > 975:
            mslp = 975
        # Special slice
        xy850 = genslice(latlons, n1=mslp, n2=100)
        T = iris.load_cube(Tf)
        Q = iris.load_cube(qf)
        T850 = T[3, :, :].extract(xy850)
        q850 = Q[3, :, :].extract(xy850)
        Tp = T850.data
        Qp = q850.data
        Tcol = np.nanmean(Tp, axis=(1, 2))
        Qcol = np.nanmean(Qp, axis=(1, 2))
        pressures = T850.coord('pressure').points
        P = np.append(pressures, mslp)
        pval = T850.data.shape[0] + 1
        Temp = Tcol
        T15 = cubemean(iris.load_cube(t15f)[11, :, :].extract(xy)).data
        T = np.append(Temp, T15)
        Tkel = T - 273.16
        humcol = Qcol
        Q15 = cubemean(iris.load_cube(q15f)[11, :, :].extract(xy)).data
        hum = np.append(humcol, Q15)
        dwpt = np.zeros_like(P)
        if pval == 18:
            humidity = ((0.263 * hum * P*100) / 2.714**((17.67*(Tkel))/(T - 29.65)))
            height = T*((mslp/P)**(1./5.257) - 1)/0.0065
            height[-1] = 1.5
            dwpt[:] = self.mcdp(Tkel, humidity)  # vectorized dew_point calc
        else:
            height, dwpt, humity, RH_650 = np.zeros((4, pval))
        RH_650 = RH_650[np.where((P > 690) & (P <= 710))[0]]
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
                          (height[::-1], P[::-1], Tkel[::-1],
                           dwpt[:: -1])))
        try:
            S = sk.Sounding(soundingdata=mydata)
            parcel = S.get_parcel('mu')
            P_lcl, P_lfc, P_el, CAPE, CIN = S.get_cape(*parcel)
        except AssertionError:
            print('dew_point = ', dwpt[:: -1])
            print('height = ', height[:: -1])
            print('pressures = ', P[:: -1])
            print('Temp = ', TKel[:: -1])
            print('AssertionError: Use a monotonically increasing abscissa')
            print('Setting to np.nan')
            P_lcl, P_lfc, P_el, CAPE, CIN = [np.nan, np.nan, np.nan, np.nan, np.nan]
        self.allvars['CAPE_P_lcl'].loc[idx] = P_lcl
        self.allvars['CAPE_P_lfc'].loc[idx] = P_lfc
        self.allvars['CAPE_P_el'].loc[idx] = P_el
        self.allvars['CAPE_CAPE'].loc[idx] = CAPE
        self.allvars['CAPE_CIN'].loc[idx] = CIN
