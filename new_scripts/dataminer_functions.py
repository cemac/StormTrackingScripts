""" Python module for working Storm tracking CSV files
    STAGE = 2
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
        self.varlistex = ['stormid', 'year', 'month', 'day', 'hour', 'llon',
                          'ulon', 'llat', 'ulat', 'centlon', 'centlat', 'area',
                          'mean_olr']
        self.vars = pd.read_csv('vars.csv')

    def gen_storms_to_keep(self, altcsvname):
        dates = gent(altcsvname, delimiter=',', names=[self.varlistex])
        dates = np.sort(dates[:], axis=-1, order=['stormid', 'mean_olr'])
        storms_to_keep = np.zeros((1, 10), float)
        # You need to bear in mind that this code wants to track the point when
        # the storm is at minimum OLR.
        for line in dates:
            strm = line['stormid']
            goodup = 0
        for rw in range(0, storms_to_keep.shape[0]):
            if int(strm) == int(storms_to_keep[rw, 8]):
                goodup += 1
                continue
        if goodup < 1 and 18 == int(line['hour']):
            if np.sum(storms_to_keep[0, :]) == 0:
                storms_to_keep[0, :] = line[self.varlist]
            else:
                temp = np.zeros((1, 10), float)
                temp[0, :] = line[self.varlist]
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
                                       str(storminfo) + '*-*.nc')[0],
                             foldername[i]]
            except IndexError:
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
        vels = 0
        vels2 = 0
        ot = 0
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
                print('WARNING: Storm missing data files, skipping...')
                continue
            # Initialise row
            allvars.loc[row[0]] = 0
            allvars['storms_to_keep'].loc[row[0]] = row.stormid
            allvars['OLRs'].loc[row[0]] = row.mean_olr
            xy = self.genslice(row.llon, row.llat, row.ulat, row.ulon)
            xyhi = self.genslice(row.llon, row.llat, row.ulat, row.ulon,
                                 n1=500, n2=800)
            xylw = self.genslice(row.llon, row.llat, row.ulat, row.ulon,
                                 n1=925, n2=800)
            xy600 = self.genslice(row.llon, row.llat, row.ulat, row.ulon,
                                  n1=600)
            evemid = [11, 17]
            lvl = [0.5, 3]
            nums = [3, 5]
            # Find if precip is meets criteria
            precipfile = flist[flist.codes == 'a04203'].file
            precip = iris.load_cube(precipfile).extract(xy)
            precipm = precip[11:15, :, :]
            precip = precip[17, :, :]
            precip = precip.collapsed(['latitude', 'longitude'],
                                      iris.analysis.PERCENTILE,
                                      percent=99).data
            precip = np.ndarray.tolist(precip)
            precipm = precipm.collapsed(['time', 'latitude', 'longitude'],
                                        iris.analysis.MEAN).data
            if precipm >= 0.1/3600. and precip <= 1.0/3600.:
                continue
            else:
                allvars['precip_99th_perc'].loc[row[0]] = precip
                pvol = (precipm * (float(row.ulon) - float(row.llon)) *
                                  (float(row.ulat) - float(row.llat)))
                allvars['precip_accum'].loc[row[0]] = pvol
            for rw in flist.itertuples():
                if rw.codes == 'a04203':
                    continue
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
                    continue
                elif rw.codes in ('c03225'):
                    u = iris.load_cube(rw.file, xy)
                    vels += 1
                elif rw.codes in ('c03226'):
                    v = iris.load_cube(rw.file, xy)
                    vels += 1
                if vels == 2:
                    vels = 0
                    for num in evemid:
                        mwind = (iris.analysis.maths.exponentiate(u[num, :, :],
                                                                  2) +
                                 iris.analysis.maths.exponentiate(v[num, :, :],
                                                                  2))
                        for lex in lvl:
                            mwind2 = iris.analysis.maths.exponentiate(mwind,
                                                                      lex)
                            mwind1p = mwind2.collapsed(['latitude', 'longitude'],
                                                       iris.analysis.PERCENTILE,
                                                       percent=99).data
                            mwind2 = mwind2.collapsed(['latitude', 'longitude'],
                                                      iris.analysis.MEAN).data
                            if lex == 0.5 and num == 11:
                                allvars['midday_wind'].loc[row[0]] = mwind2
                            elif num == 11 and lex == 3:
                                allvars['midday_wind3'].loc[row[0]] = mwind2
                            elif lex == 0.5 and num == 17:
                                allvars['eve_wind_mean'].loc[row[0]] = mwind2
                                allvars['eve_wind_99p'].loc[row[0]] = mwind1p
                            elif lex == 3 and num == 17:
                                allvars['eve_wind3_mean'].loc[row[0]] = mwind2
                                allvars['eve_wind3_99p'].loc[row[0]] = mwind1p
                    continue
                else:
                    var = iris.load_cube(rw.file)
                    if rw.codes in ('f30201'):
                        highu = var.extract(xyhi)[3, :, :, :]
                        highu = highu.collapsed(['latitude', 'longitude'],
                                                iris.analysis.MEAN)
                        lowu = var.extract(xylw)[3, :, :, :]
                        lowu = lowu.collapsed(['latitude', 'longitude'],
                                              iris.analysis.MEAN)
                        vels2 += 1
                        continue
                    elif rw.codes in ('f30202'):
                        highv = var.extract(xyhi)[3, :, :, :]
                        highv = highv.collapsed(['latitude', 'longitude'],
                                                iris.analysis.MEAN)
                        lowv = var.extract(xylw)[3, :, :, :]
                        lowv = lowv.collapsed(['latitude', 'longitude'],
                                              iris.analysis.MEAN)
                        vels2 += 1
                        continue
                    if vels2 == 2:
                        vels2 = 0
                        lowup = lowu.collapsed(['pressure'], iris.analysis.MAX)
                        hiup = highu.collapsed(['pressure'], iris.analysis.MIN)
                        mshear = lowup.data - hiup.data
                        allvars['max_shear'].loc[row[0]] = mshear
                        maxcheck = 0
                        for p1 in lowu.coord('pressure').points:
                                for p2 in highu.coord('pressure').points:
                                        lowslice = iris.Constraint(pressure=lambda cell:
                                                                   cell == p1)
                                        highslice = iris.Constraint(pressure=lambda cell:
                                                                    cell == p2)
                                        lowup2 = lowu.extract(lowslice).data
                                        lowvp2 = lowv.extract(lowslice).data
                                        highup2 = highu.extract(highslice).data
                                        highvp2 = highv.extract(highslice).data
                                        shearval = ((lowup2 - highup2)**2 +
                                                    (lowvp2 - highvp2)**2)**0.5
                                        if shearval > maxcheck:
                                            maxcheck = shearval
                        allvars['hor_shear'].loc[row[0]] = maxcheck
                        continue
                    if rw.codes in ('f30208'):
                        omega = var.extract(xy600)
                        ot += 1
                        continue
                    elif rw.codes in ('f30204'):
                        T = var.extract(xy600)
                        ot += 1
                        continue
                    else:
                        varn = var.extract(xy)
                    if ot == 2:
                        ot = 0
                        for no in nums:
                            Tnum = T[no, :, :, :]
                            Onum = omega[no, :, :, :]
                            Onum_1p = Onum.collapsed(['pressure', 'latitude',
                                                      'longitude'],
                                                     iris.analysis.MIN).data
                            Onum_1p = np.ndarray.tolist(Onum_1p)
                            Onum_holdr = Onum_1200.data
                            ps = Onum.coord('pressure').points
                            for p in range(0, Onum_holdr.shape[0]):
                                for y in range(0, Onum_holdr.shape[1]):
                                    for x in range(0, Onum_holdr.shape[2]):
                                        if omega_12_holdr[p, y, x] == Onum_1p:
                                            T_min = Tnum[p, y, x].data
                                            plev = p
                            gas = 287.058
                            g = 9.80665
                            rho = (100.*ps[plev])/(rgas*T_min)
                            wnum = -1*Onum_1p/(rho*g)
                            Onumn = Onum[plev, :, :].collapsed(['latitude',
                                                                'longitude'],
                                                               iris.analysis.MEAN).data
                            Tnum = Tnum[plev, :, :].collapsed(['latitude',
                                                               'longitude'],
                                                              iris.analysis.MEAN)
                            B1p = T_min - Tnum.data
                            if no == 3:
                                allvars['omega_1200_1p.'].loc[row[0]] = Onum_1p
                                allvars['omega_1200_mean'].loc[row[0]] = Onumn
                                allvars['buoyancy_1200_1p'].loc[row[0]] = B1p
                                allvars['max_w_1200'].loc[row[0]] = wnum
                            else:
                                allvars['omega_1800_1p.'].loc[row[0]] = Onum_1p
                                allvars['omega_1800_mean'].loc[row[0]] = Onumn
                                allvars['buoyancy_1800_1p'].loc[row[0]] = B1p
                                allvars['max_w_1800'].loc[row[0]] = wnum


            break
        allvars.to_csv('test2.csv')
        return

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
