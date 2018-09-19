'''
dataminer_1800z_all_stats_CAPE.py
find the difference between c30404 and c30403 for TCW
and the shear values
'''
import iris
import scipy.stats as stat
import numpy as np
from numpy import genfromtxt as gent
import matplotlib.pyplot as plt
from iris.experimental.equalise_cubes import equalise_attributes
import collections
from matplotlib import colors
import glob
import pandas as pd


# Replace xstart etc to match storm in box.py
def main(x1, x2, y1, y2, size_of_storm):

    fname = ('/nfs/a65/eejac/VERA/IMPALA/olr_tracking_12km/stats/' +
             'WAfrica_Rory/*/*.txt')
    C4_CC_list = []
    C4_FC_list = []
    flelist = glob.glob(fname)
    for element in range(0, len(flelist)):
        if 'fc' in flelist[element]:
                fle = pd.read_fwf(flelist[element], header=None)
                datu = np.asarray(fle)
                for rw in range(0, datu.shape[0]):
                        if datu[rw, -1] == '4' or datu[rw, -1] == '2':
                                C4_FC_list.extend([datu[rw, 0]])
        else:
                fle = pd.read_fwf(flelist[element], header=None)
                datu = np.asarray(fle)
                for rw in range(0, datu.shape[0]):
                        if datu[rw, -1] == '4' or datu[rw, -1] == '2':
                                C4_CC_list.extend([datu[rw, 0]])

    # Ok so first we want to bring in the csv file
    try:
        storms_to_keep = gent('../fc_storms_to_keep_area_'+str(size_of_storm) +
                              '_longitudes_'+str(x1)+'_'+str(x2)+'_'+str(y1) +
                              '_'+str(y2)+'_1800Z.csv', delimiter=',')
        print storms_to_keep.shape
    except IOError:
        dates = gent('../CP4_FC_precip_storms_over_box_area_' +
                     str(size_of_storm) + '_lons_' + str(x1) + '_' + str(x2) +
                     '_lat_'+str(y1)+'_'+str(y2) + '.csv', delimiter=',',
                     names=['stormid', 'year', 'month', 'day', 'hour', 'llon',
                            'ulon', 'llat', 'ulat', 'centlon', 'centlat',
                            'area', 'mean_olr'])
    dates = np.sort(dates[:], axis=-1, order=['stormid', 'mean_olr'])
    print dates.shape
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
            storms_to_keep[0, 0] = line['year']
            storms_to_keep[0, 1] = line['month']
            storms_to_keep[0, 2] = line['day']
            storms_to_keep[0, 3] = line['hour']
            storms_to_keep[0, 4] = line['llon']
            storms_to_keep[0, 5] = line['ulon']
            storms_to_keep[0, 6] = line['llat']
            storms_to_keep[0, 7] = line['ulat']
            storms_to_keep[0, 8] = line['stormid']
            storms_to_keep[0, 9] = line['mean_olr']
        else:
            temp = np.zeros((1, 10), float)
            temp[0, 0] = line['year']
            temp[0, 1] = line['month']
            temp[0, 2] = line['day']
            temp[0, 3] = line['hour']
            temp[0, 4] = line['llon']
            temp[0, 5] = line['ulon']
            temp[0, 6] = line['llat']
            temp[0, 7] = line['ulat']
            temp[0, 8] = line['stormid']
            temp[0, 9] = line['mean_olr']
            storms_to_keep = np.concatenate((storms_to_keep, temp), axis=0)

    np.savetxt('../fc_storms_to_keep_area_' + str(size_of_storm) +
               '_longitudes_' + str(x1) + '_' + str(x2) + '_' + str(y1) +
               '_' + str(y2) + '_1800Z.csv', storms_to_keep[:, :],
               delimiter=',')
    # break the try loop
    Stormnum = 0
    GOODUN = 0
    keepun = 0
    olrkeepers = []
    list_of_storms = collections.Counter(storms_to_keep[:, 8])
    try:
        guaranteed_failsafe = gent('nofile.csv', delimiter=',')
    except IOError:
        all_stormid = []
        all_max_shear = []
        all_hor_shear = []
        all_buoyancy_1800_1p = []
        all_buoyancy_1200_1p = []
        all_omega_1200_1p = []
        all_omega_1800_1p = []
        all_omega_1200_mean = []
        all_omega_1800_mean = []
        all_mass_mean_1200 = []
        all_mass_mean_1800 = []
        all_precip_99th_perc = []
        all_precip_accum = []
        all_col_w_mean = []
        all_col_w_p99 = []
        OLRs = []
        all_OLR_10_perc = []
        all_OLR_1_perc = []
        all_max_w_1200 = []
        all_max_w_1800 = []
        all_area = []
        all_mean_T15_1200 = []
        all_mean_T15_1800 = []
        all_1perc_T15_1800 = []
        all_midday_mslp = []
        all_midday_wind = []
        all_midday_wind3 = []
        all_eve_mslp_mean = []
        all_eve_wind_mean = []
        all_eve_wind3_mean = []
        all_eve_mslp_1p = []
        all_eve_wind_99p = []
        all_eve_wind3_99p = []

    for rw in range(0, storms_to_keep.shape[0]):
        OLRmin = 300.0
        ukeep925 = 0.0
        ukeep650 = 0.0
        ukeepsheer = 0.0
        olrkeepers = []
        U925 = []
        U650 = []
        USHEER = []
        a = str(storms_to_keep[rw, 0])
        b = str(storms_to_keep[rw, 1])
        c = str(storms_to_keep[rw, 2])
        d = int(storms_to_keep[rw, 3])
        if float(b) < 10:
            b = '0'+str(b)
        else:
            b = str(b)
        if float(c) < 10:
            c = '0'+str(c)
        else:
            c = str(c)
        if float(c) < 30:
            nextc = float(c) + 1
        else:
            nextc = 1
        if float(nextc) < 10:
            nextc = '0'+str(nextc)
        else:
            nextc = str(nextc)

        llo = storms_to_keep[rw, 4]
        ulo = storms_to_keep[rw, 5]
        lla = storms_to_keep[rw, 6]
        ula = storms_to_keep[rw, 7]
        xysmallslice = iris.Constraint(longitude=lambda cell: float(llo) <=
                                       cell <= float(ulo), latitude=lambda
                                       cell: float(lla) <= cell <= float(ula))

        xysmallslice_925 = iris.Constraint(pressure=lambda cell: 925 == cell,
                                           longitude=lambda cell: float(llo) <=
                                           cell <= float(ulo), latitude=lambda
                                           cell: float(lla) <= cell <=
                                           float(ula))
        xysmallslice_650 = iris.Constraint(pressure=lambda cell: 650 == cell,
                                           longitude=lambda cell: float(llo) <=
                                           cell <= float(ulo), latitude=lambda
                                           cell: float(lla) <= cell <=
                                           float(ula))
        xysmallslice_850 = iris.Constraint(pressure=lambda cell: 850 == cell,
                                           longitude=lambda cell: float(llo) <=
                                           cell <= float(ulo), latitude=lambda
                                           cell: float(lla) <= cell <=
                                           float(ula))
        xysmallslice_500 = iris.Constraint(pressure=lambda cell: 500 == cell,
                                           longitude=lambda cell: float(llo) <=
                                           cell <= float(ulo), latitude=lambda
                                           cell: float(lla) <= cell <=
                                           float(ula))
        xysmallslice_600plus = iris.Constraint(pressure=lambda cell: 600 >=
                                               cell >= 300, longitude=lambda
                                               cell: float(llo) <= cell <=
                                               float(ulo), latitude=lambda
                                               cell: float(lla) <= cell <=
                                               float(ula))
        xysmallslice_800_925 = iris.Constraint(pressure=lambda cell: 925 >=
                                               cell >= 800, longitude=lambda
                                               cell: float(llo) <= cell <=
                                               float(ulo), latitude=lambda
                                               cell: float(lla) <= cell <=
                                               float(ula))
        xysmallslice_500_800 = iris.Constraint(pressure=lambda cell: 500 <=
                                               cell <= 800 or cell == 60,
                                               longitude=lambda cell:
                                               float(llo) <= cell <=
                                               float(ulo), latitude=lambda
                                               cell: float(lla) <= cell <=
                                               float(ula))

        goodcheck = 0

        flelist = glob.glob('/nfs/a299/IMPALA/data/fc/4km/c03236/' +
                            'c03236_A1hr_inst_*4km_' + str(a[:4]) + '' +
                            str(b[:2]) + '' + str(c[:2]) + '0100-' + str(a[:4])
                            + '' + str(b[:2]) + '*0000.nc')
        if len(flelist) > 0:
            goodcheck = goodcheck + 1
            fle_T15 = iris.load_cube(str(flelist[0]))
        flelist = glob.glob('/nfs/a299/IMPALA/data/fc/4km/f30201/' +
                            'f30201_A3hr_inst_*_fc4km_' + str(a[:4]) + '' +
                            str(b[:2]) + '' + str(c[:2]) + '0300-*0000.nc')
        if len(flelist) >= 1:
            fle_u = iris.load_cube(flelist[0])
            goodcheck = goodcheck + 1
        flelist = glob.glob('/nfs/a299/IMPALA/data/fc/4km/f30202/' +
                            'f30202_A3hr_inst_*_fc4km_' + str(a[:4]) + ''
                            + str(b[:2]) + '' + str(c[:2]) +
                            '0300-*0000.nc')
        if len(flelist) >= 1:
            fle_v = iris.load_cube(flelist[0])
            goodcheck = goodcheck + 1
        flelist = glob.glob('/nfs/a299/IMPALA/data/fc/4km/f30208/' +
                            'f30208_A3hr_inst_*_fc4km_' + str(a[:4]) +
                            ''+str(b[:2])+''+str(c[:2])+'0300-*0000.nc')
        if len(flelist) >= 1:
            fle_omega = iris.load_cube(flelist[0])
            goodcheck = goodcheck + 1
        flelist = glob.glob('/nfs/a299/IMPALA/data/fc/4km/f30204/' +
                            'f30204_A3hr_inst_*_fc4km_' + str(a[:4]) +
                            '' + str(b[:2]) + '' + str(c[:2]) +
                            '0300*0000.nc')
        if len(flelist) >= 1:
                fle_T = iris.load_cube(flelist[0])
                goodcheck = goodcheck + 1
        flelist = glob.glob('/nfs/a299/IMPALA/data/fc/4km/a04203/' +
                            'a04203_A1hr_mean_*_fc4km_' + str(a[:4]) +
                            '' + str(b[:2]) + '' + str(c[:2]) +
                            '0030*2330.nc')
        if len(flelist) >= 1:
            fle_precip = iris.load_cube(flelist[0])
            goodcheck = goodcheck + 1
        flelist = glob.glob('/nfs/a299/IMPALA/data/fc/4km/c30403/' +
                            'c30403_A1hr_inst_*_fc4km_' + str(a[:4]) + '' +
                            str(b[:2]) + '' + str(c[:2]) + '0100-*0000.nc')
        if len(flelist) >= 1:
            fle_dry_mass = iris.load_cube(flelist[0])
            goodcheck = goodcheck + 1
        flelist = glob.glob('/nfs/a299/IMPALA/data/fc/4km/c30404/c30404_A1hr_inst_*_fc4km_'+str(a[:4])+''+str(b[:2])+''+str(c[:2])+'0100-*0000.nc')
        if len(flelist) >= 1:
            fle_wet_mass = iris.load_cube(flelist[0])
            goodcheck = goodcheck + 1
        flelist = glob.glob('/nfs/a299/IMPALA/data/fc/4km/a30439/a30439_A3hr_mean_*_fc4km_'+str(a[:4])+''+str(b[:2])+''+str(c[:2])+'0130-*2230.nc')
        if len(flelist) >= 1:
            fle_col_w = iris.load_cube(flelist[0])
            goodcheck = goodcheck + 1
        flelist = glob.glob('/nfs/a299/IMPALA/data/fc/4km/a03332/a03332_A1hr_mean_*_fc4km_'+str(a[:4])+''+str(b[:2])+''+str(c[:2])+'0030-*2330.nc')
        if len(flelist) >= 1:
            fle_olr = iris.load_cube(flelist[0])
            goodcheck = goodcheck + 1
        flelist = glob.glob('/nfs/a299/IMPALA/data/fc/4km/c00409/c00409_A1hr_inst_*4km_'+str(a[:4])+''+str(b[:2])+''+str(c[:2])+'0100-'+str(a[:4])+''+str(b[:2])+'*0000.nc')
        if len(flelist) >= 1:
            fle_mslp = iris.load_cube(flelist[0], xysmallslice)
            goodcheck = goodcheck + 1
        flelist = glob.glob('/nfs/a299/IMPALA/data/fc/4km/c03225/c03225_*'+str(a[:4])+''+str(b[:2])+''+str(c[:2])+'0100-*.nc')
        if len(flelist) >= 1:
            fle_u10 = iris.load_cube(flelist[0], xysmallslice)
            goodcheck = goodcheck + 1
        flelist = glob.glob('/nfs/a299/IMPALA/data/fc/4km/c03226/c03226*'+str(a[:4])+''+str(b[:2])+''+str(c[:2])+'0100-*.nc')
        if len(flelist) >= 1:
                fle_v10 = iris.load_cube(flelist[0], xysmallslice)
                goodcheck = goodcheck + 1
        if goodcheck == 13:
            cube_col_w = fle_col_w.extract(xysmallslice)
            cube_lowu = fle_u.extract(xysmallslice_800_925)
            cube_highu = fle_u.extract(xysmallslice_500_800)
            cube_lowv = fle_v.extract(xysmallslice_800_925)
            cube_highv = fle_v.extract(xysmallslice_500_800)
            cube_precip = fle_precip.extract(xysmallslice)
            cube_OLR = fle_olr.extract(xysmallslice)
            fle_wet_mass = fle_wet_mass.extract(xysmallslice)
            fle_dry_mass = fle_dry_mass.extract(xysmallslice)
            fle_mass = fle_wet_mass
            fle_mass.data = fle_wet_mass.data - fle_dry_mass.data
            fle_mass.extract(xysmallslice)
            cube_omega = fle_omega.extract(xysmallslice_600plus)
            cube_T = fle_T.extract(xysmallslice_600plus)
            cube_T15 = fle_T15.extract(xysmallslice)
            try:
                cube_col_w = cube_col_w[5, :, :, :]
                # so, this is where we want to add the horizontal shear
                #  Shear vector between a & b = (U_a - Ub)i  + (V_a  - V_b)j
                # , i.e a vector
                midday_lowu = cube_lowu[3, :, :, :]
                midday_highu = cube_highu[3, :, :, :]
                midday_lowv = cube_lowv[3, :, :, :]
                midday_highv = cube_highv[3, :, :, :]
                # first we look for the pressure levels where low and mid
                # level winds are maximised
                midday_lowu = midday_lowu.collapsed(['latitude', 'longitude'],
                                                    iris.analysis.MEAN)
                midday_highu = midday_highu.collapsed(['latitude', 'longitude'],
                                                      iris.analysis.MEAN)
                midday_lowv = midday_lowv.collapsed(['latitude', 'longitude'],
                                                    iris.analysis.MEAN)
                midday_highv = midday_highv.collapsed(['latitude', 'longitude'],
                                                     iris.analysis.MEAN)
                maxcheck = 0.
                plow = 0
                phigh = 0
                for p1 in midday_lowu.coord('pressure').points:
                    for p2 in midday_highu.coord('pressure').points:
                            lowslice = iris.Constraint(pressure=lambda cell:
                                                       cell == p1)
                            highslice = iris.Constraint(pressure=lambda cell:
                                                        cell == p2)
                            lowu = midday_lowu.extract(lowslice)
                            lowv = midday_lowv.extract(lowslice)
                            highu = midday_highu.extract(highslice)
                            highv = midday_highv.extract(highslice)
                            shearval = ((lowu.data - highu.data)**2 +
                                        (lowv.data - highv.data)**2)**0.5
                            if shearval > maxcheck:
                                    maxcheck = shearval
                hor_shear = maxcheck
                cube_midday_lowu = cube_lowu[3, :, :, :].collapsed(['latitude','longitude'], iris.analysis.MEAN)
                cube_midday_highu = cube_highu[3, :, :, :].collapsed(['latitude','longitude'], iris.analysis.MEAN)
                cube_midday_lowu = cube_midday_lowu.collapsed(['pressure'], iris.analysis.MAX)
                cube_midday_highu = cube_midday_highu.collapsed(['pressure'], iris.analysis.MIN)
                max_shear = cube_midday_lowu.data - cube_midday_highu.data
                cube_omega_1200 = cube_omega[3,:,:,:]
                cube_omega_1800 = cube_omega[5,:,:,:]
                cube_T_1200 = cube_T[3,:,:,:]
                cube_T_1800 = cube_T[5,:,:,:]
                cube_mass_1200 = fle_mass[11,:,:]
                cube_mass_1800 = fle_mass[17,:,:]
                cube_midday_precip = cube_precip[11:15,:,:]
                cube_precip = cube_precip[17,:,:]
                cube_OLR = cube_OLR[17,:,:]
                cube_midday_precip = cube_midday_precip.collapsed(['time','latitude','longitude'], iris.analysis.MEAN).data
                midday_u = fle_u10[11,:,:]
                midday_v = fle_v10[11,:,:]
                midday_mslp = fle_mslp[11,:,:]
                # first we look for the pressure levels where low and mid level winds are maximised
                midday_mslp = midday_mslp.collapsed(['latitude','longitude'], iris.analysis.MEAN)
                midday_u = iris.analysis.maths.exponentiate(midday_u, 2)
                midday_v = iris.analysis.maths.exponentiate(midday_v, 2)
                midday_wind = midday_u + midday_v
                midday_wind = iris.analysis.maths.exponentiate(midday_wind, 0.5)
                midday_wind3 = iris.analysis.maths.exponentiate(midday_wind, 3)
                midday_wind = midday_wind.collapsed(['latitude','longitude'], iris.analysis.MEAN)
                midday_wind3 = midday_wind3.collapsed(['latitude','longitude'], iris.analysis.MEAN)
                midday_mslp = midday_mslp.data
                midday_wind = midday_wind.data
                midday_wind3 = midday_wind3.data
                eve_u = fle_u10[17,:,:]
                eve_v = fle_v10[17,:,:]
                eve_mslp = fle_mslp[17,:,:]
                # first we look for the pressure levels where low and mid level
                # winds are maximised
                eve_mslp_mean = eve_mslp.collapsed(['latitude','longitude'], iris.analysis.MEAN)
                eve_mslp_1p = eve_mslp.collapsed(['latitude','longitude'], iris.analysis.PERCENTILE, percent = 99)
                eve_u = iris.analysis.maths.exponentiate(eve_u, 2)
                eve_v = iris.analysis.maths.exponentiate(eve_v, 2)
                eve_wind = eve_u + eve_v
                eve_wind = iris.analysis.maths.exponentiate(eve_wind, 0.5)
                eve_wind3 = iris.analysis.maths.exponentiate(eve_wind,3)
                eve_wind_mean = eve_wind.collapsed(['latitude','longitude'], iris.analysis.MEAN)
                eve_wind_99p = eve_wind.collapsed(['latitude','longitude'], iris.analysis.PERCENTILE, percent = 99)
                eve_wind3_mean = eve_wind3.collapsed(['latitude','longitude'], iris.analysis.MEAN)
                eve_wind3_99p = eve_wind3.collapsed(['latitude','longitude'], iris.analysis.PERCENTILE, percent = 99)
                eve_mslp_mean = eve_mslp_mean.data
                eve_mslp_1p = eve_mslp_1p.data
                eve_wind_mean = eve_wind_mean.data
                eve_wind3_mean = eve_wind3_mean.data
                eve_wind_99p = eve_wind_99p.data
                eve_wind3_99p = eve_wind3_99p.data
            #print(str(eve_wind3_mean - midday_wind3,eve_wind3_mean, midday_wind3,)+"Problem?")
            except TypeError:
                print('error')
            elif cube_midday_precip <= 0.1/3600.0:
                pressers = cube_omega_1200.coord('pressure').points
                cube_col_w_mean = cube_col_w.collapsed(['latitude','longitude'], iris.analysis.MEAN).data
                cube_col_w_p99 = cube_col_w.collapsed(['latitude','longitude'], iris.analysis.PERCENTILE, percent = 99).data
                cube_col_w_mean = np.ndarray.tolist(cube_col_w_mean)
                cube_col_w_p99 = np.ndarray.tolist(cube_col_w_p99)
                cube_omega_1200_1p = cube_omega_1200.collapsed(['pressure','latitude','longitude'], iris.analysis.MIN).data
                cube_omega_1200_1p = np.ndarray.tolist(cube_omega_1200_1p)
                cube_omega_1800_1p = cube_omega_1800.collapsed(['pressure','latitude','longitude'], iris.analysis.MIN).data
                cube_omega_1800_1p = np.ndarray.tolist(cube_omega_1800_1p)
                omega_12_holdr = cube_omega_1200.data
                for p in range(0, omega_12_holdr.shape[0]):
                    for y in range(0,omega_12_holdr.shape[1]):
                        for x in range(0,omega_12_holdr.shape[2]):
                            if omega_12_holdr[p,y,x] == cube_omega_1200_1p:
                                T_min_1200 = cube_T_1200[p,y,x].data
                                pressure_lev_1200 = p

                rgas = 287.058
                g = 9.80665
                rho = (pressers[pressure_lev_1200]*100.)/(rgas*T_min_1200)
                cube_w_1200 = -1*cube_omega_1200_1p/(rho*g)
                omega_18_holdr = cube_omega_1800.data
                for p in range(0,omega_18_holdr.shape[0]):
                    for y in range(0,omega_18_holdr.shape[1]):
                        for x in range(0,omega_18_holdr.shape[2]):
                            if omega_18_holdr[p,y,x] == cube_omega_1800_1p:
                                T_min_1800 = cube_T_1800[p,y,x].data
                                pressure_lev_1800 = p

                rho = (pressers[pressure_lev_1800]*100.)/(rgas*T_min_1800)
                cube_w_1800 = -1*cube_omega_1800_1p/(rho*g)
    		    print 'w', cube_w_1200, cube_w_1800, pressure_lev_1800, pressure_lev_1200
                cube_T_1200 = cube_T_1200[pressure_lev_1200,:,:].collapsed(['latitude','longitude'], iris.analysis.MEAN)
                cube_T_1800 = cube_T_1800[pressure_lev_1800,:,:].collapsed(['latitude','longitude'], iris.analysis.MEAN)
                cube_omega_1200_mean = cube_omega_1200[pressure_lev_1200,:,:].collapsed(['latitude','longitude'], iris.analysis.MEAN)
                cube_omega_1800_mean = cube_omega_1800[pressure_lev_1800,:,:].collapsed(['latitude','longitude'], iris.analysis.MEAN)
                cube_mass_mean_1200 = cube_mass_1200.collapsed(['latitude','longitude'], iris.analysis.MEAN).data
                cube_mass_mean_1800 = cube_mass_1800.collapsed(['latitude','longitude'], iris.analysis.MEAN).data
                cube_mass_mean_1200 = np.ndarray.tolist(cube_mass_mean_1200)
                cube_mass_mean_1800 = np.ndarray.tolist(cube_mass_mean_1800)
                cube_precip_99th_perc = cube_precip.collapsed(['latitude','longitude'], iris.analysis.PERCENTILE, percent = 99).data
                cube_precip_99th_perc = np.ndarray.tolist(cube_precip_99th_perc)
                cube_precip_volume = cube_precip.collapsed(['latitude','longitude'], iris.analysis.MEAN).data *(float(ulo) - float(llo)) * (float(ula) - float(lla))
                OLR_10p = cube_OLR.collapsed(['latitude','longitude'], iris.analysis.PERCENTILE, percent = 10).data
                OLR_1p = cube_OLR.collapsed(['latitude','longitude'], iris.analysis.PERCENTILE, percent = 1).data
                OLR_10p = np.ndarray.tolist(OLR_10p)
                OLR_1p = np.ndarray.tolist(OLR_1p)
                cube_omega_1200_mean = cube_omega_1200_mean.data
                cube_omega_1800_mean = cube_omega_1800_mean.data
                mean_T15_1200 = cube_T15[11,:,:].collapsed(['latitude','longitude'], iris.analysis.MEAN)
                mean_T15_1200 = mean_T15_1200.data
                mean_T15_1800 = cube_T15[17,:,:].collapsed(['latitude','longitude'], iris.analysis.MEAN)
                mean_T15_1800 = mean_T15_1800.data
                one_perc_T15_1800 = cube_T15[17,:,:].collapsed(['latitude','longitude'], iris.analysis.PERCENTILE, percent = 1)
                one_perc_T15_1800 = one_perc_T15_1800.data
                if cube_precip_99th_perc >= 1.0/3600.:
                    all_midday_mslp.extend([midday_mslp])
                    all_midday_wind.extend([midday_wind])
                    all_midday_wind3.extend([midday_wind3])
                    all_eve_mslp_mean.extend([eve_mslp_mean])
                    all_eve_wind_mean.extend([eve_wind_mean])
                    all_eve_wind3_mean.extend([eve_wind3_mean])
                    all_eve_mslp_1p.extend([eve_mslp_1p])
                    all_eve_wind_99p.extend([eve_wind_99p])
                    all_eve_wind3_99p.extend([eve_wind3_99p])
                    all_mean_T15_1200.extend([mean_T15_1200])
                    all_mean_T15_1800.extend([mean_T15_1800])
                    all_1perc_T15_1800.extend([one_perc_T15_1800])
                    all_max_w_1200.extend([cube_w_1200])
                    all_max_w_1800.extend([cube_w_1800])
                    all_stormid.extend([storms_to_keep[rw,8]])
                    all_max_shear.extend([max_shear])
                    all_hor_shear.extend([hor_shear])
                    all_buoyancy_1800_1p.extend([T_min_1800 - cube_T_1800.data])
                    all_buoyancy_1200_1p.extend([T_min_1200 - cube_T_1200.data])
                    all_omega_1200_1p.extend([cube_omega_1200_1p])
                    all_omega_1800_1p.extend([cube_omega_1800_1p])
                    all_omega_1200_mean.extend([cube_omega_1200_mean])
                    all_omega_1800_mean.extend([cube_omega_1800_mean])
                    all_mass_mean_1200.extend([cube_mass_mean_1200])
                    all_mass_mean_1800.extend([cube_mass_mean_1800])
                    all_precip_99th_perc.extend([cube_precip_99th_perc])
                    all_precip_accum.extend([cube_precip_volume])
                    all_col_w_mean.extend([cube_col_w_mean])
                    all_col_w_p99.extend([cube_col_w_p99])
                    OLRs.extend([storms_to_keep[rw,9]])
                    all_OLR_10_perc.extend([OLR_10p])
                    all_OLR_1_perc.extend([OLR_1p])
                    all_area.extend([(float(ulo)-float(llo))*(float(ula)-float(lla))])

        np.savetxt('../csvs/PAPER_FC_C2C4_STORMID_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_stormid, delimiter = ',')
        np.savetxt('../csvs/PAPER_FC_C2C4_1200Z_mean_T15_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_mean_T15_1200, delimiter = ',')
        np.savetxt('../csvs/PAPER_FC_C2C4_1800Z_mean_T15_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_mean_T15_1800, delimiter = ',')
        np.savetxt('../csvs/PAPER_FC_C2C4_1200Z_1perc_T15_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_1perc_T15_1800, delimiter = ',')
        np.savetxt('../csvs/PAPER_FC_C2C4_1800Z_storm_area_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_area, delimiter = ',')
        np.savetxt('../csvs/PAPER_FC_C2C4_1800Z_99p_1p_OLR_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_OLR_1_perc, delimiter = ',')
        np.savetxt('../csvs/PAPER_FC_C2C4_1800Z_99p_10p_OLR_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_OLR_10_perc, delimiter = ',')
        np.savetxt('../csvs/PAPER_FC_C2C4_1800Z_99p_mean_OLR_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', OLRs, delimiter = ',')
        np.savetxt('../csvs/PAPER_FC_C2C4_1800Z_99p_col_int_w_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_col_w_p99, delimiter = ',')
        np.savetxt('../csvs/PAPER_FC_C2C4_1800Z_mean_col_int_w_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_col_w_mean, delimiter = ',')
        np.savetxt('../csvs/PAPER_FC_C2C4_1800Z_99p_precip_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_precip_99th_perc, delimiter = ',')
        np.savetxt('../csvs/PAPER_FC_C2C4_1800Z_precip_volume_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_precip_accum, delimiter = ',')
        np.savetxt('../csvs/PAPER_FC_C2C4_1200Z_mean_TCWV_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_mass_mean_1200, delimiter = ',')
        np.savetxt('../csvs/PAPER_FC_C2C4_1800Z_mean_TCWV_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_mass_mean_1800, delimiter = ',')
        np.savetxt('../csvs/PAPER_FC_C2C4_1800Z_mean_omega_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_omega_1800_mean, delimiter = ',')
        np.savetxt('../csvs/PAPER_FC_C2C4_1800Z_min_omega_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_omega_1800_1p, delimiter = ',')
        np.savetxt('../csvs/PAPER_FC_C2C4_1200Z_mean_omega_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_omega_1200_mean, delimiter = ',')
        np.savetxt('../csvs/PAPER_FC_C2C4_1200Z_min_omega_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_omega_1200_1p, delimiter = ',')
        np.savetxt('../csvs/PAPER_FC_C2C4_1800Z_max_w_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_max_w_1800, delimiter = ',')
        np.savetxt('../csvs/PAPER_FC_C2C4_1200Z_max_w_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_max_w_1200, delimiter = ',')
        np.savetxt('../csvs/PAPER_FC_C2C4_1200Z_max_buoyancy_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_buoyancy_1200_1p, delimiter = ',')
        np.savetxt('../csvs/PAPER_FC_C2C4_1200Z_max_zonal_shear_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_max_shear, delimiter = ',')
        np.savetxt('../csvs/PAPER_FC_C2C4_1200Z_max_horizontal_shear_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_hor_shear, delimiter = ',')
        np.savetxt('../csvs/PAPER_FC_C2C4_1800Z_max_buoyancy_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_buoyancy_1800_1p, delimiter = ',')
        np.savetxt('../csvs/PAPER_FC_C2C4_1200Z_mean_mslp_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_midday_mslp, delimiter = ',')
        np.savetxt('../csvs/PAPER_FC_C2C4_1200Z_mean_10m_wind_speed_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_midday_wind, delimiter = ',')
        np.savetxt('../csvs/PAPER_FC_C2C4_1200Z_mean_10m_wind_speed_cubed_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_midday_wind3, delimiter = ',')
        np.savetxt('../csvs/PAPER_FC_C2C4_1800Z_mean_mslp_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_eve_mslp_mean, delimiter = ',')
        np.savetxt('../csvs/PAPER_FC_C2C4_1800Z_mean_10m_wind_speed_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_eve_wind_mean, delimiter = ',')
        np.savetxt('../csvs/PAPER_FC_C2C4_1800Z_mean_10m_wind_speed_cubed_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_eve_wind3_mean, delimiter = ',')
        np.savetxt('../csvs/PAPER_FC_C2C4_1800Z_99p_mslp_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_eve_mslp_1p, delimiter = ',')
        np.savetxt('../csvs/PAPER_FC_C2C4_1800Z_99p_10m_wind_speed_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_eve_wind_99p, delimiter = ',')
        np.savetxt('../csvs/PAPER_FC_C2C4_1800Z_99p_10m_wind_speed_cubed_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_eve_wind3_99p, delimiter = ',')

    try:
        storms_to_keep = gent('../OLR_12km_storms_to_keep_area_'+str(size_of_storm)+'_longitudes_'+str(x1)+'_'+str(x2)+'_'+str(y1)+'_'+str(y2)+'_1800Z.csv', delimiter = ',')
        print storms_to_keep.shape
    except IOError:
        dates = gent('../CP4_CC_precip_storms_over_box_area_'+str(size_of_storm)+'_lons_'+str(x1)+'_'+str(x2)+'_lat_'+str(y1)+'_'+str(y2)+'.csv', delimiter = ',', names = ['stormid','year','month','day','hour','llon','ulon','llat','ulat','centlon','centlat','area','mean_olr'])
        dates = np.sort(dates[:], axis = -1, order = ['stormid','mean_olr'])
        print dates.shape
        storms_to_keep = np.zeros((1,10),float)# You need to bear in mind that this code wants to track the point when the storm is at minimum OLR.
        for line in dates:
            strm = line['stormid']
            goodup = 0
            for rw in range(0,storms_to_keep.shape[0]):
                if int(strm) == int(storms_to_keep[rw,8]):
                    goodup = goodup + 1
                    continue
            if goodup < 1 and 18 == int(line['hour']):
                if np.sum(storms_to_keep[0,:]) ==0:
                    storms_to_keep[0,0] = line['year']
                    storms_to_keep[0,1] = line['month']
                    storms_to_keep[0,2] = line['day']
                    storms_to_keep[0,3] = line['hour']
                    storms_to_keep[0,4] = line['llon']
                    storms_to_keep[0,5] = line['ulon']
                    storms_to_keep[0,6] = line['llat']
                    storms_to_keep[0,7] = line['ulat']
                    storms_to_keep[0,8] = line['stormid']
                    storms_to_keep[0,9] = line['mean_olr']
                else:
                    temp = np.zeros((1,10),float)
                    temp[0,0] = line['year']
                    temp[0,1] = line['month']
                    temp[0,2] = line['day']
                    temp[0,3] = line['hour']
                    temp[0,4] = line['llon']
                    temp[0,5] = line['ulon']
                    temp[0,6] = line['llat']
                    temp[0,7] = line['ulat']
                    temp[0,8] = line['stormid']
                    temp[0,9] = line['mean_olr']
                    storms_to_keep = np.concatenate((storms_to_keep,temp),axis = 0)

    np.savetxt('../OLR_12km_storms_to_keep_area_'+str(size_of_storm)+'_longitudes_'+str(x1)+'_'+str(x2)+'_'+str(y1)+'_'+str(y2)+'_1800Z.csv',storms_to_keep[:,:], delimiter = ',')
    all_midday_mslp = []
    all_midday_wind = []
    all_midday_wind3 = []
    all_eve_mslp_mean = []
    all_eve_wind_mean = []
    all_eve_wind3_mean = []
    all_eve_mslp_1p = []
    all_eve_wind_99p = []
    all_eve_wind3_99p = []
    all_mean_T15_1200 = []
    all_mean_T15_1800 = []
    all_1perc_T15_1800 = []
    all_max_w_1200 = []
    all_max_w_1800 = []
    all_stormid = []
    all_max_shear = []
    all_hor_shear = []
    all_buoyancy_1800_1p = []
    all_buoyancy_1200_1p = []
    all_omega_1200_1p = []
    all_omega_1800_1p = []
    all_omega_1200_mean = []
    all_omega_1800_mean = []
    all_mass_mean_1200 = []
    all_mass_mean_1800 = []
    all_precip_99th_perc = []
    all_precip_accum = []
    all_col_w_mean = []
    all_col_w_p99 = []
    OLRs = []
    all_OLR_10_perc = []
    all_OLR_1_perc = []
    all_area = []
    Stormnum = 0
    GOODUN = 0
    keepun = 0
    olrkeepers = []
    list_of_storms = collections.Counter(storms_to_keep[:, 8])
    try:
        guaranteed_failsafe = gent('this_file_doesnt_exist.csv', delimiter = ',')
    except IOError:
        for rw in range(0,storms_to_keep.shape[0]):
            OLRmin = 300.0
            ukeep925 = 0.0
            ukeep650 = 0.0
            ukeepsheer = 0.0
            olrkeepers = []
            U925 = []
            U650 = []
            USHEER = []
            a = str(storms_to_keep[rw,0])
            b = str(storms_to_keep[rw,1])
            c = str(storms_to_keep[rw,2])
            d = int(storms_to_keep[rw,3])
            if float(b) < 10:
                 b = '0'+str(b)
            else:
                  b = str(b)
            if float(c) < 10:
                 c = '0'+str(c)
            else:
                 c = str(c)
            if float(c) < 30:
                     nextc = float(c) + 1
            else:
                nextc = 1
            if float(nextc) < 10:
                 nextc = '0'+str(nextc)
            else:
                 nextc = str(nextc)

            llo = storms_to_keep[rw,4]
            ulo = storms_to_keep[rw,5]
            lla = storms_to_keep[rw,6]
            ula = storms_to_keep[rw,7]
            xysmallslice = iris.Constraint(longitude=lambda cell: float(llo) <= cell <= float(ulo), latitude = lambda cell: float(lla) <= cell <= float(ula))
            xysmallslice_925 = iris.Constraint(pressure=lambda cell: 925 == cell, longitude = lambda cell: float(llo) <= cell <= float(ulo), latitude = lambda cell: float(lla) <= cell <= float(ula))
            xysmallslice_650 = iris.Constraint(pressure=lambda cell: 650 == cell, longitude = lambda cell: float(llo) <= cell <= float(ulo), latitude = lambda cell: float(lla) <= cell <= float(ula))
            xysmallslice_850 = iris.Constraint(pressure=lambda cell: 850 == cell, longitude = lambda cell: float(llo) <= cell <= float(ulo), latitude = lambda cell: float(lla) <= cell <= float(ula))
            xysmallslice_500 = iris.Constraint(pressure=lambda cell: 500 == cell, longitude = lambda cell: float(llo) <= cell <= float(ulo), latitude = lambda cell: float(lla) <= cell <= float(ula))
            xysmallslice_600plus = iris.Constraint(pressure=lambda cell: 600 >= cell >= 300, longitude = lambda cell: float(llo) <= cell <= float(ulo), latitude = lambda cell: float(lla) <= cell <= float(ula))
            xysmallslice_800_925 = iris.Constraint(pressure=lambda cell: 925 >= cell >=800, longitude = lambda cell: float(llo) <= cell <= float(ulo), latitude = lambda cell: float(lla) <= cell <= float(ula))
            xysmallslice_500_800 = iris.Constraint(pressure=lambda cell: 500 <= cell <= 800 or cell == 60, longitude = lambda cell: float(llo) <= cell <= float(ulo), latitude = lambda cell: float(lla) <= cell <= float(ula))
            goodcheck = 0
            flelist = glob.glob('/nfs/a277/IMPALA/data/4km/c03236/c03236_A1hr_inst_*4km_'+str(a[:4])+''+str(b[:2])+''+str(c[:2])+'0100-'+str(a[:4])+''+str(b[:2])+'*0000.nc')
            if len(flelist) > 0:
                goodcheck = goodcheck + 1
                fle_T15 = iris.load_cube(str(flelist[0]))
            flelist = glob.glob('/nfs/a277/IMPALA/data/4km/f30201/f30201_A3hr_inst_*_4km_'+str(a[:4])+''+str(b[:2])+''+str(c[:2])+'0300-*0000.nc')
            if len(flelist) >= 1:
                fle_u = iris.load_cube(flelist[0])
                goodcheck = goodcheck + 1
            flelist = glob.glob('/nfs/a277/IMPALA/data/4km/f30202/f30202_A3hr_inst_*_4km_'+str(a[:4])+''+str(b[:2])+''+str(c[:2])+'0300-*0000.nc')
            if len(flelist) >= 1:
                fle_v = iris.load_cube(flelist[0])
                goodcheck = goodcheck + 1
            flelist = glob.glob('/nfs/a277/IMPALA/data/4km/f30208/f30208_A3hr_inst_*_4km_'+str(a[:4])+''+str(b[:2])+''+str(c[:2])+'0300-*0000.nc')
            if len(flelist) >= 1:
                fle_omega = iris.load_cube(flelist[0])
                goodcheck = goodcheck + 1
            flelist = glob.glob('/nfs/a277/IMPALA/data/4km/f30204/f30204_A3hr_inst_*_4km_'+str(a[:4])+''+str(b[:2])+''+str(c[:2])+'0300*0000.nc')
            if len(flelist) >= 1:
                fle_T = iris.load_cube(flelist[0])
                goodcheck = goodcheck + 1
            flelist = glob.glob('/nfs/a277/IMPALA/data/4km/a04203/a04203_A1hr_mean_*_4km_'+str(a[:4])+''+str(b[:2])+''+str(c[:2])+'0030*2330.nc')
            if len(flelist) >= 1:
                fle_precip = iris.load_cube(flelist[0])
                goodcheck = goodcheck + 1
            flelist = glob.glob('/nfs/a277/IMPALA/data/4km/c30403/c30403_A1hr_inst_*_4km_'+str(a[:4])+''+str(b[:2])+''+str(c[:2])+'0100-*0000.nc')
            if len(flelist) >= 1:
                fle_dry_mass = iris.load_cube(flelist[0])
                goodcheck = goodcheck + 1
            flelist = glob.glob('/nfs/a277/IMPALA/data/4km/c30404/c30404_A1hr_inst_*_4km_'+str(a[:4])+''+str(b[:2])+''+str(c[:2])+'0100-*0000.nc')
            if len(flelist) >= 1:
                fle_wet_mass = iris.load_cube(flelist[0])
                goodcheck = goodcheck + 1
            flelist = glob.glob('/nfs/a277/IMPALA/data/4km/a30439/a30439_A3hr_mean_*_4km_'+str(a[:4])+''+str(b[:2])+''+str(c[:2])+'0130-*2230.nc')
            if len(flelist) >= 1:
                fle_col_w = iris.load_cube(flelist[0])
                goodcheck = goodcheck + 1
            flelist = glob.glob('/nfs/a277/IMPALA/data/4km/a03332/a03332_A1hr_mean_*_4km_'+str(a[:4])+''+str(b[:2])+''+str(c[:2])+'0030-*2330.nc')
            if len(flelist) >= 1:
                fle_olr = iris.load_cube(flelist[0])
                goodcheck = goodcheck + 1
            flelist = glob.glob('/nfs/a277/IMPALA/data/4km/c00409/c00409*4km_'+str(a[:4])+''+str(b[:2])+''+str(c[:2])+'0100-'+str(a[:4])+''+str(b[:2])+'*0000.nc')
            if len(flelist) >= 1:
                fle_mslp = iris.load_cube(flelist[0], xysmallslice)
                goodcheck = goodcheck + 1
            flelist = glob.glob('/nfs/a277/IMPALA/data/4km/c03225/c03225_*'+str(a[:4])+''+str(b[:2])+''+str(c[:2])+'0100-*.nc')
            if len(flelist) >= 1:
                fle_u10 = iris.load_cube(flelist[0],xysmallslice)
                goodcheck = goodcheck + 1
            flelist = glob.glob('/nfs/a277/IMPALA/data/4km/c03226/c03226*'+str(a[:4])+''+str(b[:2])+''+str(c[:2])+'0100-*.nc')
            if len(flelist) >= 1:
                        fle_v10 = iris.load_cube(flelist[0],xysmallslice)
                        goodcheck = goodcheck + 1
        if goodcheck == 13:
            cube_col_w = fle_col_w.extract(xysmallslice)
            cube_lowu = fle_u.extract(xysmallslice_800_925)
            cube_highu = fle_u.extract(xysmallslice_500_800)
            cube_lowv = fle_v.extract(xysmallslice_800_925)
            cube_highv = fle_v.extract(xysmallslice_500_800)
            cube_precip = fle_precip.extract(xysmallslice)
            cube_OLR = fle_olr.extract(xysmallslice)
            fle_wet_mass = fle_wet_mass.extract(xysmallslice)
            fle_dry_mass = fle_dry_mass.extract(xysmallslice)
            fle_mass = fle_wet_mass
            fle_mass.data = fle_wet_mass.data - fle_dry_mass.data
            fle_mass.extract(xysmallslice)
            cube_omega = fle_omega.extract(xysmallslice_600plus)
            cube_T15 = fle_T15.extract(xysmallslice)
            cube_T = fle_T.extract(xysmallslice_600plus)
            try:
                cube_col_w = cube_col_w[5,:,:,:]
                midday_lowu = cube_lowu[3,:,:,:]
                midday_highu = cube_highu[3,:,:,:]
                midday_lowv = cube_lowv[3,:,:,:]
                midday_highv = cube_highv[3,:,:,:]
                # first we look for the pressure levels where low and mid level winds are maximised
                midday_lowu = midday_lowu.collapsed(['latitude','longitude'], iris.analysis.MEAN)
                midday_highu = midday_highu.collapsed(['latitude','longitude'], iris.analysis.MEAN)
                midday_lowv = midday_lowv.collapsed(['latitude','longitude'], iris.analysis.MEAN)
                midday_highv = midday_highv.collapsed(['latitude','longitude'], iris.analysis.MEAN)
                maxcheck = 0.
                plow = 0
                phigh = 0
                for p1 in midday_lowu.coord('pressure').points:
                    for p2 in midday_highu.coord('pressure').points:
                        lowslice = iris.Constraint(pressure = lambda cell: cell == p1)
                        highslice = iris.Constraint(pressure = lambda cell: cell == p2)
                        lowu = midday_lowu.extract(lowslice)
                        lowv = midday_lowv.extract(lowslice)
                        highu = midday_highu.extract(highslice)
                        highv = midday_highv.extract(highslice)
                        shearval = ((lowu.data - highu.data)**2 + (lowv.data - highv.data)**2)**0.5
                        if shearval > maxcheck:
                            maxcheck = shearval
                hor_shear = maxcheck
                cube_midday_lowu = cube_lowu[3,:,:,:].collapsed(['latitude','longitude'], iris.analysis.MEAN)
                cube_midday_highu = cube_highu[3,:,:,:].collapsed(['latitude','longitude'], iris.analysis.MEAN)
                cube_midday_lowu = cube_midday_lowu.collapsed(['pressure'], iris.analysis.MAX)
                cube_midday_highu = cube_midday_highu.collapsed(['pressure'], iris.analysis.MIN)
                max_shear = cube_midday_lowu.data - cube_midday_highu.data
                cube_omega_1200 = cube_omega[3,:,:,:]
                cube_omega_1800 = cube_omega[5,:,:,:]
                cube_T_1200 = cube_T[3,:,:,:]
                cube_T_1800 = cube_T[5,:,:,:]
                cube_mass_1200 = fle_mass[11,:,:]
                cube_mass_1800 = fle_mass[17,:,:]
                cube_midday_precip = cube_precip[11:15,:,:]
                cube_precip = cube_precip[17,:,:]
                cube_OLR = cube_OLR[17,:,:]
                #cube_midday_precip = cube_precip[11:15,:,:]
                cube_midday_precip = cube_midday_precip.collapsed(['time','latitude','longitude'], iris.analysis.MEAN).data
                midday_u = fle_u10[11,:,:]
                midday_v = fle_v10[11,:,:]
                midday_mslp = fle_mslp[11,:,:]
                # first we look for the pressure levels where low and mid level winds are maximised
                midday_mslp = midday_mslp.collapsed(['latitude','longitude'], iris.analysis.MEAN)
                midday_u = iris.analysis.maths.exponentiate(midday_u, 2)
                midday_v = iris.analysis.maths.exponentiate(midday_v, 2)
                midday_wind = midday_u + midday_v
                midday_wind = iris.analysis.maths.exponentiate(midday_wind, 0.5)
                midday_wind3 = iris.analysis.maths.exponentiate(midday_wind, 3)
                midday_wind = midday_wind.collapsed(['latitude','longitude'], iris.analysis.MEAN)
                midday_wind3 = midday_wind3.collapsed(['latitude','longitude'], iris.analysis.MEAN)
                midday_mslp = midday_mslp.data
                midday_wind = midday_wind.data
                midday_wind3 = midday_wind3.data
                eve_u = fle_u10[17,:,:]
                eve_v = fle_v10[17,:,:]
                eve_mslp = fle_mslp[17,:,:]
                # first we look for the pressure levels where low and mid level winds are maximised
                eve_mslp_mean = eve_mslp.collapsed(['latitude','longitude'], iris.analysis.MEAN)
                eve_mslp_1p = eve_mslp.collapsed(['latitude','longitude'], iris.analysis.PERCENTILE, percent = 99)
                eve_u = iris.analysis.maths.exponentiate(eve_u, 2)
                eve_v = iris.analysis.maths.exponentiate(eve_v, 2)
                eve_wind = eve_u + eve_v
                eve_wind = iris.analysis.maths.exponentiate(eve_wind, 0.5)
                eve_wind3 = iris.analysis.maths.exponentiate(eve_wind,3)
                eve_wind_mean = eve_wind.collapsed(['latitude','longitude'], iris.analysis.MEAN)
                eve_wind_99p = eve_wind.collapsed(['latitude','longitude'], iris.analysis.PERCENTILE, percent = 99)
                eve_wind3_mean = eve_wind3.collapsed(['latitude','longitude'], iris.analysis.MEAN)
                eve_wind3_99p = eve_wind3.collapsed(['latitude','longitude'], iris.analysis.PERCENTILE, percent = 99)
                eve_mslp_mean = eve_mslp_mean.data
                eve_mslp_1p = eve_mslp_1p.data
                eve_wind_mean = eve_wind_mean.data
                eve_wind3_mean = eve_wind3_mean.data
                eve_wind_99p = eve_wind_99p.data
                eve_wind3_99p = eve_wind3_99p.data
            except TypeError:
                print 'error'
            else:
                if cube_midday_precip <= 0.1/3600.:
                    pressers = cube_omega_1200.coord('pressure').points
                    cube_col_w_mean = cube_col_w.collapsed(['latitude','longitude'], iris.analysis.MEAN).data
                    cube_col_w_p99 = cube_col_w.collapsed(['latitude','longitude'], iris.analysis.PERCENTILE, percent = 99).data
                    cube_col_w_mean = np.ndarray.tolist(cube_col_w_mean)
                    cube_col_w_p99 = np.ndarray.tolist(cube_col_w_p99)
                    cube_omega_1200_1p = cube_omega_1200.collapsed(['pressure','latitude','longitude'], iris.analysis.MIN).data
                    cube_omega_1200_1p = np.ndarray.tolist(cube_omega_1200_1p)
                    cube_omega_1800_1p = cube_omega_1800.collapsed(['pressure','latitude','longitude'], iris.analysis.MIN).data
                    cube_omega_1800_1p = np.ndarray.tolist(cube_omega_1800_1p)
                    omega_12_holdr = cube_omega_1200.data
                    for p in range(0, omega_12_holdr.shape[0]):
                        for y in range(0,omega_12_holdr.shape[1]):
                            for x in range(0,omega_12_holdr.shape[2]):
                                if omega_12_holdr[p,y,x] == cube_omega_1200_1p:
                                    T_min_1200 = cube_T_1200[p,y,x].data
                                    pressure_lev_1200 = p
                    rgas = 287.058
                    g = 9.80665
                    rho = (100.*pressers[pressure_lev_1200])/(rgas*T_min_1200)
                    cube_w_1200 = -1*cube_omega_1200_1p/(rho*g)
                    omega_18_holdr = cube_omega_1800.data
                    for p in range(0,omega_18_holdr.shape[0]):
                        for y in range(0,omega_18_holdr.shape[1]):
                            for x in range(0,omega_18_holdr.shape[2]):
                                if omega_18_holdr[p,y,x] == cube_omega_1800_1p:
                                    T_min_1800 = cube_T_1800[p,y,x].data
                                    pressure_lev_1800 = p
                    rho = (100.*pressers[pressure_lev_1800])/(rgas*T_min_1800)
                    cube_w_1800 = -1*cube_omega_1800_1p/(rho*g)
                    cube_T_1200 = cube_T_1200[pressure_lev_1200,:,:].collapsed(['latitude','longitude'], iris.analysis.MEAN)
                    cube_T_1800 = cube_T_1800[pressure_lev_1800,:,:].collapsed(['latitude','longitude'], iris.analysis.MEAN)
                    cube_omega_1200_mean = cube_omega_1200[pressure_lev_1200,:,:].collapsed(['latitude','longitude'], iris.analysis.MEAN)
                    cube_omega_1800_mean = cube_omega_1800[pressure_lev_1800,:,:].collapsed(['latitude','longitude'], iris.analysis.MEAN)
                    cube_mass_mean_1200 = cube_mass_1200.collapsed(['latitude','longitude'], iris.analysis.MEAN).data
                    cube_mass_mean_1800 = cube_mass_1800.collapsed(['latitude','longitude'], iris.analysis.MEAN).data
                    cube_mass_mean_1200 = np.ndarray.tolist(cube_mass_mean_1200)
                    cube_mass_mean_1800 = np.ndarray.tolist(cube_mass_mean_1800)
                    cube_precip_99th_perc = cube_precip.collapsed(['latitude','longitude'], iris.analysis.PERCENTILE, percent = 99).data
                    cube_precip_99th_perc = np.ndarray.tolist(cube_precip_99th_perc)
                    cube_precip_volume = cube_precip.collapsed(['latitude','longitude'], iris.analysis.MEAN).data *(float(ulo) - float(llo)) * (float(ula) - float(lla))
                    OLR_10p = cube_OLR.collapsed(['latitude','longitude'], iris.analysis.PERCENTILE, percent = 10).data
                    OLR_1p = cube_OLR.collapsed(['latitude','longitude'], iris.analysis.PERCENTILE, percent = 1).data
                    OLR_10p = np.ndarray.tolist(OLR_10p)
                    OLR_1p = np.ndarray.tolist(OLR_1p)
                    cube_omega_1200_mean = cube_omega_1200_mean.data
                    mean_T15_1200 = cube_T15[11,:,:].collapsed(['latitude','longitude'], iris.analysis.MEAN)
                    mean_T15_1200 = mean_T15_1200.data
                    mean_T15_1800 = cube_T15[17,:,:].collapsed(['latitude','longitude'], iris.analysis.MEAN)
                    mean_T15_1800 = mean_T15_1800.data
                    one_perc_T15_1800 = cube_T15[17,:,:].collapsed(['latitude','longitude'], iris.analysis.PERCENTILE, percent = 1)
                    one_perc_T15_1800 = one_perc_T15_1800.data
                    cube_omega_1800_mean = cube_omega_1800_mean.data
                    if cube_precip_99th_perc >= 1.0/3600.:
                        all_midday_mslp.extend([midday_mslp])
                        all_midday_wind.extend([midday_wind])
                        all_midday_wind3.extend([midday_wind3])
                        all_eve_mslp_mean.extend([eve_mslp_mean])
                        all_eve_wind_mean.extend([eve_wind_mean])
                        all_eve_wind3_mean.extend([eve_wind3_mean])
                        all_eve_mslp_1p.extend([eve_mslp_1p])
                        all_eve_wind_99p.extend([eve_wind_99p])
                        all_eve_wind3_99p.extend([eve_wind3_99p])
                        all_mean_T15_1200.extend([mean_T15_1200])
                        all_mean_T15_1800.extend([mean_T15_1800])
                        all_1perc_T15_1800.extend([one_perc_T15_1800])
                        all_max_w_1200.extend([cube_w_1200])
                        all_max_w_1800.extend([cube_w_1800])
                        all_stormid.extend([storms_to_keep[rw,8]])
                        all_max_shear.extend([max_shear])
                        all_hor_shear.extend([hor_shear])
                        all_buoyancy_1800_1p.extend([T_min_1800 - cube_T_1800.data])
                        all_buoyancy_1200_1p.extend([T_min_1200 - cube_T_1200.data])
                        all_omega_1200_1p.extend([cube_omega_1200_1p])
                        all_omega_1800_1p.extend([cube_omega_1800_1p])
                        all_omega_1200_mean.extend([cube_omega_1200_mean])
                        all_omega_1800_mean.extend([cube_omega_1800_mean])
                        all_mass_mean_1200.extend([cube_mass_mean_1200])
                        all_mass_mean_1800.extend([cube_mass_mean_1800])
                        all_precip_99th_perc.extend([cube_precip_99th_perc])
                        all_precip_accum.extend([cube_precip_volume])
                        all_col_w_mean.extend([cube_col_w_mean])
                        all_col_w_p99.extend([cube_col_w_p99])
                        OLRs.extend([storms_to_keep[rw,9]])
                        all_OLR_10_perc.extend([OLR_10p])
                        all_OLR_1_perc.extend([OLR_1p])
                        all_area.extend([(float(ulo)-float(llo))*(float(ula)-float(lla))])

    np.savetxt('../csvs/PAPER_CC_C2C4_STORMID_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_stormid, delimiter = ',')
    np.savetxt('../csvs/PAPER_CC_C2C4_1800Z_storm_area_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_area, delimiter = ',')
    np.savetxt('../csvs/PAPER_CC_C2C4_1800Z_99p_1p_OLR_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_OLR_1_perc, delimiter = ',')
    np.savetxt('../csvs/PAPER_CC_C2C4_1800Z_99p_10p_OLR_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_OLR_10_perc, delimiter = ',')
    np.savetxt('../csvs/PAPER_CC_C2C4_1800Z_99p_mean_OLR_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', OLRs, delimiter = ',')
    np.savetxt('../csvs/PAPER_CC_C2C4_1800Z_99p_col_int_w_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_col_w_p99, delimiter = ',')
    np.savetxt('../csvs/PAPER_CC_C2C4_1800Z_mean_col_int_w_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_col_w_mean, delimiter = ',')
    np.savetxt('../csvs/PAPER_CC_C2C4_1800Z_99p_precip_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_precip_99th_perc, delimiter = ',')
    np.savetxt('../csvs/PAPER_CC_C2C4_1800Z_precip_volume_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_precip_accum, delimiter = ',')
    np.savetxt('../csvs/PAPER_CC_C2C4_1200Z_mean_TCWV_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_mass_mean_1200, delimiter = ',')
    np.savetxt('../csvs/PAPER_CC_C2C4_1800Z_mean_TCWV_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_mass_mean_1800, delimiter = ',')
    np.savetxt('../csvs/PAPER_CC_C2C4_1800Z_mean_omega_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_omega_1800_mean, delimiter = ',')
    np.savetxt('../csvs/PAPER_CC_C2C4_1800Z_min_omega_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_omega_1800_1p, delimiter = ',')
    np.savetxt('../csvs/PAPER_CC_C2C4_1200Z_mean_omega_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_omega_1200_mean, delimiter = ',')
    np.savetxt('../csvs/PAPER_CC_C2C4_1200Z_min_omega_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_omega_1200_1p, delimiter = ',')
    np.savetxt('../csvs/PAPER_CC_C2C4_1200Z_max_buoyancy_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_buoyancy_1200_1p, delimiter = ',')
    np.savetxt('../csvs/PAPER_CC_C2C4_1200Z_max_zonal_shear_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_max_shear, delimiter = ',')
    np.savetxt('../csvs/PAPER_CC_C2C4_1200Z_max_horizontal_shear_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_hor_shear, delimiter = ',')
    np.savetxt('../csvs/PAPER_CC_C2C4_1800Z_max_buoyancy_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_buoyancy_1800_1p, delimiter = ',')
    np.savetxt('../csvs/PAPER_CC_C2C4_1800Z_max_w_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_max_w_1800, delimiter = ',')
    np.savetxt('../csvs/PAPER_CC_C2C4_1200Z_max_w_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_max_w_1200, delimiter = ',')
    np.savetxt('../csvs/PAPER_CC_C2C4_1200Z_mean_T15_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_mean_T15_1200, delimiter = ',')
    np.savetxt('../csvs/PAPER_CC_C2C4_1800Z_mean_T15_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_mean_T15_1800, delimiter = ',')
    np.savetxt('../csvs/PAPER_CC_C2C4_1200Z_1perc_T15_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_1perc_T15_1800, delimiter = ',')
    np.savetxt('../csvs/PAPER_CC_C2C4_1200Z_mean_mslp_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_midday_mslp, delimiter = ',')
    np.savetxt('../csvs/PAPER_CC_C2C4_1200Z_mean_10m_wind_speed_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_midday_wind, delimiter = ',')
    np.savetxt('../csvs/PAPER_CC_C2C4_1200Z_mean_10m_wind_speed_cubed_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_midday_wind3, delimiter = ',')
    np.savetxt('../csvs/PAPER_CC_C2C4_1800Z_mean_mslp_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_eve_mslp_mean, delimiter = ',')
    np.savetxt('../csvs/PAPER_CC_C2C4_1800Z_mean_10m_wind_speed_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_eve_wind_mean, delimiter = ',')
    np.savetxt('../csvs/PAPER_CC_C2C4_1800Z_mean_10m_wind_speed_cubed_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_eve_wind3_mean, delimiter = ',')
    np.savetxt('../csvs/PAPER_CC_C2C4_1800Z_99p_mslp_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_eve_mslp_1p, delimiter = ',')
    np.savetxt('../csvs/PAPER_CC_C2C4_1800Z_99p_10m_wind_speed_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_eve_wind_99p, delimiter = ',')
    np.savetxt('../csvs/PAPER_CC_C2C4_1800Z_99p_10m_wind_speed_cubed_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', all_eve_wind3_99p, delimiter = ',')




if "__name__" == "__main__":
   main(x1,x2,y1,y2)
