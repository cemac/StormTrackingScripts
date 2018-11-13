# right, so in this file we are setting up all the information needed
# for our CAPE calculations.
# Looking at the literature, we need pressure, temperature, dewpoint temperature, height, and specific humidity (g/kg)
# for all the intense storms. We will use the pressure values 850 hPa upwards in order to remove the issues with 925 hPa T values for some storms.
import iris
import scipy.stats as stat
import numpy as np
from numpy import genfromtxt as gent
import matplotlib.pyplot as plt
from iris.experimental.equalise_cubes import equalise_attributes
import collections
from matplotlib import colors
import glob
import meteocalc
import skewt
from skewt import SkewT as sk
from skewt import SkewT
import pandas as pd
import metpy.calc as metcalc
from metpy.plots import add_metpy_logo, SkewT
from metpy.units import units

def main(xstart,xend,ystart,yend,size_of_storm):

 C4_FC_list = []
 C4_CC_list = []
 flelist = glob.glob('/nfs/a65/eejac/VERA/IMPALA/olr_tracking_12km/stats/WAfrica_Rory/*/*.txt')
 for element in range(0, len(flelist)):
        if 'fc'  in flelist[element]:
                fle = pd.read_fwf(flelist[element], header = None)
                datu = np.asarray(fle)
                for rw in range(0, datu.shape[0]):
                        if datu[rw,-1] == '4' or datu[rw,-1] == '2':
                            C4_FC_list.extend([datu[rw,0]])
        else:
                fle = pd.read_fwf(flelist[element], header = None)
                datu = np.asarray(fle)
                for rw in range(0, datu.shape[0]):
                        if datu[rw,-1] == '4' or datu[rw,-1] == '2':
                            C4_CC_list.extend([datu[rw,0]])
                        #if datu[rw,-1] == '4':
                        #        C4_list.extend([datu[rw,0]])
                        #if datu[rw,-1] == '2':
                        #        C2_list.extend([datu[rw,0]])
 CC_storm_IDs = gent('../csvs/PAPER_CC_C2C4_STORMID_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', delimiter = ',')
 FC_storm_IDs = gent('../csvs/PAPER_FC_C2C4_STORMID_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', delimiter = ',')

 all_pressure = iris.cube.CubeList([])
 all_T = iris.cube.CubeList([])
 all_dewpt = iris.cube.CubeList([])
 all_height = iris.cube.CubeList([])
 all_q = iris.cube.CubeList([])
 all_RH = iris.cube.CubeList([])
 all_metrics = iris.cube.CubeList([])
 Stormnum = 0
 GOODUN = 0
 keepun = 0
 counter = 0
 cntr = 0
 CAPE_stats = np.zeros((len(np.ndarray.tolist(FC_storm_IDs)),6),float)
 try:
     storms_to_keep = gent('../fc_storms_to_keep_area_'+str(size_of_storm)+'_longitudes_'+str(xstart)+'_'+str(xend)+'_'+str(ystart)+'_'+str(yend)+'_1800Z.csv', delimiter = ',')
     print storms_to_keep.shape
 except IOError:
  dates = gent('../CP4_FC_precip_storms_over_box_area_'+str(size_of_storm)+'_lons_'+str(xstart)+'_'+str(xend)+'_lat_'+str(ystart)+'_'+str(yend)+'.csv', delimiter = ',', names = ['stormid','year','month','day','hour','llon','ulon','llat','ulat','centlon','centlat','area','mean_olr'])
  dates = np.sort(dates[:], axis = -1, order = ['stormid','mean_olr'])
  print dates.shape
  storms_to_keep = np.zeros((1,10),float) # You need to bear in mind that this code wants to track the point when the storm is at minimum OLR.
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

 np.savetxt('../fc_storms_to_keep_area_'+str(size_of_storm)+'_longitudes_'+str(xstart)+'_'+str(xend)+'_'+str(ystart)+'_'+str(yend)+'_1800Z.csv',storms_to_keep[:,:], delimiter = ',')

 yearchecker = ['1997','1998','1999','2000','2001','2002','2003','2004','2005','2006']
 model_to_year = ['ag057','ah261','ah261','aj575']
 print storms_to_keep.shape
 OLRs_CC = []
 CAPE_stats_CC = np.zeros((storms_to_keep.shape[0],6), float)
 Stormnum = 0
 GOODUN = 0
 keepun = 0
 olrkeepers = []
 uwinds = np.zeros((1,9), float)
 uwindsNS = np.zeros((1,9),float)
 list_of_storms = collections.Counter(storms_to_keep[:,8])
 all_cube = np.zeros((1,15,6), float)
 rubix = 0
 rubix2 = 0
 RH_650hPa = []
 for rw in range(0, storms_to_keep.shape[0]):
  if float(storms_to_keep[rw,0]) < 100:
   continue
  #elif str(int(storms_to_keep[rw,8])) in C4_FC_list and storms_to_keep[rw,8] in FC_storm_IDs:
  elif storms_to_keep[rw,8] in FC_storm_IDs:
    print 'We are in'
    rubix = rubix + 1
  #elif str(int(storms_to_keep[rw,8])) in C4_FC_list:
  #else:
    OLRmin = 300.0
    ukeep925 = 0.0
    ukeep650 = 0.0
    ukeepsheer = 0.0
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
    xysmallslice = iris.Constraint(longitude = lambda cell: float(llo) <= cell <= float(ulo), latitude = lambda cell: float(lla) <= cell <= float(ula))
# Now, the first thing we do is load precipitation and see if it is an extreme storm
    try:
            flelist = glob.glob('/nfs/a299/IMPALA/data/fc/4km/a04203/a04203_A1hr_mean_*4km_'+str(a[:4])+''+str(b[:2])+''+str(c[:2])+'0030-'+str(a[:4])+''+str(b[:2])+''+str(c[:2])+'2330.nc')
            fle_precip = iris.load_cube(str(flelist[0]))
    except IOError:
        print 'No precip file'
    else:
            cube_precip = fle_precip.extract(xysmallslice)
            p99 = cube_precip[17,:,:].collapsed(['latitude','longitude'], iris.analysis.PERCENTILE, percent = 99).data*3600.
            cube_midday_precip = cube_precip[11:15,:,:]
            cube_midday_precip = cube_midday_precip.collapsed(['time','latitude','longitude'], iris.analysis.MEAN).data
            cube_precip = cube_precip[17,:,:].collapsed(['latitude','longitude'], iris.analysis.MEAN).data*3600.
            print cube_midday_precip * 3600.
#        if cube_midday_precip <= 0.1/3600: rubix2 = rubix2 + 1
#        if p99 >= 1. and cube_midday_precip <= 0.1/3600.:
            print 'extreme storm'
            goodup = 0
            try:
                flecount = []
                flelist = glob.glob('/nfs/a299/IMPALA/data/fc/4km/f30204/f30204_A3hr_inst_*4km_'+str(a[:4])+''+str(b[:2])+''+str(c[:2])+'0300-*0000.nc')
                flecount.extend([len(flelist)])
                if len(flelist) > 0:
                    fle_T = iris.load_cube(str(flelist[0]))
                    goodup = goodup + 1
                    flelist = glob.glob('/nfs/a299/IMPALA/data/fc/4km/f30205/f30205_A3hr_inst_*4km_'+str(a[:4])+''+str(b[:2])+''+str(c[:2])+'0300*0000.nc')
                flecount.extend([len(flelist)])
                if len(flelist) > 0:
                    goodup = goodup + 1
                    fle_q = iris.load_cube(str(flelist[0]))
                    flelist = glob.glob('/nfs/a299/IMPALA/data/fc/4km/c00409/c00409_A1hr_inst_*4km_'+str(a[:4])+''+str(b[:2])+''+str(c[:2])+'0100-*0000.nc')
                flecount.extend([len(flelist)])
                if len(flelist) > 0:
                    goodup = goodup + 1
                    fle_mslp = iris.load_cube(str(flelist[0]))
                    flelist = glob.glob('/nfs/a299/IMPALA/data/fc/4km/c03236/c03236_A1hr_inst_*4km_'+str(a[:4])+''+str(b[:2])+''+str(c[:2])+'0100-*0000.nc')
                flecount.extend([len(flelist)])
                if len(flelist) > 0:
                    goodup = goodup + 1
                    fle_T15 = iris.load_cube(str(flelist[0]))
                    flelist = glob.glob('/nfs/a299/IMPALA/data/fc/4km/c03237/c03237_A1hr_inst_*4km_'+str(a[:4])+''+str(b[:2])+''+str(c[:2])+'0100-*0000.nc')
                flecount.extend([len(flelist)])
                if len(flelist) > 0:
                    goodup = goodup + 1
                    fle_q15 = iris.load_cube(str(flelist[0]))

###################################################################################

                    flelist = glob.glob('/nfs/a299/IMPALA/data/fc/4km/f30201/f30201_A3hr_inst_*4km_'+str(a[:4])+''+str(b[:2])+''+str(c[:2])+'0300-*0000.nc')
                    flecount.extend([len(flelist)])
                if len(flelist) > 0:
                    fle_u = iris.load_cube(str(flelist[0]),xysmallslice)
                    goodup = goodup + 1
                    flelist = glob.glob('/nfs/a299/IMPALA/data/fc/4km/f30202/f30202_A3hr_inst_*4km_'+str(a[:4])+''+str(b[:2])+''+str(c[:2])+'0300-*0000.nc')
                    flecount.extend([len(flelist)])
                if len(flelist) > 0:
                    fle_v = iris.load_cube(str(flelist[0]),xysmallslice)
                    goodup = goodup + 1
###############################################################################


                print flecount

            except IOError:
                print 'easy error'
            else:

                #if goodup > 6: rubix2 = rubix2 + 1
                if goodup > 6 and len(fle_u.coord('pressure').points) == 18:
                    cube_mslp = fle_mslp[11,:,:].extract(xysmallslice)
                    cube_mslp = cube_mslp.collapsed(['latitude','longitude'],iris.analysis.MEAN)
                    q = cube_mslp.data/100.
                    cube_u = fle_u[3,:,:]
                    cube_v = fle_v[3,:,:]
                    cube_u = cube_u.collapsed(['latitude','longitude'], iris.analysis.MEAN)
                    cube_v = cube_v.collapsed(['latitude','longitude'], iris.analysis.MEAN)
                    # What we want now is the direction and speed of the winds
                    # abs = (cube_u**2 + cube_v**2)**0.5

                    cube_T15 = fle_T15[11,:,:].extract(xysmallslice)
                    cube_T15 = cube_T15.collapsed(['latitude','longitude'],iris.analysis.MEAN)
                    cube_q15 = fle_q15[11,:,:].extract(xysmallslice)
                    cube_q15 = cube_q15.collapsed(['latitude','longitude'],iris.analysis.MEAN)
                    if q > 975:
                            xysmallslice_850_upwards = iris.Constraint(pressure = lambda cell: 975. >= cell>=100., longitude = lambda cell: float(llo) <= cell <= float(ulo), latitude = lambda cell: float(lla) <= cell <= float(ula))
                    else:
                            xysmallslice_850_upwards = iris.Constraint(pressure = lambda cell: float(q) >= cell>=100., longitude = lambda cell: float(llo) <= cell <= float(ulo), latitude = lambda cell: float(lla) <= cell <= float(ula))
                    cube_T = fle_T[3,:,:].extract(xysmallslice_850_upwards)
                    cube_q = fle_q[3,:,:].extract(xysmallslice_850_upwards)
                    pressure_shape = cube_T.data.shape[0]
                    # all_cube = cube_T[:,:7,1]
                    # Because we are now using near the surface, we want to make sure that T is realistic
                    T_data = cube_T.data
                    q_data = cube_q.data
                    T_collapsed = np.zeros((T_data.shape[0]),float)
                    q_collapsed = np.zeros((q_data.shape[0]), float)
                    for p in range(0, T_data.shape[0]):
                        for y in range(0, T_data.shape[1]):
                            for x in range(0, T_data.shape[2]):
                                if T_data[p,y,x] < 100:
                                    T_data[p,y,x] = float('nan')
                                    q_data[p,y,x] = float('nan')
                        T_collapsed[p] = np.nanmean(T_data[p,:,:])
                        q_collapsed[p] = np.nanmean(q_data[p,:,:])
                    cube_T = cube_T.collapsed(['latitude','longitude'], iris.analysis.MEAN)
                    cube_q = cube_q.collapsed(['latitude','longitude'], iris.analysis.MEAN)
                    cube_T.data = T_collapsed
                    cube_q.data = q_collapsed
                    cube_pressures = cube_q.copy()
                    pressures = cube_T[:,:].coord('pressure').points
                    pressures = np.ndarray.tolist(pressures)
                    pressures.extend([q])
                    pressures = np.asarray(pressures)
                    fake_T = fle_T[3,:pressure_shape+1,1,1]
                    fake_q = fle_q[3,:pressure_shape+1,1,1]
                    fake_T.data[:pressure_shape] = T_collapsed
                    fake_T.data[pressure_shape] = cube_T15.data
                    fake_q.data[:pressure_shape] = q_collapsed
                    fake_q.data[pressure_shape] = cube_q15.data
                    cube_T = fake_T
                    cube_q = fake_q
    #                cube_RH = cube_q.copy()
    #                cube_dewp = cube_q.copy()
                    cube_pressures = cube_q.copy()
                    Ps = np.zeros((len(pressures)),float)
                    cube_pressures.data = pressures*100
                    if np.sum(all_cube) < 1:
                        all_cube = np.zeros((1, len(pressures), 9), float)
                    else:
                        temp_cube = np.zeros((1, len(pressures), 9), float)

                    if len(pressures) == 18:
                        for p in range(0, len(pressures)):
                            if 710. >= pressures[p] > 690.:
                                RH_650hPa.extend([(0.263 * cube_q.data[p] * cube_pressures.data[p])/2.714**((17.67*(cube_T.data[p] - 273.16))/(cube_T.data[p] - 29.65))])
                            if all_cube.shape[0] == 1:
                                all_cube[0,p,4] = (0.263 * cube_q.data[p] * cube_pressures.data[p])/2.714**((17.67*(cube_T.data[p] - 273.16))/(cube_T.data[p] - 29.65))
                                all_cube[0,p,5] = meteocalc.dew_point(temperature = cube_T[p].data - 273.16, humidity=all_cube[0,p,4])
                            else:
                                temp_cube[0,p,4] = (0.263 * cube_q.data[p] * cube_pressures.data[p])/2.714**((17.67*(cube_T.data[p] - 273.16))/(cube_T.data[p] - 29.65))
                                temp_cube[0,p,5] = meteocalc.dew_point(temperature = cube_T[p].data - 273.16, humidity = temp_cube[0,p,4])
                            if p < len(pressures)-1:
                                if all_cube.shape[0] == 1:
                                    all_cube[0,p,1] = cube_T[p].data*((cube_mslp.data/cube_pressures.data[p])**(1./5.257) - 1)/0.0065
                                else:
                                    temp_cube[0,p,1] = cube_T[p].data*((cube_mslp.data/cube_pressures.data[p])**(1./5.257) - 1)/0.0065

                            else:
                                if all_cube.shape[0] == 1:
                                    all_cube[0,p,1] = 1.5
                                else:
                                    temp_cube[0,p,1] = 1.5

    # So, here we also want to compute CAPE and CIN for each storm







                        if all_cube.shape[0] == 1:
                            all_cube[0,:,0] = cube_pressures.data/100
                            all_cube[0,:,2] = cube_T.data
                            all_cube[0,:,3] = cube_q.data
                            all_cube[0,:,6] = p99
                            all_cube[0,:,2] = all_cube[0,:,2] - 273.16
                            all_cube[0,:,7] = cube_u.data
                            all_cube[0,:,8] = cube_v.data
                            #mydata = dict(zip(('hght','pres','temp','dwpt'),(all_cube[:,1].data[::-1],all_cube[:,0].data[::-1], all_cube[:,2].data[::-1],all_cube[:,5].data[::-1])))
                            print all_cube[0,::-1,0]
                            mydata = dict(zip(('hght','pres','temp','dwpt'),(all_cube[0,::-1,1],all_cube[0,::-1,0], all_cube[0,::-1,2],all_cube[0,::-1,5])))
                            S=sk.Sounding(soundingdata=mydata)
                            parcel = S.get_parcel('mu')
                            P_lcl,P_lfc,P_el,CAPE,CIN=S.get_cape(*parcel)
                            CAPE_stats[cntr,0] = P_lcl
                            CAPE_stats[cntr,1] = P_lfc
                            CAPE_stats[cntr,2] = P_el
                            CAPE_stats[cntr,3] = CAPE
                            CAPE_stats[cntr,4] = CIN
                            CAPE_stats[cntr,5] = storms_to_keep[rw,8]
                            cntr = cntr + 1

                        else:
                            temp_cube[0,:,0] = cube_pressures.data/100
                            temp_cube[0,:,2] = cube_T.data
                            temp_cube[0,:,3] = cube_q.data
                            temp_cube[0,:,6] = p99
                            temp_cube[0,:,2] = temp_cube[0,:,2] - 273.16
                            temp_cube[0,:,7] = cube_u.data
                            temp_cube[0,:,8] = cube_v.data
                            mydata = dict(zip(('hght','pres','temp','dwpt'),(temp_cube[0,::-1,1],temp_cube[0,::-1,0], temp_cube[0,::-1,2],temp_cube[0,::-1,5])))
                            S=sk.Sounding(soundingdata=mydata)
                            parcel = S.get_parcel('mu')
                            P_lcl,P_lfc,P_el,CAPE,CIN=S.get_cape(*parcel)
                            CAPE_stats[cntr,0] = P_lcl
                            CAPE_stats[cntr,1] = P_lfc
                            CAPE_stats[cntr,2] = P_el
                            CAPE_stats[cntr,3] = CAPE
                            CAPE_stats[cntr,4] = CIN
                            CAPE_stats[cntr,5] = storms_to_keep[rw,8]
                            cntr = cntr + 1
                        try:
                        #if all_cube.shape[0] > 1:
                            if temp_cube.shape[1] == all_cube.shape[1]:
                                all_cube = np.concatenate((all_cube, temp_cube), axis = 0)
                        except UnboundLocalError or ValueError:
                            continue
 all_cube = np.average(all_cube, axis = 0)
 np.savetxt('FC_c2c4_storms_midday_stats_for_TEPHI_1800Z_storms.csv', all_cube, delimiter = ',')
 np.savetxt('../csvs/FC_c2c4_1200_CAPE_CIN_1800_STORMS.csv', CAPE_stats, delimiter = ',')
 np.savetxt('../csvs/FC_c2c4_1200_700_hPa_RH_1800_storms.csv', RH_650hPa, delimiter = ',')

 # format for all_cube
 # pressure, height, temp, specific humidity, RH, dewpoint, precip 99th percentile, u winds, v winds
 mydata = dict(zip(('hght','pres','temp','dwpt'),(all_cube[:,1][::-1],all_cube[:,0][::-1], all_cube[:,2][::-1],all_cube[:,5][::-1])))
# S=SkewT.Sounding(soundingdata=mydata)
# S.plot_skewt(color='r')

# S.show()
# parcel = S.get_parcel('mu')
# P_lcl,P_lfc,P_el,CAPE,CIN=S.get_cape(*parcel)
# CAPE_stats_CC[counter,0] = P_lcl
# CAPE_stats_CC[counter,1] = P_lfc
# CAPE_stats_CC[counter,2] = P_el
# CAPE_stats_CC[counter,3] = CAPE
# CAPE_stats_CC[counter,4] = CIN
# CAPE_stats_CC[counter,5] = storms_to_keep[rw,8]
# counter = counter + 1
# print counter
# np.savetxt('CAPE_stats_1200Z_most_unstable_cc_non_monotonic_with_surface_data_1_mm_p99_precip_min_no_midday_rain_c4_storms.csv', CAPE_stats_CC, delimiter = ',')

 print 'SUCCESS!!!', rubix,rubix2, cntr
 P = np.ndarray.tolist(all_cube[:,0][::-1]) * units.hPa
 T = np.ndarray.tolist(all_cube[:,2][::-1]) * units.degC
 Td = np.ndarray.tolist(all_cube[:,5][::-1]) * units.degC
 print P
 print T
 print Td
 plt.clf()

 skew = SkewT()
 skew.plot(all_cube[:,0][::-1], all_cube[:,2][::-1], 'r')
 skew.plot(all_cube[:,0][::-1], all_cube[:,5][::-1],'g')
 skew.plot_barbs(all_cube[:,0][::-1], all_cube[:,7][::-1]*1.94, all_cube[:,8][::-1]*1.94)
 lcl_pressure, lcl_temperature = metcalc.lcl(P[0], T[0], Td[0])
 skew.plot(lcl_pressure, lcl_temperature, 'ko', markerfacecolor='black')

# Calculate full parcel profile and add to plot as black line
 prof = metcalc.parcel_profile(P, T[0], Td[0]).to('degC')
 skew.plot(P, prof, 'k', linewidth=2)

# Shade areas of CAPE and CIN
 skew.shade_cin(P, T, prof)
 skew.shade_cape(P, T, prof)

 skew.plot_dry_adiabats()
 skew.plot_moist_adiabats()
 skew.plot_mixing_lines()
 skew.ax.set_xlim(-40, 50)
 fig = plt.gcf()
 plt.savefig('C4_STORMS_MIDDAY_TEPHIGRAMS_FC.png')
 plt.show()

if "__name__" == "__main__":
   main(xstart,xend,ystart,yend)
