# right, so in this file we are setting up all the information needed
# for our CAPE calculations.
# Looking at the literature, we need pressure, temperature, dewpoint temperature, height, and specific humidity (g/kg)
# for all the intense storms. We will use the pressure values 850 hPa upwards in order to remove the issues with 925 hPa T values for some storms.
import iris
import numpy as np
from numpy import genfromtxt as gent
import glob
import meteocalc
from skewt import SkewT as sk
import pandas as pd


def main(xstart, xend, ystart, yend, size_of_storm):

 C4_FC_list = []
 C4_CC_list = []
 flelist = glob.glob('/nfs/a65/eejac/VERA/IMPALA/olr_tracking_12km/stats/WAfrica_Rory/*/*.txt')
 for element in range(0, len(flelist)):
        if 'fc' in flelist[element]:
            print [element]
                fle = pd.read_fwf(flelist[element])
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


 CAPE_stats = np.zeros((len(np.ndarray.tolist(CC_storm_IDs)),6),float)
 try:
     storms_to_keep = gent('../OLR_12km_storms_to_keep_area_'+str(size_of_storm)+'_longitudes_'+str(xstart)+'_'+str(xend)+'_'+str(ystart)+'_'+str(yend)+'_1800Z.csv', delimiter = ',')
     print storms_to_keep.shape
 except IOError:
  dates = gent('../CP4_CC_precip_storms_over_box_area_'+str(size_of_storm)+'_lons_'+str(xstart)+'_'+str(xend)+'_lat_'+str(ystart)+'_'+str(yend)+'.csv', delimiter = ',', names = ['stormid','year','month','day','hour','llon','ulon','llat','ulat','centlon','centlat','area','mean_olr'])
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

 np.savetxt('../OLR_12km_storms_to_keep_area_'+str(size_of_storm)+'_longitudes_'+str(xstart)+'_'+str(xend)+'_'+str(ystart)+'_'+str(yend)+'_1800Z.csv',storms_to_keep[:,:], delimiter = ',')


 all_cube = np.zeros((1,15,6), float)
 RH_650hPa = []
 for rw in range(0, storms_to_keep.shape[0]):
  if float(storms_to_keep[rw,0]) < 100:
   continue
  elif storms_to_keep[rw,8] in CC_storm_IDs:
    a = str(storms_to_keep[rw,0])
    b = str(storms_to_keep[rw,1])
    c = str(storms_to_keep[rw,2])
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
            flelist = glob.glob('/nfs/a277/IMPALA/data/4km/a04203/a04203_A1hr_mean_*4km_'+str(a[:4])+''+str(b[:2])+''+str(c[:2])+'0030-'+str(a[:4])+''+str(b[:2])+''+str(c[:2])+'2330.nc')
            fle_precip = iris.load_cube(str(flelist[0]))
    except IOError:
        print 'No precip file'
    else:
            cube_precip = fle_precip.extract(xysmallslice)
            p99 = cube_precip[17,:,:].collapsed(['latitude','longitude'], iris.analysis.PERCENTILE, percent = 99).data*3600.
            cube_midday_precip = cube_precip[11:15,:,:]
            cube_midday_precip = cube_midday_precip.collapsed(['time','latitude','longitude'], iris.analysis.MEAN).data
            cube_precip = cube_precip[17,:,:].collapsed(['latitude','longitude'], iris.analysis.MEAN).data*3600.
            print 'extreme storm'
            goodup = 0
            try:

                flelist = glob.glob('/nfs/a277/IMPALA/data/4km/f30204/f30204_A3hr_inst_*4km_'+str(a[:4])+''+str(b[:2])+''+str(c[:2])+'0300-'+str(a[:4])+''+str(b[:2])+'*0000.nc')
                if len(flelist) > 0:
                    fle_T = iris.load_cube(str(flelist[0]))
                    goodup = goodup + 1
                    flelist = glob.glob('/nfs/a277/IMPALA/data/4km/f30205/f30205_A3hr_inst_*4km_'+str(a[:4])+''+str(b[:2])+''+str(c[:2])+'0300-'+str(a[:4])+''+str(b[:2])+'*0000.nc')
                if len(flelist) > 0:
                    goodup = goodup + 1
                    fle_q = iris.load_cube(str(flelist[0]))
                    flelist = glob.glob('/nfs/a277/IMPALA/data/4km/c00409/c00409_A1hr_inst_*4km_'+str(a[:4])+''+str(b[:2])+''+str(c[:2])+'0100-'+str(a[:4])+''+str(b[:2])+'*0000.nc')
                if len(flelist) > 0:
                    goodup = goodup + 1
                    fle_mslp = iris.load_cube(str(flelist[0]))
                    flelist = glob.glob('/nfs/a277/IMPALA/data/4km/c03236/c03236_A1hr_inst_*4km_'+str(a[:4])+''+str(b[:2])+''+str(c[:2])+'0100-'+str(a[:4])+''+str(b[:2])+'*0000.nc')
                if len(flelist) > 0:
                    goodup = goodup + 1
                    fle_T15 = iris.load_cube(str(flelist[0]))
                    flelist = glob.glob('/nfs/a277/IMPALA/data/4km/c03237/c03237_A1hr_inst_*4km_'+str(a[:4])+''+str(b[:2])+''+str(c[:2])+'0100-'+str(a[:4])+''+str(b[:2])+'*0000.nc')
                if len(flelist) > 0:
                    goodup = goodup + 1
                    fle_q15 = iris.load_cube(str(flelist[0]))

###################################################################################

                    flelist = glob.glob('/nfs/a277/IMPALA/data/4km/f30201/f30201_A3hr_inst_*4km_'+str(a[:4])+''+str(b[:2])+''+str(c[:2])+'0300-'+str(a[:4])+''+str(b[:2])+'*0000.nc')
                if len(flelist) > 0:
                    fle_u = iris.load_cube(str(flelist[0]),xysmallslice)
                    goodup = goodup + 1
                    flelist = glob.glob('/nfs/a277/IMPALA/data/4km/f30202/f30202_A3hr_inst_*4km_'+str(a[:4])+''+str(b[:2])+''+str(c[:2])+'0300-'+str(a[:4])+''+str(b[:2])+'*0000.nc')
                if len(flelist) > 0:
                    fle_v = iris.load_cube(str(flelist[0]),xysmallslice)
                    goodup = goodup + 1
###############################################################################

            except IOError:
                print 'easy error'
            else:
                if goodup > 6 and len(fle_u.coord('pressure').points) == 18:
                    cube_mslp = fle_mslp[11,:,:].extract(xysmallslice)
                    cube_mslp = cube_mslp.collapsed(['latitude','longitude'],iris.analysis.MEAN)
                    print cube_mslp.data
                    q = cube_mslp.data/100.
                    cube_u = fle_u[3,:,:]
                    cube_v = fle_v[3,:,:]
                    cube_u = cube_u.collapsed(['latitude','longitude'], iris.analysis.MEAN)
                    cube_v = cube_v.collapsed(['latitude','longitude'], iris.analysis.MEAN)
                    # What we want now is the direction and speed of the winds
                    #                wind_abs = (cube_u**2 + cube_v**2)**0.5

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
                    #all_cube = cube_T[:,:7,1]
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
                                all_cube[0,p,5] = meteocalc.dew_point(temperature = cube_T[p].data - 273.16, humidity = all_cube[0,p,4])
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
                        print cube_pressures.data

                        if all_cube.shape[0] == 1:
                            all_cube[0,:,0] = cube_pressures.data/100
                            all_cube[0,:,2] = cube_T.data
                            all_cube[0,:,3] = cube_q.data
                            all_cube[0,:,6] = p99
                            all_cube[0,:,2] = all_cube[0,:,2] - 273.16
                            all_cube[0,:,7] = cube_u.data
                            all_cube[0,:,8] = cube_v.data
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
                            print all_cube[0,::-1,1], CAPE
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
                            print temp_cube[0,::-1,1], CAPE
                            CAPE_stats[cntr,5] = storms_to_keep[rw,8]
                            cntr = cntr + 1
                        try:
                        # if all_cube.shape[0] > 1:
                            if temp_cube.shape[1] == all_cube.shape[1]:
                                all_cube = np.concatenate((all_cube, temp_cube), axis = 0)
                        except UnboundLocalError or ValueError:
                            continue
 all_cube = np.average(all_cube, axis = 0)
 np.savetxt('CC_c2c4_storms_midday_stats_for_TEPHI_1800Z_storms.csv', all_cube, delimiter = ',')
 np.savetxt('../csvs/CC_c2c4_1200_CAPE_CIN_1800_STORMS.csv', CAPE_stats, delimiter = ',')
 np.savetxt('../csvs/CC_c2c4_1200_700_hPa_RH_1800_storms.csv', RH_650hPa, delimiter = ',')
 print 'SUCCESS', cntr
