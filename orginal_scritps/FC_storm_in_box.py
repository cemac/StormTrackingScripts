'''
FC version of Rory's orginal script
'''

def main(x1,x2,y1,y2,size_of_storm):

 import numpy as np
 from numpy import genfromtxt as gent
 import os
 import iris
 import glob
# The issue has to be here!
 storm_diurnal = np.zeros((24),float)
 areas = np.zeros((24),float)
 dys = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30']
 good_days = np.zeros((1,13),float)
 cnr = 0
 yyyy = ['1997','1998','1999','2000','2001','2002','2003','2004','2005','2006']
 for yr in yyyy:

#  if yr == '1998':
#	 cube = iris.load_cube('ac144a.pa19970610_17.pp')
#  else:
  cube = iris.load('/nfs/a277/IMPALA/data/4km/a03332_12km/a03332_A1hr_mean_ah261_4km_200012070030-200012072330.nc')

  cube = cube[1]
  lon = cube.coord('longitude').points
  lat = cube.coord('latitude').points
  lons = lon.tolist()
  lats = lat.tolist()

  for x in range(0,len(lons)):
	if lons[x] <= float(x1) <= lons[x+1]:
           leftx = x
        if lons[x] <= float(x2) <= lons[x+1]:
           rightx = x
  for y in range(0,len(lats)):
	if lats[y] <= float(y1) <= lats[y+1]:
		lowy = y
	if lats[y] <= float(y2) <= lats[y+1]:
		highy = y
  for m in range(6,10):
   for d1 in range(0,30):
    for day in range(1,25):
     flelist = glob.glob('/nfs/a299/IMPALA/data/fc/4km/precip_tracking_12km_hourly/'+str(yr)+'/a04203*4km_'+str(yr)+'0'+str(m)+str(dys[d1])+'0030-'+str(yr)+'0'+str(m)+str(dys[d1])+'2330_'+str(day)+'.txt')
     for fle in flelist:
        file =  open(fle,'r')
        lines = file.readlines()
        file.close()
        for line in lines:
         line = line.strip()
         if line.find('box') != -1 and line.find('life') != -1:
           a  = line.find('life')
           b = line.find('box')
           c = line.find('area')
           d = line.find('centroid')
           e = int(line[c+5:d-1])*144.0

           q = line.find('storm')
           f = line[b+4:a-1]
           g = f.find(',')
           llat = float(f[0:g])
           nllat = f[g+1:]
           h = nllat.find(',')
           llon = float(nllat[:h])
           nllon = nllat[h+1:]
           k = nllon.find(',')
           ulat = llat + float(nllon[:k])
           ulon = llon + float(nllon[k+1:])
           StormID = str(line[q+6:c-1])
           meanOLR = line.find('mean')
           minOLR = line.find('min')
           mean_olr = str(line[meanOLR+5:minOLR-1])


#For Niamey, we want have a very small location. So we want to track
# any storm that goes over Niamey.
# this means the storm spans farther to the west, and east, north and south

           if e >= size_of_storm: #We want storms of a certain size
            center_lon = (float(lons[int(llon-1)]) + float(lons[int(ulon-1)])) /2.0
            center_lat = (float(lats[int(llat-1)]) + float(lats[int(ulat-1)]))/2.0
            if float(lons[int(leftx)]) <= center_lon <= float(lons[int(rightx)]) and float(lats[int(lowy)]) <= center_lat <= float(lats[int(highy)]):
                cnr = cnr + 1
                if good_days[0,0] < 1:
                  good_days[0,0] = StormID
                  good_days[0,1] = yr
                  good_days[0,2] = m
                  good_days[0,3] = dys[d1]
                  good_days[0,4] = day
                  good_days[0,5] = lons[int(llon-1)]
                  good_days[0,6] = lons[int(ulon-1)]
                  good_days[0,7] = lats[int(llat-1)]
                  good_days[0,8] =  lats[int(ulat-1)]
                  good_days[0,9] = center_lon
                  good_days[0,10] = center_lat
                  good_days[0,11] = e
		  good_days[0,12] = mean_olr
                else:
                  temp = np.zeros((1,13), float)
                  temp[0,0] = StormID
                  temp[0,1] = yr
                  temp[0,2] = m
                  temp[0,3] = dys[d1]
                  temp[0,4] = day
                  temp[0,5] = lons[int(llon-1)]
                  temp[0,6] = lons[int(ulon-1)]
                  temp[0,7] = lats[int(llat-1)]
                  temp[0,8] =  lats[int(ulat-1)]
                  temp[0,9] = center_lon
                  temp[0,10] = center_lat
                  temp[0,11] = e
		  temp[0,12] = mean_olr

                  good_days = np.concatenate((good_days,temp),axis = 0)



 for x in range(0,24):
     if areas[x] >1.0:
       areas[x] = areas[x]/float(storm_diurnal[x])
 
 np.savetxt('CP4_FC_precip_storms_over_box_area_'+str(size_of_storm)+'_lons_'+str(x1)+'_'+str(x2)+'_lat_'+str(y1)+'_'+str(y2)+'.csv', good_days[:], delimiter = ',')




if "__name__" == "__main__":
   main(x1,x2,y1,y2)
