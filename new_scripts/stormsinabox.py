'''
Storm_in_a_box.py based off of Rory's script
'''
import glob
import numpy as np
import iris


class storminbox(object):
    '''Description
       Stage 1: currently a suit of functions for finding information on
       storms in region and Generating cvs files of that information.
    '''
    def __init__(self, x1, x2, y1, y2, size_of_storm):

        fname = ('/nfs/a277/IMPALA/data/4km/a03332_12km/a03332_A1hr_mean_' +
                 'ah261_4km_200012070030-200012072330.nc')
        # file root for generating file list
        froot = '/nfs/a277/IMPALA/data/4km/precip_tracking_12km_hourly/'
        df = pd.DataFrame()
        df2 = pd.DataFrame(columns=['file'], index=[range(0, len(df))])
        df['file'] = (glob.glob(froot+'*/a04203*4km*.txt'))
        for row in df.itertuples():
            if row.file[90:92] in [str(x).zfill(2) for x in range(6, 10)]:
                df2.loc[row[0]] = 0
                df2['file'].loc[row[0]] = row.file
        df = df2.reset_index(drop=True)
                    for fle in flelist:
                        f = open(fle, 'r')
                        lines = f.readlines()
                        f.close()
                        for line in lines:
                            line = line.strip()
                            if line.find('box') != -1 and line.find('life') != -1:
                                a = line.find('life')
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

                                # For Niamey, we want have a very small location.
                                # So we want to track any storm that goes over
                                # Niamey. This means the storm spans farther to the
                                # west, and east, north and south
                                # We want storms of a certain size
                                if e >= size_of_storm:
                                    center_lon = (float(lons[int(llon-1)]) +
                                                  float(lons[int(ulon-1)])) / 2.0
                                    center_lat = (float(lats[int(llat-1)]) +
                                                  float(lats[int(ulat-1)])) / 2.0
                                if (float(lons[int(leftx)]) <= center_lon <=
                                        float(lons[int(rightx)]) and
                                        float(lats[int(lowy)]) <= center_lat <=
                                        float(lats[int(highy)])):
                                    cnr += 1
                                    if good_days[0, 0] < 1:
                                        good_days[0, 0] = StormID
                                        good_days[0, 1] = yr
                                        good_days[0, 2] = m
                                        good_days[0, 3] = dys[d1]
                                        good_days[0, 4] = day
                                        good_days[0, 5] = lons[int(llon-1)]
                                        good_days[0, 6] = lons[int(ulon-1)]
                                        good_days[0, 7] = lats[int(llat-1)]
                                        good_days[0, 8] = lats[int(ulat-1)]
                                        good_days[0, 9] = center_lon
                                        good_days[0, 10] = center_lat
                                        good_days[0, 11] = e
                                        good_days[0, 12] = mean_olr
                                    else:
                                        temp = np.zeros((1, 13), float)
                                        temp[0, 0] = StormID
                                        temp[0, 1] = yr
                                        temp[0, 2] = m
                                        temp[0, 3] = dys[d1]
                                        temp[0, 4] = day
                                        temp[0, 5] = lons[int(llon-1)]
                                        temp[0, 6] = lons[int(ulon-1)]
                                        temp[0, 7] = lats[int(llat-1)]
                                        temp[0, 8] = lats[int(ulat-1)]
                                        temp[0, 9] = center_lon
                                        temp[0, 10] = center_lat
                                        temp[0, 11] = e
                                        temp[0, 12] = mean_olr
                                        good_days = np.concatenate((good_days,
                                                                    temp), axis=0)

        for x in range(0, 24):
            if areas[x] > 1.0:
                areas[x] = areas[x]/float(storm_diurnal[x])

        np.savetxt('CP4_CC_precip_storms_over_box_area_' + str(size_of_storm) +
                   '_lons_' + str(x1) + '_' + str(x2) + '_lat_' + str(y1) + '_' +
                   str(y2)+'.csv', good_days[:], delimiter=',')
