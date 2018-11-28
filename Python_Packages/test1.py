from os.path import expanduser
import numpy as np
import time
import pandas as pd
import StormScriptsPy3 as SSP3


# Box dimensions
x1, x2 = [345, 375]
y1, y2 = [10, 18]
# run id
idstring = ['fc_test', 'cc_test']
size_of_storm = 5000
# corresponding data_dirs
dataroot = ('/nfs/a299/IMPALA/data/fc/4km/', '/nfs/a277/IMPALA/data/4km/')
# Where to store the generated files
stormhome = expanduser("~")+'/AMMA2050/Python_Packages/'
'''
start_time = time.time()

c = SSP3.S_Box.StormInBox(x1, x2, y1, y2, size_of_storm, idstring[0])
c.gen_storm_box_csv()

print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()

c = SSP3.S_Box.StormInBox(x1, x2, y1, y2, size_of_storm, idstring[1])
c.gen_storm_box_csv()

print("--- %s seconds ---" % (time.time() - start_time))
'''

# FC
start_time = time.time()

dmf = SSP3.dm_functions(dataroot[0], CAPE='Y', TEPHI='Y')
csvname = stormhome + idstring[0] + 'storms_over_box_area5000_lons_345_375_lat_10_18.csv'
storms_to_keep = pd.read_csv(csvname, sep=',')
dmf.genvarscsv(idstring[0], storms_to_keep)

print("--- %s seconds ---" % (time.time() - start_time))

# CC
start_time = time.time()

dmf = SSP3.dm_functions(dataroot[1], CAPE='Y', TEPHI='Y')
csvname = stormhome + idstring[1] + 'storms_over_box_area5000_lons_345_375_lat_10_18.csv'
storms_to_keep = pd.read_csv(csvname, sep=',')
dmf.genvarscsv(idstring[1], storms_to_keep)

print("--- %s seconds ---" % (time.time() - start_time))
