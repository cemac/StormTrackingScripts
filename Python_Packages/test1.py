from os.path import expanduser
import numpy as np
import time
import pandas as pd
import StormScriptsPy3 as SSP3

#start_time = time.time()
x1, x2 = [345, 375]
y1, y2 = [10, 18]
idstring = 'ptest'
size_of_storm = 5000
#c = pstromsinabox.StormInBox(x1, x2, y1, y2, size_of_storm, idstring)
#c.genstormboxcsv()
#print("--- %s seconds ---" % (time.time() - start_time))
start_time1 = time.time()
fcorcc = 0
csvroot = ('ptestfc', ' ptestcc')
dataroot = ('/nfs/a299/IMPALA/data/fc/4km/', '/nfs/a277/IMPALA/data/4km/')
stormhome = expanduser("~")+'/AMMA2050/Parrallel_Scripts'
csvname = stormhome+'/1000storms_over_box_area5000_lons_345_375_lat_10_18.csv'
# Generating file list
dmf = SSP3.dm_functions(dataroot[1], CAPE='Y', TEPHI='Y')
storms_to_keep = pd.read_csv(csvname, sep=',')
dmf.genvarscsv(csvroot[fcorcc], storms_to_keep)
print("--- %s seconds ---" % (time.time() - start_time1))
