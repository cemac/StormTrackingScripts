import pstromsinabox
from os.path import expanduser
import numpy as np
import time
from numpy import genfromtxt as gent
from Pdataminer_funcsv2 import dm_functions
import pandas as pd

start_time = time.time()
x1, x2 = [345, 375]
y1, y2 = [10, 18]
idstring = 'ptest'
size_of_storm = 5000
c = pstromsinabox.storminbox(x1, x2, y1, y2, size_of_storm, idstring)
c.genstormboxcsv()
print("--- %s seconds ---" % (time.time() - start_time))
start_time1 = time.time()
fcorcc = 0
csvroot = ('ptestfc', ' ptestcc')
dataroot = ('/nfs/a299/IMPALA/data/fc/4km/', '/nfs/a277/IMPALA/data/4km/')
stormhome = expanduser("~")+'/AMMA2050/Parrallel_Scripts'
run = ('fc_storms_over_box_area', 'fc_storms_to_keep_area_',
       'cc_storms_to_keep_area_')
csvname = (stormhome + '/' + run[0] + str(size_of_storm) +
           '_lons_' + str(x1) + '_' + str(x2) + '_lat_' + str(y1) + '_' +
           str(y2) + '.csv')
# Generating file list
dmf = dm_functions(dataroot[fcorcc], CAPE='Y', TEPHI='Y')
print("--- %s seconds ---" % (time.time() - start_time1))
print("--- %s seconds ---" % (time.time() - start_time))
