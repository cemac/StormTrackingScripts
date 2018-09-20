'''
New data miner script based off:
dataminer_1800z_all_stats_CAPE.py
find the difference between c30404 and c30403 for TCW
and the shear values
STAGE 1 : remove requirement for folder structure
'''
# Modules required
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
from dataminer_functions import dmfuctions
from os.path import expanduser
# Variables
size_of_storm = 5000
x1, x2 = [345, 375]
y1, y2 = [10, 18]
# Written a module dmfuctions with suite of funtions all using this information
dmf = dmfuctions(x1, x2, y1, y2, size_of_storm)
fname = ('/nfs/a65/eejac/VERA/IMPALA/olr_tracking_12km/stats/' +
         'WAfrica_Rory/*/*.txt')
stormhome = expanduser("~")+'AMMA2050'
csvname = (home + '/fc_storms_to_keep_area_' + str(size_of_storm) +
           '_longitudes_' + str(x1) + '_' + str(x2) + '_' + str(y1) + '_' +
           str(y2) + '_1800Z.csv')
altcsvname = (home + '/CP4_FC_precip_storms_over_box_area_' +
              str(size_of_storm) + '_lons_' + str(x1) + '_' + str(x2) +
              '_lat_' + str(y1) + '_' + str(y2) + '.csv')
# Generating file list
C4_CC_list = []
C4_FC_list = []
flelist = glob.glob(fname)
for element in range(0, len(flelist)):
    fle = pd.read_fwf(flelist[element], header=None)
    datu = np.asarray(fle)
    for rw in range(0, datu.shape[0]):
        if datu[rw, -1] == '4' or datu[rw, -1] == '2':
            if 'fc' in flelist[element]:
                C4_FC_list.extend([datu[rw, 0]])
            else:
                C4_CC_list.extend([datu[rw, 0]])
# find storms
try:
    # try fc file format
    storms_to_keep = gent(csvname, delimiter=',')
    print storms_to_keep.shape
except IOError:
    print('Generating csv file ...')
    # if its not there we'll have to generate it from CP4
    storms_to_keep = dfm.gen_storms_to_keep(altcsvname)
np.savetxt(csvname, storms_to_keep[:, :], delimiter=',')
Stormnum = 0
GOODUN = 0
keepun = 0
olrkeepers = []
list_of_storms = collections.Counter(storms_to_keep[:, 8])
try:
    guaranteed_failsafe = gent('nofile.csv', delimiter=',')
except IOError:
    dmf.gen_var_csv()
