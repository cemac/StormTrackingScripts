'''
New data miner script based off:
dataminer_1800z_all_stats_CAPE.py
find the difference between c30404 and c30403 for TCW
and the shear values
'''
# Modules required
from os.path import expanduser
import numpy as np
from dataminer_functions import dm_functions
import pandas as pd

# Variables
size_of_storm = 5000
x1, x2 = [345, 375]
y1, y2 = [10, 18]
fcorcc = 0
csvroot = ('testfc', ' testcc')
dataroot = ('/nfs/a299/IMPALA/data/fc/4km/', '/nfs/a277/IMPALA/data/4km/')
stormhome = expanduser("~")+'/AMMA2050/new_scripts'
run = ('teststorms_over_box_area', 'fc_storms_to_keep_area_',
       'cc_storms_to_keep_area_')
csvname = (stormhome + '/' + run[0] + str(size_of_storm) +
           '_lons_' + str(x1) + '_' + str(x2) + '_lat_' + str(y1) + '_' +
           str(y2) + '.csv')
# Generating file list
dmf = dm_functions(dataroot[fcorcc])
# find storms
try:
    storms_to_keep = pd.read_csv(csvname, sep=',')
except IOError:
    print('Generating csv file of storms to keep ...')
    # if its not there we'll have to generate it from CP4
    storms_to_keep = dmf.gen_storms_to_keep(csvname)
    np.savetxt(csvname, storms_to_keep[:, :], delimiter=',')
dmf.gen_var_csvs(csvroot[fcorcc], storms_to_keep, CAPE='Y', TEPHI='Y')
