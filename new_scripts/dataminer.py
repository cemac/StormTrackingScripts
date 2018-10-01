'''
New data miner script based off:
dataminer_1800z_all_stats_CAPE.py
find the difference between c30404 and c30403 for TCW
and the shear values
STAGE 1 : remove requirement for folder structure
'''
# Modules required
from os.path import expanduser
import numpy as np
from numpy import genfromtxt as gent
from dataminer_functions import dmfuctions

# Variables
size_of_storm = 5000
x1, x2 = [345, 375]
y1, y2 = [10, 18]
csvroot = ('testfc', ' testcc')
dataroot = ('/nfs/a299/IMPALA/data/fc/4km/', '/nfs/a277/IMPALA/data/4km/')
stormhome = expanduser("~")+'/AMMA2050'
csvname = (stormhome + '/fc_storms_to_keep_area_' + str(size_of_storm) +
           '_longitudes_' + str(x1) + '_' + str(x2) + '_' + str(y1) + '_' +
           str(y2) + '_1800Z.csv')
altcsvname = (stormhome + '/CP4_FC_precip_storms_over_box_area_' +
              str(size_of_storm) + '_lons_' + str(x1) + '_' + str(x2) +
              '_lat_' + str(y1) + '_' + str(y2) + '.csv')
# Generating file list
dmf = dmfuctions(x1, x2, y1, y2, size_of_storm, dataroot[0])
# find storms
try:
    storms_to_keep = gent(csvname, delimiter=',')
except IOError:
    print('Generating csv file of storms to keep ...')
    # if its not there we'll have to generate it from CP4
    storms_to_keep = dmf.gen_storms_to_keep(csvname)
    np.savetxt(csvname, storms_to_keep[:, :], delimiter=',')
dmf.gen_var_csvs(csvroot[0], storms_to_keep)
