# -*- coding: utf-8 -*-
"""
This script calls and runs Rory's scripts in series to Reproduce his fgures
Storm in a box:
x1,x2,y1,y2 = lats and longs of regions
size of storm = m?
requires same numbers for each
"""
# import function 'main from first script'
import os
#from FC_storm_in_box import main
#from dataminer_1800Z_all_stats_and_CAPE import main as main2
from paper_fc_1200_CAPE_1800_storm import main as main3
from c2c4_storms_rainfall_correlations_for_paper import main as main4
# Storm in a box writes a file:
# CP4_CC_precip_storms_over_box_area_size_lons_x1_x2_lat_y1_y2.csv
# FC Storm in a box writes a file:
# CP4_FC_precip_storms_over_box_area_size_lons_x1_x2_lat_y1_y2.csv
# dataminer_1800Z_all_stats_and_CAPE
# calls for the FC version
x = [345, 375]
y = [10, 18]
size = 5000
#main(x[0], x[1], y[0], y[1], size)
# main(x1,x2,y1,y2,size_of_storm)
#os.chdir('/nfs/a277/earhbu/StormTrackingScripts/dataminer')
#main2(x[0], x[1], y[0], y[1], size)
# He does a check and if there are a number of failures he prints problem?
# Produces a large number of problems...
#os.chdir('/nfs/a277/earhbu/StormTrackingScripts/CAPE_pp_files_extreme_precip')
#main3(x[0], x[1], y[0], y[1], size)
os.chdir('/nfs/a277/earhbu/StormTrackingScripts')
main4(x[0], x[1], y[0], y[1], size)
