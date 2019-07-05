"""Example Script

    How to generate storm data
   :copyright: © 2018 University of Leeds.
   :license: BSD-2 Clause.

To use:

    python Generating_files.py

.. CEMAC_stomtracking:
   https://github.com/cemac/StormTrackingScripts
"""

from os.path import expanduser
import numpy as np
import time
import pandas as pd
import StormScriptsPy3 as SSP3

# Specify Box dimensions and size of storm
x1, x2 = [345, 375]
y1, y2 = [10, 18]
size_of_storm = 5000
# run ids future and current
idstring = ['fc_', 'cc_']
# Where to store the generated files
stormhome = expanduser("~")+'/StormTrackingScripts/'
#
# future climate storms
#
# Initialise the Storm
c = SSP3.S_Box.StormInBox(x1, x2, y1, y2, size_of_storm, idstring[0], run='fc')
c.gen_storm_box_csv()
#
# current climate storms
#
# run defaults to cc so we don't need to specify
c = SSP3.S_Box.StormInBox(x1, x2, y1, y2, size_of_storm, idstring[1])
c.gen_storm_box_csv()
#
# NB you could specify a diferent and set the root variable
# (i.e. '/nfs/a299/IMPALA/data/Crazy_NEW_SCENARIO/4km/')
#
# NOW MINE the data on the storms
#
# set corresponding data_dirs
dataroot = ('/nfs/a299/IMPALA/data/fc/4km/', '/nfs/a277/IMPALA/data/4km/')
#
# future climate
#
# Initialise the data miner
dmf = SSP3.dm_functions(dataroot[0], CAPE='Y', TEPHI='Y')
# Create a Pandas data frame of the file you generated
csvname = stormhome + idstring[0] + 'storms_over_box_area5000_lons_345_375_lat_10_18.csv'
storms_to_keep = pd.read_csv(csvname, sep=',')
# Mine the Strom data - this will take some time.
dmf.genvarscsv(idstring[0], storms_to_keep)
#
# current climate
#
# Initialise the data miner
dmf = SSP3.dm_functions(dataroot[1], CAPE='Y', TEPHI='Y')
# Create a Pandas data frame of the file you generated
csvname = stormhome + idstring[1] + 'storms_over_box_area5000_lons_345_375_lat_10_18.csv'
storms_to_keep = pd.read_csv(csvname, sep=',')
# Mine the Strom data - this will take some time.
dmf.genvarscsv(idstring[1], storms_to_keep)
#
#
# After a few hours you will have your mined data
# (depends on number of cores available)
