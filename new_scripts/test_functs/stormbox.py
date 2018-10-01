#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 13:47:32 2018

@author: earhbu
"""

import glob
import pandas as pd

x1, x2 = [345, 375]
y1, y2 = [10, 18]
size = 5000

fname = ('/nfs/a277/IMPALA/data/4km/a03332_12km/a03332_A1hr_mean_' +
         'ah261_4km_200012070030-200012072330.nc')
# file root for generating file list
froot = '/nfs/a277/IMPALA/data/4km/precip_tracking_12km_hourly/'
df =pd.DataFrame()
df2 =pd.DataFrame(columns=['file'],index=[range(0, len(df))])
df['file'] = (glob.glob(froot+'*/a04203*4km*.txt'))
for row in df.itertuples():
    if row.file[90:92] in [str(x).zfill(2) for x in range(6, 10)]:
        df2.loc[row[0]] = 0
        df2['file'].loc[row[0]] = row.file
df = df2.reset_index(drop=True)