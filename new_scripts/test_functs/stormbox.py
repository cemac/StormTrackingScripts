#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 13:47:32 2018
testing code
"""

import glob
import pandas as pd
import iris

x1, x2 = [345, 375]
y1, y2 = [10, 18]
size_of_storm = 5000
varslist = ['stormid', 'year', 'month', 'day', 'hour', 'llon',
            'ulon', 'llat', 'ulat', 'centlon', 'centlat', 'area',
            'mean_olr']

fname = ('/nfs/a277/IMPALA/data/4km/a03332_12km/a03332_A1hr_mean_' +
         'ah261_4km_200012070030-200012072330.nc')
# file root for generating file list
froot = '/nfs/a277/IMPALA/data/4km/precip_tracking_12km_hourly/'
df =pd.DataFrame()
df2 =pd.DataFrame(columns=['file'],index=[range(0, len(df))])
df['file'] = (glob.glob(froot+'*/a04203*4km*.txt'))
cube = iris.load(fname)[1]
lon = cube.coord('longitude').points
lat = cube.coord('latitude').points
for row in df.itertuples():
    if row.file[90:92] in [str(x).zfill(2) for x in range(6, 10)]:
        df2.loc[row[0]] = 0
        df2['file'].loc[row[0]] = row.file
df = df2.reset_index(drop=True)
cols = ['storm', 'no', 'area', 'centroid', 'box', 'life', 'u', 'v', 'mean', 'min',
        'max', 'accreted', 'parent', 'child', 'cell']
for row in df.itertuples():
    vars = pd.read_csv(row.file, names=cols,  header=None, delim_whitespace=True)
    var2 = vars[pd.notnull(vars['mean'])]
    stormsdf = pd.DataFrame(columns=varslist)
    size = var2.area.str[5::]
    var2['area'] = pd.to_numeric(size)*144
    storms = var2[var2.area >= size_of_storm].reset_index(drop=True)
    storms[['centlat','centlon']] = storms['centroid'].str.split(',',expand=True)
    for row in storms.itertuples():
        storms['centlat'].loc[row[0]] = lat[int(pd.to_numeric(row.centlat[9::]))]
        storms['centlon'].loc[row[0]] = lon[int(pd.to_numeric(row.centlon))]
    storms = storms[storms.centlon <= x2].reset_index(drop=True)
    storms = storms[storms.centlon >= x1].reset_index(drop=True)
    storms = storms[storms.centlat <= y2].reset_index(drop=True)
    storms = storms[storms.centlat >= y1].reset_index(drop=True)
    # Any in this file?
    if len(storms)==0:
        continue
    print('found one')
    stormsdf = pd.DataFrame(columns=varslist)
    stormsdf.stormid = storms.no
    stormsdf.year = pd.to_datetime(df.file[10][86:98]).year
    stormsdf.month = pd.to_datetime(df.file[10][86:98]).month
    stormsdf.day = pd.to_datetime(df.file[10][86:98]).day
    stormsdf.hour = pd.to_datetime(df.file[10][86:98]).hour
    stormsdf.area = storms.area
    stormsdf.mean_olr = storms['mean'].str[5::]
    stormsdf.ulon = storms.u.str[2::]
    stormsdf.ulat = storms.v.str[2::]
    break
    
