#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 13:45:12 2018

@author: earhbu
"""
# modules required
import os
import glob
import pandas as pd
import numpy as np


# Variables passed in
dataroot = '/nfs/a299/IMPALA/data/fc/4km/'
storminfo = '19990601'

## func
vars = pd.read_csv('../vars.csv')
varcodes = vars['code']
foldername = varcodes
df = pd.DataFrame(columns=['file','codes'], index=[range(0, len(foldername))])
for i in range(0, len(foldername)):
            try:
                df.loc[i] = [glob.glob(str(dataroot) + str(foldername[i]) + '/' +
                                      str(foldername[i]) + '*_' + str(storminfo) + '*-*.nc')[0]
                             , foldername[i]]
            except IndexError:
                pass
flist = df
varnames=vars['varname']
## func2
if varcodes is not None:
    foldername = varcodes
else:
    ds = pd.Series(os.listdir(dataroot))
    foldername = ds[ds.str.len() == 6].reset_index(drop=True)
df = pd.DataFrame(columns=['file', 'codes', 'varname'])
for i in range(0, len(foldername)):
    try:
        df.loc[i] = [glob.glob(str(dataroot) + str(foldername[i])
                                       + '/' + str(foldername[i]) + '*_' +
                                       str(storminfo) + '*-*.nc')[0],
                             foldername[i], varnames[i]]
    except IndexError:
        pass
