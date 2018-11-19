# -*- coding: utf-8 -*-
"""Paralell functions


This module was developed by CEMAC as part of the AMAMA 2050 Project. A lot
of the processing invloved in this work relies on searching through thousands
of files and extracting variables. Perfect for multithreading

Example:
    To use::
        module load Pfuncts
        Pfuncts.parallelize_dataframe(df, func, nice)
Attributes:
    varslist(list): List of vairables required in dataminer
    fname(str): File to extract lat and lons
    froot(str): Root folder of data

.. CEMAC_stomtracking:
   https://github.com/cemac/StormTrackingScripts
"""

import multiprocessing
import pandas as pd
import numpy as np
from numba import autojit
import iris


def _pickle_method(m):
    '''Taken from
    https://laszukdawid.com/2017/12/13/multiprocessing-in-python-all-about-pickling/
    multiprocessing with in a class requires some adaptation to pickling.
    Circumnavigates GIL lock in pickles - which allows objects to cominicate
    with each other. Outside of a class there is no pickling.
    '''
    class_self = m.im_class if m.im_self is None else m.im_self
    return getattr, (class_self, m.im_func.func_name)


def parallelize_dataframe(df, func, nice):
    '''parallelize a data frame

    Attributes:
        df: dataframe
        func: function
        nice: niceness 1/nice share of machince

    Returns:
        df: DataFrame chucked to different processors
    '''
    nprocs = int(multiprocessing.cpu_count()/nice)
    df_split = np.array_split(df, nprocs)
    pool = multiprocessing.Pool(nprocs)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def yes_or_no(question):
    reply = str(input(question+' (y/n): ')).lower().strip()
    if reply[0] == 'y':
        answ = True
    elif reply[0] == 'n':
        answ = False
    else:
        print("You did not enter one of 'y' or 'n'. Assumed 'n'.")
        answ = False
    return answ


def genslice(latlons, n1=None, n2=None):
    '''Description:
        Extrac iris cube slices of a variable
       Attributes:
        llon: lower longitude
        llat: lower latitude
        ulat: upper latitude
        ulon: upper longitude
        n1: pressure low
        n2: pressure high
    '''
    if n1 is None and n2 is None:
        xysmallslice = iris.Constraint(longitude=lambda cell: float(llon)
                                       <= cell <= float(ulon),
                                       latitude=lambda cell: float(llat) <=
                                       cell <= float(ulat))
    elif n1 is not None and n2 is None:
        xysmallslice = iris.Constraint(pressure=lambda cell: n1 ==
                                       cell, longitude=lambda cell:
                                       float(llon) <= cell <= float(ulon),
                                       latitude=lambda cell: float(llat)
                                       <= cell <= float(ulat))
    elif n1 == 500 and n2 == 800:
        xysmallslice = iris.Constraint(pressure=lambda cell: n1 <= cell <=
                                       n2 or cell == 60, longitude=lambda cell:
                                       float(llon) <= cell <= float(ulon),
                                       latitude=lambda cell: float(llat) <=
                                       cell <= float(ulat))
    else:
        xysmallslice = iris.Constraint(pressure=lambda cell: n1 >=
                                       cell >= n2, longitude=lambda cell:
                                       float(llon) <= cell <= float(ulon),
                                       latitude=lambda cell: float(llat) <=
                                       cell <= float(ulat))
    return xysmallslice


@autojit
def cubemean(var):
    '''Description:
        Find the mean of an iris cube variable
       Attributes:
        var: iris cube variable
    '''
    return var.collapsed(['latitude', 'longitude'], iris.analysis.MEAN)


@autojit
def cube99(var, per=99):
    '''Description:
        Find the Nth PERCENTILE of an iris cube variable
       Attributes:
        var: iris cube variable
        p(int): PERCENTILE normally 1 or 99
    '''
    return var.collapsed(['latitude', 'longitude'], iris.analysis.PERCENTILE,
                         percent=per).data


def vels(flist, xy):
    uf = flist[flist.varname == 'u10'].file
    u = iris.load_cube(uf, xy)
    vf = flist[flist.varname == 'v10'].file
    v = iris.load_cube(vf, xy)
    return u, v


def olrs(flist, xy):
    olr_f = flist[flist.varname == 'olr'].file
    OLR = iris.load_cube(olr_f).extract(xy)
    OLR = OLR[17, :, :]
    olr_10p = cube99(OLR, per=10)
    olr_1p = cube99(OLR, per=1)
    return olr_10p, olr_1p


def colws(flist, xy):
    colwf = flist[flist.varname == 'col_w'].file
    colw = iris.load_cube(colwf).extract(xy)
    varn = colw[5, :, :, :]
    varmean = cubemean(varn).data
    varn99p = cube99(varn)
    return varn99p, varmean


def precips(flist, xy):
    precipfile = flist[flist.varname == 'precip'].file
    precip = iris.load_cube(precipfile).extract(xy)
    precipm = precip[11:15, :, :]
    precip = precip[17, :, :]
    precip99 = cube99(precip)
    precip = np.ndarray.tolist(precip99)
    return precipm, precip99, precip
