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
    fname = ('/nfs/a277/IMPALA/data/4km/a03332_12km/a03332_A1hr_mean_' +
             'ah261_4km_200012070030-200012072330.nc')
    cube = iris.load(fname)[1]
    lon = cube.coord('longitude').points.tolist()
    lat = cube.coord('latitude').points.tolist()
    llon, llat, ulat, ulon = latlons
    # llon = lon[int(llon)]
    # ulon = lon[int(ulon)]
    # llat = lat[int(llat)]
    # ulat = lat[int(ulat)]
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
