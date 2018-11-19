# -*- coding: utf-8 -*-
"""Cube functions


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

import pandas as pd
import numpy as np
from numba import autojit
import iris


# Stand alone methods
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
    llon, llat, ulat, ulon = latlons
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
