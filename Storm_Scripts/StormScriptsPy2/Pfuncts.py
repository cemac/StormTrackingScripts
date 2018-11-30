# -*- coding: utf-8 -*-
"""Shared Funtions

.. module:: Pfuncts
    :platform: Unix
    :synopis: shared funtions namely parallel

.. moduleauther: CEMAC (UoL)

.. description: This module was developed by CEMAC as part of the AMAMA 2050
   Project. This module contrains the funtions required by multiple modules.
   However these functions can be used as stand alone if required.

   :copyright: © 2018 University of Leeds.
   :license: BSD-2 Clause.

Example:
    To use::
        import StormScriptsPy3 as SSP3
        var = SSP3.Pfuncts.method(args)

.. CEMAC_stomtracking:
   https://github.com/cemac/StormTrackingScripts

This module was developed by CEMAC as part of the AMAMA 2050 Project. A lot
of the processing invloved in this work relies on searching through thousands
of files and extracting variables. Perfect for multithreading
"""

import multiprocessing
import pandas as pd
import numpy as np
import iris


def _pickle_method(m):
    '''Taken from
    https://laszukdawid.com/2017/12/13/multiprocessing-in-python-all-about-pickling/
    multiprocessing with in a class requires some adaptation to pickling.
    Circumnavigates GIL lock in pickles - which allows objects to cominicate
    with each other. Outside of a class there is no pickling.
    Example:
        copyreg.pickle(types.MethodType, Pfuncts._pickle_method)
    '''
    class_self = m.im_class if m.im_self is None else m.im_self
    return getattr, (class_self, m.im_func.func_name)


def parallelize_dataframe(df, func, nice):
    '''parallelize a data frame

    Args:
        df (dateframe): dataframe
        func : function, the function to be done
        nice (int): niceness 1/nice share of machince

    Returns:
        df: processed dataframe
    '''
    # Set nprocs make sure at least 1!!
    nprocs = int(multiprocessing.cpu_count()/nice) + 1
    df_split = np.array_split(df, nprocs)  # Chunk up the data frame
    pool = multiprocessing.Pool(nprocs)  # create pool
    # Do the work and put the chucnks back togther
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def yes_or_no(question):
    """Ask the user a yes or no question in python or command line
    """
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
    """Generate slice to slice iris cube

    Args:
        latlons (:obj:`list` of :obj:`int`): list of lat-lons:
            [llon, llat, ulat, ulon]
        n1 (:obj:`int`, optiona): lower pressure bound. Defaults to None.
        n2 (:obj:`int`, optiona): upper pressure bound. Defaults to None.

    Returns:
        iris.Constraint: xysmallslice
    """
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


def cubemean(var):
    '''Find the mean of an iris cube variable
    Args:
        var: iris cube variable
    Returns:
        foat: var.cubemean
    '''
    return var.collapsed(['latitude', 'longitude'], iris.analysis.MEAN)


def cube99(var, per=99):
    '''Find the Nth PERCENTILE of an iris cube variable
    Args:
        var: iris cube variable
        per (:obj:`int`, optional): PERCENTILE normally 1 or 99. Defaults to 99

    Returns:
        foat: var.cube99
    '''
    return var.collapsed(['latitude', 'longitude'], iris.analysis.PERCENTILE,
                         percent=per).data


def vels(flist, xy):
    '''Get Velocity cubes
    Args:
        flist (dataframe): dateframe with filenames (netCDF) and varnames
        xy (iris.Constraint): Iris Constraint e.g. from genslice

    Returns:
        Iris.Cube: u
        Iris.Cube: v
    '''
    uf = flist[flist.varname == 'u10'].file
    u = iris.load_cube(uf, xy)
    vf = flist[flist.varname == 'v10'].file
    v = iris.load_cube(vf, xy)
    return u, v


def olrs(flist, xy):
    '''Get OLRS 10p and 1p
    Args:
        flist (dataframe): dateframe with filenames (netCDF) and varnames
        xy (iris.Constraint): Iris Constraint e.g. from genslice

    Returns:
        float: olr_1p
        float: olr_10p
    '''
    olr_f = flist[flist.varname == 'olr'].file
    OLR = iris.load_cube(olr_f).extract(xy)
    OLR = OLR[17, :, :]
    olr_10p = cube99(OLR, per=10)
    olr_1p = cube99(OLR, per=1)
    return olr_10p, olr_1p


def colws(flist, xy):
    '''Get col_w
    Args:
        flist (dataframe): dateframe with filenames (netCDF) and varnames
        xy (iris.Constraint): Iris Constraint e.g. from genslice

    Returns:
        float: varn99p
        float: varmean
    '''
    colwf = flist[flist.varname == 'col_w'].file
    colw = iris.load_cube(colwf).extract(xy)
    varn = colw[5, :, :, :]
    varmean = cubemean(varn).data
    varn99p = cube99(varn)
    return varn99p, varmean


def precips(flist, xy):
    '''Get precipition values
    Args:
        flist (dataframe): dateframe with filenames (netCDF) and varnames
        xy (iris.Constraint): Iris Constraint e.g. from genslice

    Returns:
        float: precip99
        float: precipm
        Iris.Cube: precip
    '''
    precipfile = flist[flist.varname == 'precip'].file
    precip = iris.load_cube(precipfile).extract(xy)
    precipm = precip[11:15, :, :]
    precip = precip[17, :, :]
    precip99 = cube99(precip)
    precip = np.ndarray.tolist(precip99)
    return precipm, precip99, precip
