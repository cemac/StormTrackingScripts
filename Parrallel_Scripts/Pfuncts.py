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


def _pickle_method(m):
    '''Taken from
    https://laszukdawid.com/2017/12/13/multiprocessing-in-python-all-about-pickling/
    multiprocessing with in a class requires some adaptation to pickling.
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
