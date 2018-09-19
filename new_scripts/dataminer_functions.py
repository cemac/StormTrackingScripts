""" Python module for working Storm tracking CSV files
"""


from numpy import genfromtxt as gent
import numpy as np


class dmfuctions:
    '''Description
       Tbc
    '''

# Global variables

    def __init__(self, x1, x2, y1, y2, storm_size):

        # Variables
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.storm_size = storm_size

    def gen_storms_to_keep(self, altcsvname):
        dates = gent(altcsvname, delimiter=',', names=['stormid', 'year',
                                                       'month',
                                                       'day', 'hour', 'llon',
                                                       'ulon', 'llat', 'ulat',
                                                       'centlon', 'centlat',
                                                       'area', 'mean_olr'])
        dates = np.sort(dates[:], axis=-1, order=['stormid', 'mean_olr'])
        storms_to_keep = np.zeros((1, 10), float)
        # You need to bear in mind that this code wants to track the point when
        # the storm is at minimum OLR.
        for line in dates:
            strm = line['stormid']
            goodup = 0
        for rw in range(0, storms_to_keep.shape[0]):
            if int(strm) == int(storms_to_keep[rw, 8]):
                goodup = goodup + 1
                continue
        if goodup < 1 and 18 == int(line['hour']):
            if np.sum(storms_to_keep[0, :]) == 0:
                storms_to_keep[0, :] = line['year', 'month', 'day', 'hour',
                                            'llon', 'ulon', 'llat', 'ulat',
                                            'stormid', 'mean_olr']
            else:
                temp = np.zeros((1, 10), float)
                temp[0, :] = line['year', 'month', 'day', 'hour', 'llon',
                                  'ulon', 'llat', 'ulat', 'stormid',
                                  'mean_olr']
                storms_to_keep = np.concatenate((storms_to_keep, temp), axis=0)
        return(storms_to_keep)
