""" Python module for working Storm tracking CSV files
    STAGE = 1
    * Stage 1 development create functions for data mining
      reduce code size
    * Stage 2 remove hard coding
    * Stage 3 improve effciency
    * Stage 4 integrate
"""


from numpy import genfromtxt as gent
import numpy as np


class dmfuctions:
    '''Description
       Stage 1: currently a suit of functions for finding information on
       storms in region and Generating cvs files of that information.
    '''

# Global variables

    def __init__(self, x1, x2, y1, y2, storm_size):

        # Variables
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2
        self.storm_size = storm_size
        self.varlist = ['year', 'month', 'day', 'hour', 'llon', 'ulon', 'llat',
                        'ulat', 'stormid', 'mean_olr']

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

    def gen_var_csvs(self, csvroot, storms_to_keep):
        '''
        Generate variable csv files to show information about the storm
        csv root = file pattern for file to be Written
        '''
        stormsdf = pd.DataFrame(storms_to_keep, columns=self.varlist)
        # Variables
        all_stormid = []
        all_max_shear = []
        all_hor_shear = []
        all_buoyancy_1800_1p = []
        all_buoyancy_1200_1p = []
        all_omega_1200_1p = []
        all_omega_1800_1p = []
        all_omega_1200_mean = []
        all_omega_1800_mean = []
        all_mass_mean_1200 = []
        all_mass_mean_1800 = []
        all_precip_99th_perc = []
        all_precip_accum = []
        all_col_w_mean = []
        all_col_w_p99 = []
        OLRs = []
        all_OLR_10_perc = []
        all_OLR_1_perc = []
        all_max_w_1200 = []
        all_max_w_1800 = []
        all_area = []
        all_mean_T15_1200 = []
        all_mean_T15_1800 = []
        all_1perc_T15_1800 = []
        all_midday_mslp = []
        all_midday_wind = []
        all_midday_wind3 = []
        all_eve_mslp_mean = []
        all_eve_wind_mean = []
        all_eve_wind3_mean = []
        all_eve_mslp_1p = []
        all_eve_wind_99p = []
        all_eve_wind3_99p = []
        OLRmin = 300.0
        ukeep925 = 0.0
        ukeep650 = 0.0
        ukeepsheer = 0.0
        # for each row create a small sclice using iris
        for row in stormdf.itertuples():
            xysmallslice = iris.Constraint(longitude=lambda cell: row.llon
                                           <= cell <= row.ulon, latitude=lambda
                                           cell: row.llat <= cell <= row.ulat)
            xysmallslice_925 = iris.Constraint(pressure=lambda cell: 925 ==
                                               cell, longitude=lambda cell:
                                               row.llon <= cell <= row.ulon,
                                               latitude=lambda cell: row.llat
                                               <= cell <= row.ulat)
            xysmallslice_650 = iris.Constraint(pressure=lambda cell: 650 ==
                                               cell, longitude=lambda cell:
                                               row.llon <= cell <= row.ulon,
                                               latitude=lambda cell: row.llat
                                               <= cell <= row.ulat)
            xysmallslice_850 = iris.Constraint(pressure=lambda cell: 850 ==
                                               cell, longitude=lambda cell:
                                               row.llon <= cell <= row.ulon,
                                               latitude=lambda cell: row.llat
                                               <= cell <= row.ulat)
            xysmallslice_500 = iris.Constraint(pressure=lambda cell: 500 ==
                                               cell, longitude=lambda cell:
                                               row.llon <= cell <= row.ulon,
                                               latitude=lambda cell: row.llat
                                               <= cell <= row.ulat)
            xysmallslice_600plus = iris.Constraint(pressure=lambda cell: 600 >=
                                                   cell >= 300,
                                                   longitude=lambda cell:
                                                   row.llon <= cell <=
                                                   row.ulon, latitude=lambda
                                                   cell: row.llat <= cell
                                                   <= row.ulat)
            xysmallslice_800_925 = iris.Constraint(pressure=lambda cell:
                                                   925 >= cell >= 800,
                                                   longitude=lambda cell:
                                                   row.llon <= cell <=
                                                   row.ulon, latitude=lambda
                                                   cell: row.llat <= cell <=
                                                   row.ulat)
            xysmallslice_500_800 = iris.Constraint(pressure=lambda cell: 500
                                                   <= cell <= 800 or cell ==
                                                   60, longitude=lambda cell:
                                                   row.llon <= cell <=
                                                   row.ulon, latitude=lambda
                                                   cell: row.llat <= cell
                                                   <= row.ulat)
            goodcheck = 0

            # files lists
            # INVESTIGATE the methodaology of this?  could I go through floder
            # finding files containing the variable and the storm information
            storminfor = (str(int(row.year)) + str(int(row.month)).zfill(2) +
                          str(int(row.day)).zfill(2))
            # If any files with structure exsit
            for i in range(14):
                flelist = self.gen_flist(self, storminfor, i)
                if len(fleist) >= 1:
                    fle_var = load_cube(flelist[0])
                    goodcheck += 1

            if goodcheck == 13:
                cube_col_w = fle_col_w.extract(xysmallslice)
                cube_lowu = fle_u.extract(xysmallslice_800_925)
                cube_highu = fle_u.extract(xysmallslice_500_800)
                cube_lowv = fle_v.extract(xysmallslice_800_925)
                cube_highv = fle_v.extract(xysmallslice_500_800)
                cube_precip = fle_precip.extract(xysmallslice)
                cube_OLR = fle_olr.extract(xysmallslice)
                fle_wet_mass = fle_wet_mass.extract(xysmallslice)
                fle_dry_mass = fle_dry_mass.extract(xysmallslice)
                fle_mass = fle_wet_mass
                fle_mass.data = fle_wet_mass.data - fle_dry_mass.data
                fle_mass.extract(xysmallslice)
                cube_omega = fle_omega.extract(xysmallslice_600plus)
                cube_T = fle_T.extract(xysmallslice_600plus)
                cube_T15 = fle_T15.extract(xysmallslice)

    return

    def gen_flist(self, storminfor, iter):
        '''
        Generate flist
        '''
        dataroot = '/nfs/a299/IMPALA/data/fc/4km/'
        foldername = ['c03236/c03236', 'f30201/f30201', 'f30202/f30202',
                      'f30208/f30208', 'f30204/f30204', 'a04203/a04203',
                      'a03332/a03332', 'c30403/c30403', 'c00409/c00409',
                      'c30404/c30404', 'a30439/a30439', 'c03225/c03225_*'
                      'c03226/c03226*']
        varflist = ['_A1hr_inst_*4km', '_A3hr_inst_*_fc4km_'
                    '_A1hr_mean_*_fc4km_', '_A1hr_inst_*_fc4km_',
                    '_A3hr_mean_*_fc4km_']
        numstr = ['*0000.nc', '0300-*0000.nc', '0030*2330.nc',
                  '0100-*0000.nc', '0130-*2230.nc', '0030-*2330.nc',
                  '*0000.nc', '0100-*.nc', '']

        flelist = glob.glob(dataroot + foldername[iter] + varflist[] +
                            storminfor[] + numstr[])
    return fleist
