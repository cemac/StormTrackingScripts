'''
Storm_in_a_box.py based off of Rory's script
'''
import glob
import numpy as np
import iris


class storminbox(object):
    '''Description
       Stage 1: currently a suit of functions for finding information on
       storms in region and Generating cvs files of that information.
    '''
    def __init__(self, x1, x2, y1, y2, size_of_storm):

        self.varslist = ['stormid', 'year', 'month', 'day', 'hour', 'llon',
                         'ulon', 'llat', 'ulat', 'centlon', 'centlat', 'area',
                         'mean_olr']
        stormsdf = pd.DataFrame(columns=self.varlist)
        fname = ('/nfs/a277/IMPALA/data/4km/a03332_12km/a03332_A1hr_mean_' +
                 'ah261_4km_200012070030-200012072330.nc')
        # file root for generating file list
        froot = '/nfs/a277/IMPALA/data/4km/precip_tracking_12km_hourly/'
        df = pd.DataFrame()
        df2 = pd.DataFrame(columns=['file'], index=[range(0, len(df))])
        df['file'] = (glob.glob(froot+'*/a04203*4km*.txt'))
        # The data frame lists every file but we only want june to october
        for row in df.itertuples():
            if row.file[90:92] in [str(x).zfill(2) for x in range(6, 10)]:
                df2.loc[row[0]] = 0
                df2['file'].loc[row[0]] = row.file
        df = df2.reset_index(drop=True)
        # Storms we want have this infor
        cols = ['storm', 'no', 'area', 'centroid', 'box', 'life', 'u', 'v',
                'mean', 'min', 'max', 'accreted', 'parent', 'child', 'cell']
        for row in df.itertuples():
            vars = pd.read_csv(row.file, names=cols,  header=None,
                               delim_whitespace=True)
            # the txt files have stoms and then child cells with surplus info
            var2 = vars[pd.notnull(vars['mean'])]
            size = var2.area.str[5::]
            var2['area'] = pd.to_numeric(size)*144
            # If it meets our criteria
            storms = var2[var2.area >= size_of_storm]
            # lets create a data frame of the varslist components
            # join DataFrame to stormsdf and move on to next file.
        stormsdf.to_csv(idstring + 'storms_over_box_area' + str(size_of_storm)
                        + '_lons_' + str(x1) + '_' + str(x2) + '_lat_' +
                        str(y1) + '_' + str(y2)+'.csv')
