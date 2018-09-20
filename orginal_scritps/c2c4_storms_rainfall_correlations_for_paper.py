import scipy.stats as stat
import numpy as np
from numpy import genfromtxt as gent
import matplotlib.pyplot as plt
from matplotlib import colors
from numpy import linspace, meshgrid
from matplotlib.mlab import griddata
import pandas as pd
import glob
def grid(x,y,z, resX = 100, resY = 100):
	xi = linspace(min(x), max(x), resX)
	yi = linspace(min(y), max(y), resY)
	Z = griddata(x,y,z,xi,yi, interp = 'linear')
	X, Y = meshgrid(xi, yi)
	return X,Y,Z

def main(xstart,xend,ystart,yend,size_of_storm):


 all_cube_T15_1200_CC = gent('csvs/PAPER_CC_C2C4_1200Z_mean_T15_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', delimiter = ',')
 all_cube_T15_1200_FC = gent('csvs/PAPER_FC_C2C4_1200Z_mean_T15_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', delimiter = ',')

 all_cube_T15_1800_CC = gent('csvs/PAPER_CC_C2C4_1200Z_1perc_T15_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', delimiter = ',')
 all_cube_T15_1800_FC = gent('csvs/PAPER_FC_C2C4_1200Z_1perc_T15_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', delimiter = ',')

 all_cube_mslp_1200_CC = gent('csvs/PAPER_CC_C2C4_1200Z_mean_mslp_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', delimiter = ',')
 all_cube_mslp_1200_FC = gent('csvs/PAPER_FC_C2C4_1200Z_mean_mslp_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', delimiter = ',')

 all_cube_mslp_1800_CC = gent('csvs/PAPER_CC_C2C4_1800Z_99p_mslp_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', delimiter = ',')
 all_cube_mslp_1800_FC = gent('csvs/PAPER_FC_C2C4_1800Z_99p_mslp_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', delimiter = ',')

 all_cube_10u_1200_CC = gent('csvs/PAPER_CC_C2C4_1200Z_mean_10m_wind_speed_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', delimiter = ',')
 all_cube_10u_1200_FC = gent('csvs/PAPER_FC_C2C4_1200Z_mean_10m_wind_speed_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', delimiter = ',')

 all_cube_10u_1800_CC = gent('csvs/PAPER_CC_C2C4_1800Z_99p_10m_wind_speed_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', delimiter = ',')
 all_cube_10u_1800_FC = gent('csvs/PAPER_FC_C2C4_1800Z_99p_10m_wind_speed_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', delimiter = ',')

 all_cube_10u_cubed_1200_CC = gent('csvs/PAPER_CC_C2C4_1200Z_mean_10m_wind_speed_cubed_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', delimiter = ',')
 all_cube_10u_cubed_1200_FC = gent('csvs/PAPER_FC_C2C4_1200Z_mean_10m_wind_speed_cubed_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', delimiter = ',')

 all_cube_10u_cubed_1800_CC = gent('csvs/PAPER_CC_C2C4_1800Z_mean_10m_wind_speed_cubed_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', delimiter = ',')
 all_cube_10u_cubed_1800_FC = gent('csvs/PAPER_FC_C2C4_1800Z_mean_10m_wind_speed_cubed_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', delimiter = ',')

 all_cube_cold_pool_CC = all_cube_T15_1800_CC - all_cube_T15_1200_CC
 all_cube_cold_pool_FC = all_cube_T15_1800_FC - all_cube_T15_1200_FC
 all_cube_mslp_diff_CC = all_cube_mslp_1800_CC - all_cube_mslp_1200_CC
 all_cube_mslp_diff_FC = all_cube_mslp_1800_FC - all_cube_mslp_1200_FC
 all_cube_10m_u_diff_CC = all_cube_10u_1800_CC - all_cube_10u_1200_CC
 all_cube_10m_u_diff_FC = all_cube_10u_1800_FC - all_cube_10u_1200_FC
 all_cube_10m_u_cubed_diff_CC = all_cube_10u_cubed_1800_CC - all_cube_10u_cubed_1200_CC
 all_cube_10m_u_cubed_diff_FC = all_cube_10u_cubed_1800_FC - all_cube_10u_cubed_1200_FC
 print all_cube_10m_u_cubed_diff_FC[:]
 stopper = raw_input('stop')

 all_cube_zonal_shear_CC = gent('csvs/PAPER_CC_C2C4_1200Z_max_zonal_shear_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', delimiter = ',')
 all_cube_zonal_shear_FC = gent('csvs/PAPER_FC_C2C4_1200Z_max_zonal_shear_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', delimiter = ',')

 all_cube_rh650_hpa_CC = gent('csvs/CC_c2c4_1200_700_hPa_RH_1800_storms.csv', delimiter = ',')
 all_cube_rh650_hpa_FC = gent('csvs/CC_c2c4_1200_700_hPa_RH_1800_storms.csv', delimiter = ',')

 all_cube_horizontal_shear_CC = gent('csvs/PAPER_CC_C2C4_1200Z_max_horizontal_shear_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv' , delimiter = ',')
 all_cube_horizontal_shear_FC = gent('csvs/PAPER_FC_C2C4_1200Z_max_horizontal_shear_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv' , delimiter = ',')
 all_cube_area_CC = gent('csvs/PAPER_CC_C2C4_1800Z_storm_area_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', delimiter = ',')
 all_cube_area_FC = gent('csvs/PAPER_FC_C2C4_1800Z_storm_area_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', delimiter = ',')

 all_cube_TCW_1800_CC = gent('csvs/PAPER_CC_C2C4_1800Z_mean_TCWV_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', delimiter = ',')
 all_cube_TCW_1800_FC = gent('csvs/PAPER_FC_C2C4_1800Z_mean_TCWV_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', delimiter = ',')

 all_cube_TCW_1200_CC = gent('csvs/PAPER_CC_C2C4_1200Z_mean_TCWV_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', delimiter = ',')
 all_cube_TCW_1200_FC = gent('csvs/PAPER_FC_C2C4_1200Z_mean_TCWV_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', delimiter = ',')

 all_cube_precip_99p_CC = gent('csvs/PAPER_CC_C2C4_1800Z_99p_precip_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', delimiter = ',')
 all_cube_precip_99p_FC = gent('csvs/PAPER_FC_C2C4_1800Z_99p_precip_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', delimiter = ',')

 all_cube_precip_volume_CC = gent('csvs/PAPER_CC_C2C4_1800Z_precip_volume_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', delimiter = ',')
 all_cube_precip_volume_FC = gent('csvs/PAPER_FC_C2C4_1800Z_precip_volume_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', delimiter = ',')
 print len(all_cube_precip_volume_CC), len(all_cube_horizontal_shear_CC)
 stopper = raw_input('stop')

 all_cube_min_omega_1800_CC = gent('csvs/PAPER_CC_C2C4_1800Z_min_omega_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', delimiter = ',')
 all_cube_min_omega_1800_FC = gent('csvs/PAPER_FC_C2C4_1800Z_min_omega_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', delimiter = ',')

 all_cube_max_buoy_1800_CC = gent('csvs/PAPER_CC_C2C4_1800Z_max_buoyancy_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', delimiter = ',')
 all_cube_max_buoy_1800_FC = gent('csvs/PAPER_FC_C2C4_1800Z_max_buoyancy_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', delimiter = ',')

 all_cube_min_omega_1200_CC = gent('csvs/PAPER_CC_C2C4_1200Z_min_omega_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', delimiter = ',')
 all_cube_min_omega_1200_FC = gent('csvs/PAPER_FC_C2C4_1200Z_min_omega_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', delimiter = ',')

 all_cube_max_w_1800_CC = gent('csvs/PAPER_CC_C2C4_1800Z_max_w_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', delimiter = ',')
 all_cube_max_w_1800_FC = gent('csvs/PAPER_FC_C2C4_1800Z_max_w_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', delimiter = ',')

 all_cube_mean_OLR_CC = gent('csvs/PAPER_CC_C2C4_1800Z_99p_mean_OLR_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', delimiter = ',')
 all_cube_mean_OLR_FC = gent('csvs/PAPER_FC_C2C4_1800Z_99p_mean_OLR_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', delimiter = ',')

 all_cube_stormid_CC = gent('csvs/PAPER_CC_C2C4_STORMID_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', delimiter = ',')
 all_cube_stormid_FC = gent('csvs/PAPER_FC_C2C4_STORMID_1800Z_storms_99p_rainfall_above_1mm_no_midday_rain.csv', delimiter = ',')

 CAPE_stats_CC = gent('csvs/CC_c2c4_1200_CAPE_CIN_1800_STORMS.csv', delimiter = ',')
 CAPE_stats_FC = gent('csvs/FC_c2c4_1200_CAPE_CIN_1800_STORMS.csv', delimiter = ',')

 # format for CAPE_stats
 # pressure lifting condensation level, level free convection, equalibrium level, CAPE, CIN, STORM ID
 all_cube_CAPE_CC = CAPE_stats_CC[:,3]
 all_cube_CAPE_FC = CAPE_stats_FC[:,3]

 all_cube_CIN_CC = CAPE_stats_CC[:,4]
 all_cube_CIN_FC = CAPE_stats_FC[:,4]

 all_cube_STRMID_CC= CAPE_stats_CC[:,5]
 all_cube_STRMID_FC = CAPE_stats_FC[:,5]

 all_cube_precip_99p_CC = all_cube_precip_99p_CC*3600
 all_cube_precip_99p_FC = all_cube_precip_99p_FC*3600
 all_cube_precip_volume_CC = all_cube_precip_volume_CC*3600
 all_cube_precip_volume_FC = all_cube_precip_volume_FC*3600

 print len(all_cube_STRMID_CC), len(all_cube_stormid_CC), len(all_cube_STRMID_FC), len(all_cube_stormid_FC)
 checker = 0
 killlist = []
 for i in range(0,len(all_cube_stormid_FC)):
	if all_cube_stormid_FC[i] in all_cube_STRMID_FC: 
		checker = checker + 1
	else:
		killlist.extend([i])
# okay, so now we have the storms that exist in both relms. Let's delete the other storms
 all_cube_mslp_diff_fc = np.delete(all_cube_mslp_diff_FC, killlist, axis = 0)
 all_cube_10m_u_diff_fc = np.delete(all_cube_10m_u_diff_FC, killlist, axis = 0)
 all_cube_10m_u_cubed_diff_fc = np.delete(all_cube_10m_u_cubed_diff_FC, killlist, axis = 0)
 all_cube_cold_pool_fc = np.delete(all_cube_cold_pool_FC, killlist, axis = 0)

 all_cube_T15_1800_fc = np.delete(all_cube_T15_1800_FC, killlist, axis = 0)
 all_cube_mslp_1800_fc = np.delete(all_cube_mslp_1800_FC, killlist, axis = 0)
 all_cube_10u_1800_fc = np.delete(all_cube_10u_1800_FC, killlist, axis = 0)
 all_cube_10u_cubed_1800_fc = np.delete(all_cube_10u_cubed_1800_FC, killlist, axis = 0)

 all_cube_zonal_shear_fc = np.delete(all_cube_zonal_shear_FC, killlist, axis = 0)
 all_cube_horizontal_shear_fc = np.delete(all_cube_horizontal_shear_FC, killlist, axis = 0)
 all_cube_area_fc = np.delete(all_cube_area_FC, killlist, axis = 0)
 all_cube_TCW_1800_fc = np.delete(all_cube_TCW_1800_FC, killlist, axis = 0)
 all_cube_TCW_1200_fc = np.delete(all_cube_TCW_1200_FC, killlist, axis = 0)
 all_cube_precip_99p_fc = np.delete(all_cube_precip_99p_FC, killlist, axis = 0)
 all_cube_precip_volume_fc = np.delete(all_cube_precip_volume_FC, killlist, axis = 0)
 all_cube_min_omega_1800_fc = np.delete(all_cube_min_omega_1800_FC, killlist, axis = 0)
 all_cube_max_buoy_1800_fc = np.delete(all_cube_max_buoy_1800_FC, killlist, axis = 0)
 all_cube_max_w_1800_fc = np.delete(all_cube_max_w_1800_FC, killlist, axis = 0)
 all_cube_min_omega_1200_fc = np.delete(all_cube_min_omega_1200_FC, killlist, axis = 0)
 all_cube_mean_OLR_fc = np.delete(all_cube_mean_OLR_FC, killlist, axis = 0)
 all_cube_stormid_fc = np.delete(all_cube_stormid_FC, killlist, axis = 0)
 all_cube_STRMID_FC = all_cube_STRMID_FC[:len(all_cube_zonal_shear_fc)]
 all_cube_CAPE_FC = all_cube_CAPE_FC[:len(all_cube_zonal_shear_fc)]
 all_cube_CIN_FC = all_cube_CIN_FC[:len(all_cube_zonal_shear_fc)]
 # we need to do the CAPE checker seperately
 all_cube_rh_650_hpa_fc = all_cube_rh650_hpa_FC[:len(all_cube_zonal_shear_fc)]
 checker = 0
 for i in range(0,len(all_cube_stormid_fc)):
	checker = checker + all_cube_stormid_fc[i] - all_cube_STRMID_FC[i]
 print checker, 'all good?'
 checker = 0
 killlist = []
 for i in range(0,len(all_cube_CAPE_FC)):
	if all_cube_CAPE_FC[i] > 100 and all_cube_max_w_1800_fc[i] > 0: 
		checker = checker + 1
	else:
		killlist.extend([i])
# okay, so now we have the storms that exist in both relms. Let's delete the other storms
 all_cube_T15_1800_fc = np.delete(all_cube_T15_1800_fc, killlist, axis = 0)
 all_cube_mslp_1800_fc = np.delete(all_cube_mslp_1800_fc, killlist, axis = 0)
 all_cube_10u_1800_fc = np.delete(all_cube_10u_1800_fc, killlist, axis = 0)
 all_cube_10u_cubed_1800_fc = np.delete(all_cube_10u_cubed_1800_fc, killlist, axis = 0)
 all_cube_mslp_diff_fc = np.delete(all_cube_mslp_diff_fc, killlist, axis = 0)
 all_cube_10m_u_diff_fc = np.delete(all_cube_10m_u_diff_fc, killlist, axis = 0)
 all_cube_10m_u_cubed_diff_fc = np.delete(all_cube_10m_u_cubed_diff_fc, killlist, axis = 0)
 all_cube_cold_pool_fc = np.delete(all_cube_cold_pool_fc, killlist, axis = 0)
 all_cube_zonal_shear_fc = np.delete(all_cube_zonal_shear_fc, killlist, axis = 0)
 all_cube_horizontal_shear_fc = np.delete(all_cube_horizontal_shear_fc, killlist, axis = 0)
 all_cube_area_fc = np.delete(all_cube_area_fc, killlist, axis = 0)
 all_cube_TCW_1800_fc = np.delete(all_cube_TCW_1800_fc, killlist, axis = 0)
 all_cube_TCW_1200_fc = np.delete(all_cube_TCW_1200_fc, killlist, axis = 0)
 all_cube_precip_99p_fc = np.delete(all_cube_precip_99p_fc, killlist, axis = 0)
 all_cube_precip_volume_fc = np.delete(all_cube_precip_volume_fc, killlist, axis = 0)
 all_cube_min_omega_1800_fc = np.delete(all_cube_min_omega_1800_fc, killlist, axis = 0)
 all_cube_max_buoy_1800_fc = np.delete(all_cube_max_buoy_1800_fc, killlist, axis = 0)
 all_cube_max_w_1800_fc = np.delete(all_cube_max_w_1800_fc, killlist, axis = 0)
 all_cube_min_omega_1200_fc = np.delete(all_cube_min_omega_1200_fc, killlist, axis = 0)
 all_cube_mean_OLR_fc = np.delete(all_cube_mean_OLR_fc, killlist, axis = 0)
 all_cube_stormid_fc = np.delete(all_cube_stormid_fc, killlist, axis = 0)
 all_cube_rh_650_hpa_fc = np.delete(all_cube_rh_650_hpa_fc, killlist, axis = 0)
 all_cube_STRMID_FC = np.delete(all_cube_STRMID_FC, killlist, axis = 0)
 all_cube_CAPE_FC = np.delete(all_cube_CAPE_FC, killlist, axis = 0)
 all_cube_CIN_FC = np.delete(all_cube_CIN_FC, killlist, axis = 0)
 # we need to do the CAPE checker seperately
 checker = 0
 for i in range(0,len(all_cube_stormid_fc)):
	checker = checker + all_cube_stormid_fc[i] - all_cube_STRMID_FC[i]
 print checker, 'all good?'
 checker = 0
 print len(all_cube_zonal_shear_fc)
 stopper = raw_input('FC check')


 killlist = []
 for i in range(0,len(all_cube_stormid_CC)):
	if all_cube_stormid_CC[i] in all_cube_STRMID_CC: 
		checker = checker + 1
	else:
		killlist.extend([i])
# okay, so now we have the storms that exist in both relms. Let's delete the other storms
 all_cube_T15_1800_cc = np.delete(all_cube_T15_1800_CC, killlist, axis = 0)
 all_cube_mslp_1800_cc = np.delete(all_cube_mslp_1800_CC, killlist, axis = 0)
 all_cube_10u_1800_cc = np.delete(all_cube_10u_1800_CC, killlist, axis = 0)
 all_cube_10u_cubed_1800_cc = np.delete(all_cube_10u_cubed_1800_CC, killlist, axis = 0)
 all_cube_mslp_diff_cc = np.delete(all_cube_mslp_diff_CC, killlist, axis = 0)
 all_cube_10m_u_diff_cc = np.delete(all_cube_10m_u_diff_CC, killlist, axis = 0)
 all_cube_10m_u_cubed_diff_cc = np.delete(all_cube_10m_u_cubed_diff_CC, killlist, axis = 0)
 all_cube_cold_pool_cc = np.delete(all_cube_cold_pool_CC, killlist, axis = 0)
 all_cube_zonal_shear_cc = np.delete(all_cube_zonal_shear_CC, killlist, axis = 0)
 all_cube_horizontal_shear_cc = np.delete(all_cube_horizontal_shear_CC, killlist, axis = 0)
 print np.mean(all_cube_horizontal_shear_cc), np.mean(all_cube_zonal_shear_cc)
 stopper = raw_input('stop')
 all_cube_area_cc = np.delete(all_cube_area_CC, killlist, axis = 0)
 all_cube_TCW_1800_cc = np.delete(all_cube_TCW_1800_CC, killlist, axis = 0)
 all_cube_TCW_1200_cc = np.delete(all_cube_TCW_1200_CC, killlist, axis = 0)
 all_cube_precip_99p_cc = np.delete(all_cube_precip_99p_CC, killlist, axis = 0)
 all_cube_precip_volume_cc = np.delete(all_cube_precip_volume_CC, killlist, axis = 0)
 all_cube_min_omega_1800_cc = np.delete(all_cube_min_omega_1800_CC, killlist, axis = 0)
 all_cube_max_buoy_1800_cc = np.delete(all_cube_max_buoy_1800_CC, killlist, axis = 0)
 all_cube_max_w_1800_cc = np.delete(all_cube_max_w_1800_CC, killlist, axis = 0)
 all_cube_min_omega_1200_cc = np.delete(all_cube_min_omega_1200_CC, killlist, axis = 0)
 all_cube_mean_OLR_cc = np.delete(all_cube_mean_OLR_CC, killlist, axis = 0)
 all_cube_stormid_cc = np.delete(all_cube_stormid_CC, killlist, axis = 0)
 all_cube_STRMID_CC = all_cube_STRMID_CC[:len(all_cube_zonal_shear_cc)]
 all_cube_CAPE_CC = all_cube_CAPE_CC[:len(all_cube_zonal_shear_cc)]
 all_cube_rh_650_hpa_cc = all_cube_rh650_hpa_CC[:len(all_cube_zonal_shear_cc)]
 all_cube_CIN_CC = all_cube_CIN_CC[:len(all_cube_zonal_shear_cc)]
 # we need to do the CAPE checker seperately
 checker = 0
 for i in range(0,len(all_cube_stormid_cc)):
	checker = checker + all_cube_stormid_cc[i] - all_cube_STRMID_CC[i]
 print checker, 'all good?'
 checker = 0
 killlist = []
 for i in range(0,len(all_cube_CAPE_CC)):
	if all_cube_CAPE_CC[i] > 100 and all_cube_max_w_1800_cc[i] > 0: 
		checker = checker + 1
	else:
		killlist.extend([i])
# okay, so now we have the storms that exist in both relms. Let's delete the other storms
 all_cube_T15_1800_cc = np.delete(all_cube_T15_1800_cc, killlist, axis = 0)
 all_cube_mslp_1800_cc = np.delete(all_cube_mslp_1800_cc, killlist, axis = 0)
 all_cube_10u_1800_cc = np.delete(all_cube_10u_1800_cc, killlist, axis = 0)
 all_cube_10u_cubed_1800_cc = np.delete(all_cube_10u_cubed_1800_cc, killlist, axis = 0)
 all_cube_mslp_diff_cc = np.delete(all_cube_mslp_diff_cc, killlist, axis = 0)
 all_cube_10m_u_diff_cc = np.delete(all_cube_10m_u_diff_cc, killlist, axis = 0)
 all_cube_10m_u_cubed_diff_cc = np.delete(all_cube_10m_u_cubed_diff_cc, killlist, axis = 0)
 all_cube_cold_pool_cc = np.delete(all_cube_cold_pool_cc, killlist, axis = 0)
 all_cube_zonal_shear_cc = np.delete(all_cube_zonal_shear_cc, killlist, axis = 0)
 all_cube_horizontal_shear_cc = np.delete(all_cube_horizontal_shear_cc, killlist, axis = 0)
 all_cube_area_cc = np.delete(all_cube_area_cc, killlist, axis = 0)
 all_cube_TCW_1800_cc = np.delete(all_cube_TCW_1800_cc, killlist, axis = 0)
 all_cube_TCW_1200_cc = np.delete(all_cube_TCW_1200_cc, killlist, axis = 0)
 all_cube_precip_99p_cc = np.delete(all_cube_precip_99p_cc, killlist, axis = 0)
 all_cube_precip_volume_cc = np.delete(all_cube_precip_volume_cc, killlist, axis = 0)
 all_cube_min_omega_1800_cc = np.delete(all_cube_min_omega_1800_cc, killlist, axis = 0)
 all_cube_max_buoy_1800_cc = np.delete(all_cube_max_buoy_1800_cc, killlist, axis = 0)
 all_cube_max_w_1800_cc = np.delete(all_cube_max_w_1800_cc, killlist, axis = 0)
 all_cube_min_omega_1200_cc = np.delete(all_cube_min_omega_1200_cc, killlist, axis = 0)
 all_cube_mean_OLR_cc = np.delete(all_cube_mean_OLR_cc, killlist, axis = 0)
 all_cube_stormid_cc = np.delete(all_cube_stormid_cc, killlist, axis = 0)
 all_cube_STRMID_CC = np.delete(all_cube_STRMID_CC, killlist, axis = 0)
 all_cube_CAPE_CC = np.delete(all_cube_CAPE_CC, killlist, axis = 0)
 all_cube_rh_650_hpa_cc = np.delete(all_cube_rh_650_hpa_cc, killlist, axis = 0)
 all_cube_CIN_CC = np.delete(all_cube_CIN_CC, killlist, axis = 0)
 # we need to do the CAPE checker seperately
 checker = 0
 for i in range(0,len(all_cube_stormid_cc)):
	checker = checker + all_cube_stormid_cc[i] - all_cube_STRMID_CC[i]
 print checker, 'all good?'
 checker = 0
 print len(all_cube_zonal_shear_cc), len(all_cube_zonal_shear_fc)
 stopper = raw_input('CC check')



 triple_threat_1200_cc = all_cube_horizontal_shear_cc*all_cube_TCW_1200_cc*all_cube_min_omega_1200_cc
 triple_threat_1200_fc = all_cube_horizontal_shear_fc*all_cube_TCW_1200_fc*all_cube_min_omega_1200_fc
 triple_threat_1800_cc = all_cube_horizontal_shear_cc*all_cube_TCW_1800_cc*all_cube_min_omega_1800_cc
 triple_threat_1800_fc = all_cube_horizontal_shear_fc*all_cube_TCW_1800_fc*all_cube_min_omega_1800_fc

 TCW_omega_1800_cc = all_cube_TCW_1800_cc*all_cube_min_omega_1800_cc
 TCW_omega_1800_fc = all_cube_TCW_1800_fc*all_cube_min_omega_1800_fc

 shear_TCW_1200_cc = all_cube_horizontal_shear_cc*all_cube_TCW_1200_cc
 shear_TCW_1200_fc = all_cube_horizontal_shear_fc*all_cube_TCW_1200_fc
 shear_TCW_1800_cc = all_cube_horizontal_shear_cc*all_cube_TCW_1800_cc
 shear_TCW_1800_fc = all_cube_horizontal_shear_fc*all_cube_TCW_1800_fc
 firstperc_CC = 1.
 firstperc_FC = 1.
 lastperc_CC = 1.
 lastperc_FC = 1.
# lastperc_CC = 1.
# lastperc_FC = 1.

# Now, we want to take the C4 storms from Julia's storm tracking
# we need to use Pandas to do this



# plt.figure(figsize = (20,10))
 plt.clf()
 plt.subplot(3,4,1)
 plt.ylabel('99th percentile \n precipitation rate (mm/hr)')
 plt.scatter(all_cube_horizontal_shear_cc, all_cube_precip_99p_cc, c='r', marker = '*', s = 5)
 plt.scatter(all_cube_horizontal_shear_fc, all_cube_precip_99p_fc, c='b', marker = 'v', s = 5)
 plt.xlabel('1200 UTC wind shear magnitude (m/s)')
# plt.ylabel('99p precipitation (mm/hr)')
 correlfirst = stat.pearsonr(all_cube_horizontal_shear_cc[:],all_cube_precip_99p_cc[:])
 ptest = correlfirst[1]
 correlfirst_cc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(all_cube_horizontal_shear_cc), np.poly1d(np.polyfit(all_cube_horizontal_shear_cc,all_cube_precip_99p_cc,1))(np.unique(all_cube_horizontal_shear_cc)),color ='r', label = 'CC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(all_cube_horizontal_shear_cc), np.poly1d(np.polyfit(all_cube_horizontal_shear_cc,all_cube_precip_99p_cc,1))(np.unique(all_cube_horizontal_shear_cc)),color ='r', linestyle = '--',label = 'CC Correl '+str(correlfirst))
 correlfirst = stat.pearsonr(all_cube_horizontal_shear_fc[:],all_cube_precip_99p_fc[:])
 ptest = correlfirst[1]
 correlfirst_fc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(all_cube_horizontal_shear_fc), np.poly1d(np.polyfit(all_cube_horizontal_shear_fc,all_cube_precip_99p_fc,1))(np.unique(all_cube_horizontal_shear_fc)),color ='k', label = 'FC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(all_cube_horizontal_shear_fc), np.poly1d(np.polyfit(all_cube_horizontal_shear_fc,all_cube_precip_99p_fc,1))(np.unique(all_cube_horizontal_shear_fc)),color ='k', linestyle = '--',label = 'FC Correl '+str(correlfirst))
 plt.title('(a) - CC correl = '+str(correlfirst_cc)+', FC correl = '+str(correlfirst_fc), loc = 'right', fontsize = 8)

 plt.subplot(3,4,11)
# all_cube_TCW_1200_cc = np.delete(all_cube_TCW_1200_CC, killlist, axis = 0)
 plt.scatter(all_cube_cold_pool_cc, all_cube_precip_99p_cc, c='r', marker = '*', s = 5)
 plt.scatter(all_cube_cold_pool_fc, all_cube_precip_99p_fc, c='b', marker = 'v', s = 5)
 plt.xlabel('Cold pool marker (K)')
# plt.ylabel('99p precipitation (mm/hr)')
 correlfirst = stat.pearsonr(all_cube_cold_pool_cc[:],all_cube_precip_99p_cc[:])
 ptest = correlfirst[1]
 correlfirst_cc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(all_cube_cold_pool_cc), np.poly1d(np.polyfit(all_cube_cold_pool_cc,all_cube_precip_99p_cc,1))(np.unique(all_cube_cold_pool_cc)),color ='r', label = 'CC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(all_cube_cold_pool_cc), np.poly1d(np.polyfit(all_cube_cold_pool_cc,all_cube_precip_99p_cc,1))(np.unique(all_cube_cold_pool_cc)),color ='r', linestyle = '--',label = 'CC Correl '+str(correlfirst))
 correlfirst = stat.pearsonr(all_cube_cold_pool_fc[:],all_cube_precip_99p_fc[:])
 ptest = correlfirst[1]
 correlfirst_fc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(all_cube_cold_pool_fc), np.poly1d(np.polyfit(all_cube_cold_pool_fc,all_cube_precip_99p_fc,1))(np.unique(all_cube_cold_pool_fc)),color ='k', label = 'FC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(all_cube_cold_pool_fc), np.poly1d(np.polyfit(all_cube_cold_pool_fc,all_cube_precip_99p_fc,1))(np.unique(all_cube_cold_pool_fc)),color ='k', linestyle = '--',label = 'FC Correl '+str(correlfirst))
 plt.title('(k) - CC correl = '+str(correlfirst_cc)+', FC correl = '+str(correlfirst_fc), loc = 'right', fontsize = 8)


# all_cube_TCW_1800_cc = np.delete(all_cube_TCW_1200_CC, killlist, axis = 0)
 plt.subplot(3,4,2)
 plt.scatter(all_cube_TCW_1800_cc, all_cube_precip_99p_cc, c='r', marker = '*', s = 5)
 plt.scatter(all_cube_TCW_1800_fc, all_cube_precip_99p_fc, c='b', marker = 'v', s = 5)
 plt.xlabel('1800 UTC TCWV (kg/m2)')
# plt.ylabel('99p precipitation (mm/hr)')
 correlfirst = stat.pearsonr(all_cube_TCW_1800_cc[:],all_cube_precip_99p_cc[:])
 ptest = correlfirst[1]
 correlfirst_cc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(all_cube_TCW_1800_cc), np.poly1d(np.polyfit(all_cube_TCW_1800_cc,all_cube_precip_99p_cc,1))(np.unique(all_cube_TCW_1800_cc)),color ='r', label = 'CC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(all_cube_TCW_1800_cc), np.poly1d(np.polyfit(all_cube_TCW_1800_cc,all_cube_precip_99p_cc,1))(np.unique(all_cube_TCW_1800_cc)),color ='r', linestyle = '--',label = 'CC Correl '+str(correlfirst))
 correlfirst = stat.pearsonr(all_cube_TCW_1800_fc[:],all_cube_precip_99p_fc[:])
 ptest = correlfirst[1]
 correlfirst_fc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(all_cube_TCW_1800_fc), np.poly1d(np.polyfit(all_cube_TCW_1800_fc,all_cube_precip_99p_fc,1))(np.unique(all_cube_TCW_1800_fc)),color ='k', label = 'FC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(all_cube_TCW_1800_fc), np.poly1d(np.polyfit(all_cube_TCW_1800_fc,all_cube_precip_99p_fc,1))(np.unique(all_cube_TCW_1800_fc)),color ='k', linestyle = '--',label = 'FC Correl '+str(correlfirst))
 plt.title('(b) - CC correl = '+str(correlfirst_cc)+', FC correl = '+str(correlfirst_fc), loc = 'right', fontsize = 8)
# plt.legend(fontsize = 9)


 depcc =  all_cube_min_omega_1800_cc
 depfc = all_cube_min_omega_1800_fc
 plt.subplot(3,4,3)
 plt.scatter(depcc, all_cube_precip_99p_cc, c='r', marker = '*', s = 5)
 plt.scatter(depfc, all_cube_precip_99p_fc, c='b', marker = 'v', s = 5)
 plt.xlabel('1800 UTC minimum omega (Pa/s)')
# plt.ylabel('99p precipitation (mm/hr)')
 correlfirst = stat.pearsonr(depcc[:],all_cube_precip_99p_cc[:])
 ptest = correlfirst[1]
 correlfirst_cc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(depcc), np.poly1d(np.polyfit(depcc,all_cube_precip_99p_cc,1))(np.unique(depcc)),color ='r', label = 'CC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(depcc), np.poly1d(np.polyfit(depcc,all_cube_precip_99p_cc,1))(np.unique(depcc)),color ='r', linestyle = '--',label = 'CC Correl '+str(correlfirst))
 correlfirst = stat.pearsonr(depfc[:],all_cube_precip_99p_fc[:])
 ptest = correlfirst[1]
 correlfirst_fc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(depfc), np.poly1d(np.polyfit(depfc,all_cube_precip_99p_fc,1))(np.unique(depfc)),color ='k', label = 'FC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(depfc), np.poly1d(np.polyfit(depfc,all_cube_precip_99p_fc,1))(np.unique(depfc)),color ='k', linestyle = '--',label = 'FC Correl '+str(correlfirst))
 plt.title('(c) - CC correl = '+str(correlfirst_cc)+', FC correl = '+str(correlfirst_fc), loc = 'right', fontsize = 8)

# all_cube_mean_OLR_cc = np.delete(all_cube_mean_OLR_CC, killlist, axis = 0)
 depcc =  all_cube_mean_OLR_cc
 depfc = all_cube_mean_OLR_fc
 plt.subplot(3,4,7)
 plt.scatter(depcc, all_cube_precip_99p_cc, c='r', marker = '*', s = 5)
 plt.scatter(depfc, all_cube_precip_99p_fc, c='b', marker = 'v', s = 5)
 plt.xlabel('1800 UTC mean OLR (W/m2)')
# plt.ylabel('99p precipitation (mm/hr)')
 correlfirst = stat.pearsonr(depcc[:],all_cube_precip_99p_cc[:])
 ptest = correlfirst[1]
 correlfirst_cc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(depcc), np.poly1d(np.polyfit(depcc,all_cube_precip_99p_cc,1))(np.unique(depcc)),color ='r', label = 'CC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(depcc), np.poly1d(np.polyfit(depcc,all_cube_precip_99p_cc,1))(np.unique(depcc)),color ='r', linestyle = '--',label = 'CC Correl '+str(correlfirst))
 correlfirst = stat.pearsonr(depfc[:],all_cube_precip_99p_fc[:])
 ptest = correlfirst[1]
 correlfirst_fc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(depfc), np.poly1d(np.polyfit(depfc,all_cube_precip_99p_fc,1))(np.unique(depfc)),color ='k', label = 'FC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(depfc), np.poly1d(np.polyfit(depfc,all_cube_precip_99p_fc,1))(np.unique(depfc)),color ='k', linestyle = '--',label = 'FC Correl '+str(correlfirst))
 plt.title('(g) - CC correl = '+str(correlfirst_cc)+', FC correl = '+str(correlfirst_fc), loc = 'right', fontsize = 8)


# all_cube_CAPE_CC = CAPE_stats_CC[:,3]
 depcc =  np.ndarray.tolist(all_cube_CAPE_CC)
 depfc = np.ndarray.tolist(all_cube_CAPE_FC)
 print len(depfc), len(all_cube_precip_99p_fc)
 plt.subplot(3,4,8)
 plt.scatter(depcc, all_cube_precip_99p_cc, c='r', marker = '*', s = 5)
 plt.scatter(depfc, all_cube_precip_99p_fc, c='b', marker = 'v', s = 5)
 plt.xlabel('1200 UTC mean MU-CAPE (J)')
# plt.ylabel('99p precipitation (mm/hr)')
 correlfirst = stat.pearsonr(depcc[:],all_cube_precip_99p_cc[:])
 ptest = correlfirst[1]
 correlfirst_cc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(depcc), np.poly1d(np.polyfit(depcc,all_cube_precip_99p_cc,1))(np.unique(depcc)),color ='r', label = 'CC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(depcc), np.poly1d(np.polyfit(depcc,all_cube_precip_99p_cc,1))(np.unique(depcc)),color ='r', linestyle = '--',label = 'CC Correl '+str(correlfirst))
 correlfirst = stat.pearsonr(depfc[:],all_cube_precip_99p_fc[:])
 ptest = correlfirst[1]
 correlfirst_fc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(depfc), np.poly1d(np.polyfit(depfc,all_cube_precip_99p_fc,1))(np.unique(depfc)),color ='k', label = 'FC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(depfc), np.poly1d(np.polyfit(depfc,all_cube_precip_99p_fc,1))(np.unique(depfc)),color ='k', linestyle = '--',label = 'FC Correl '+str(correlfirst))
 plt.title('(h) - CC correl = '+str(correlfirst_cc)+', FC correl = '+str(correlfirst_fc), loc = 'right', fontsize = 8)

# all_cube_precip_volume_cc = np.delete(all_cube_precip_volume_CC, killlist, axis = 0)
 depcc =  all_cube_precip_volume_cc
 depfc = all_cube_precip_volume_fc
 plt.subplot(3,4,10)
 plt.scatter(depcc, all_cube_precip_99p_cc, c='r', marker = '*', s = 5)
 plt.scatter(depfc, all_cube_precip_99p_fc, c='b', marker = 'v', s = 5)
 plt.xlabel('1800 UTC 1-hour total rainfall (kg)')
# plt.ylabel('99p precipitation (mm/hr)')
 correlfirst = stat.pearsonr(depcc[:],all_cube_precip_99p_cc[:])
 ptest = correlfirst[1]
 correlfirst_cc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(depcc), np.poly1d(np.polyfit(depcc,all_cube_precip_99p_cc,1))(np.unique(depcc)),color ='r', label = 'CC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(depcc), np.poly1d(np.polyfit(depcc,all_cube_precip_99p_cc,1))(np.unique(depcc)),color ='r', linestyle = '--',label = 'CC Correl '+str(correlfirst))
 correlfirst = stat.pearsonr(depfc[:],all_cube_precip_99p_fc[:])
 ptest = correlfirst[1]
 correlfirst_fc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(depfc), np.poly1d(np.polyfit(depfc,all_cube_precip_99p_fc,1))(np.unique(depfc)),color ='k', label = 'FC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(depfc), np.poly1d(np.polyfit(depfc,all_cube_precip_99p_fc,1))(np.unique(depfc)),color ='k', linestyle = '--',label = 'FC Correl '+str(correlfirst))
 plt.title('(j) - CC correl = '+str(correlfirst_cc)+', FC correl = '+str(correlfirst_fc), loc = 'right', fontsize = 8)

 depcc =  all_cube_rh_650_hpa_cc
 depfc = all_cube_rh_650_hpa_fc
 plt.subplot(3,4,9)
 plt.scatter(depcc, all_cube_precip_99p_cc, c='r', marker = '*', s = 5)
 plt.scatter(depfc, all_cube_precip_99p_fc, c='b', marker = 'v', s = 5)
 plt.xlabel('1200 UTC mean 700 hPa RH (%)')
# plt.ylabel('99p precipitation (mm/hr)')
 correlfirst = stat.pearsonr(depcc[:],all_cube_precip_99p_cc[:])
 ptest = correlfirst[1]
 correlfirst_cc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(depcc), np.poly1d(np.polyfit(depcc,all_cube_precip_99p_cc,1))(np.unique(depcc)),color ='r', label = 'CC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(depcc), np.poly1d(np.polyfit(depcc,all_cube_precip_99p_cc,1))(np.unique(depcc)),color ='r', linestyle = '--',label = 'CC Correl '+str(correlfirst))
 correlfirst = stat.pearsonr(depfc[:],all_cube_precip_99p_fc[:])
 ptest = correlfirst[1]
 correlfirst_fc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(depfc), np.poly1d(np.polyfit(depfc,all_cube_precip_99p_fc,1))(np.unique(depfc)),color ='k', label = 'FC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(depfc), np.poly1d(np.polyfit(depfc,all_cube_precip_99p_fc,1))(np.unique(depfc)),color ='k', linestyle = '--',label = 'FC Correl '+str(correlfirst))
 plt.title('(i) - CC correl = '+str(correlfirst_cc)+', FC correl = '+str(correlfirst_fc), loc = 'right', fontsize = 8)

# all_cube_area_cc = np.delete(all_cube_area_CC, killlist, axis = 0)
 depcc =  shear_TCW_1800_cc
 depfc = shear_TCW_1800_fc
 plt.subplot(3,4,4)
 plt.scatter(depcc, all_cube_precip_99p_cc, c='r', marker = '*', s = 5)
 plt.scatter(depfc, all_cube_precip_99p_fc, c='b', marker = 'v', s = 5)
 plt.xlabel('1200 UTC shear x 1800Z TCWV')
# plt.ylabel('99p precipitation (mm/hr)')
 correlfirst = stat.pearsonr(depcc[:],all_cube_precip_99p_cc[:])
 ptest = correlfirst[1]
 correlfirst_cc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(depcc), np.poly1d(np.polyfit(depcc,all_cube_precip_99p_cc,1))(np.unique(depcc)),color ='r', label = 'CC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(depcc), np.poly1d(np.polyfit(depcc,all_cube_precip_99p_cc,1))(np.unique(depcc)),color ='r', linestyle = '--',label = 'CC Correl '+str(correlfirst))
 correlfirst = stat.pearsonr(depfc[:],all_cube_precip_99p_fc[:])
 ptest = correlfirst[1]
 correlfirst_fc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(depfc), np.poly1d(np.polyfit(depfc,all_cube_precip_99p_fc,1))(np.unique(depfc)),color ='k', label = 'FC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(depfc), np.poly1d(np.polyfit(depfc,all_cube_precip_99p_fc,1))(np.unique(depfc)),color ='k', linestyle = '--',label = 'FC Correl '+str(correlfirst))
 plt.title('(d) - CC correl = '+str(correlfirst_cc)+', FC correl = '+str(correlfirst_fc), loc = 'right', fontsize = 8)





# all_cube_area_cc = np.delete(all_cube_area_CC, killlist, axis = 0)
 depcc =  TCW_omega_1800_cc
 depfc = TCW_omega_1800_fc
 plt.subplot(3,4,5)
 plt.scatter(depcc, all_cube_precip_99p_cc, c='r', marker = '*', s = 5)
 plt.scatter(depfc, all_cube_precip_99p_fc, c='b', marker = 'v', s = 5)
 plt.xlabel('1800 UTC TCWV x omega')
# plt.ylabel('99p precipitation (mm/hr)')
 correlfirst = stat.pearsonr(depcc[:],all_cube_precip_99p_cc[:])
 ptest = correlfirst[1]
 correlfirst_cc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(depcc), np.poly1d(np.polyfit(depcc,all_cube_precip_99p_cc,1))(np.unique(depcc)),color ='r', label = 'CC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(depcc), np.poly1d(np.polyfit(depcc,all_cube_precip_99p_cc,1))(np.unique(depcc)),color ='r', linestyle = '--',label = 'CC Correl '+str(correlfirst))
 correlfirst = stat.pearsonr(depfc[:],all_cube_precip_99p_fc[:])
 ptest = correlfirst[1]
 correlfirst_fc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(depfc), np.poly1d(np.polyfit(depfc,all_cube_precip_99p_fc,1))(np.unique(depfc)),color ='k', label = 'FC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(depfc), np.poly1d(np.polyfit(depfc,all_cube_precip_99p_fc,1))(np.unique(depfc)),color ='k', linestyle = '--',label = 'FC Correl '+str(correlfirst))
 plt.title('(e) - CC correl = '+str(correlfirst_cc)+', FC correl = '+str(correlfirst_fc), loc = 'right', fontsize = 8)

# all_cube_area_cc = np.delete(all_cube_area_CC, killlist, axis = 0)
 depcc =  triple_threat_1800_cc
 depfc = triple_threat_1800_fc
 plt.subplot(3,4,6)
 plt.scatter(depcc, all_cube_precip_99p_cc, c='r', marker = '*', s = 5)
 plt.scatter(depfc, all_cube_precip_99p_fc, c='b', marker = 'v', s = 5)
 plt.xlabel('1800 UTC TCWV x omega x midday shear')
# plt.ylabel('99p precipitation (mm/hr)')
 correlfirst = stat.pearsonr(depcc[:],all_cube_precip_99p_cc[:])
 ptest = correlfirst[1]
 correlfirst_cc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(depcc), np.poly1d(np.polyfit(depcc,all_cube_precip_99p_cc,1))(np.unique(depcc)),color ='r', label = 'CC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(depcc), np.poly1d(np.polyfit(depcc,all_cube_precip_99p_cc,1))(np.unique(depcc)),color ='r', linestyle = '--',label = 'CC Correl '+str(correlfirst))
 correlfirst = stat.pearsonr(depfc[:],all_cube_precip_99p_fc[:])
 ptest = correlfirst[1]
 correlfirst_fc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(depfc), np.poly1d(np.polyfit(depfc,all_cube_precip_99p_fc,1))(np.unique(depfc)),color ='k', label = 'FC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(depfc), np.poly1d(np.polyfit(depfc,all_cube_precip_99p_fc,1))(np.unique(depfc)),color ='k', linestyle = '--',label = 'FC Correl '+str(correlfirst))
 plt.title('(f) - CC correl = '+str(correlfirst_cc)+', FC correl = '+str(correlfirst_fc), loc = 'right', fontsize = 8)

 plt.subplots_adjust(hspace = 0.6, wspace = 0.3)
 plt.show()

 #Okay, so now we want to look at the PDFs of shear across climates
 plt.clf()
 plt.subplot(3,4,1)
 depcc = all_cube_precip_99p_cc
 depfc = all_cube_precip_99p_fc
 mean_depcc = np.average(depcc)
 mean_depfc = np.average(depfc)
 correl = stat.ttest_ind(depcc,depfc,equal_var = False)
 Sigval = 1.0 - correl[1]
 print Sigval, correl[1]
 Sigval = float("{0:.3f}".format(Sigval))
 bins = np.linspace(np.min([np.min(depfc[:]),np.min(depcc[:])]), np.max([np.max(depfc[:]),np.max(depcc[:])]), 25)
 plt.title('(a) - Confidence Interval = '+str(Sigval), fontsize = 10)
# n, bins,patches = plt.hist(depcc[:],bins,normed = 1,facecolor = 'g',label = 'CC')
# n, bins,patches = plt.hist(depfc[:],bins,normed = 1,facecolor = 'b',alpha = 0.5,label = 'FC')
 n, bins,patches = plt.hist(depcc[:],bins,normed = 1,facecolor = 'g',label = 'mean = '+str(float("{0:.2f}".format(mean_depcc))))
 n, bins,patches = plt.hist(depfc[:],bins,normed = 1,facecolor = 'b',alpha = 0.5,label = 'mean = '+str(float("{0:.2f}".format(mean_depfc))))
 plt.scatter(mean_depcc, 0, c= 'g')
 plt.scatter(mean_depfc, 0 , c = 'b')
 plt.xlabel('1800 UTC 99p precipitation rate (mm/hr)')
 plt.ylabel('Probaility density')
 plt.legend(fontsize = 8)
 plt.subplot(3,4,2)
 depcc = all_cube_zonal_shear_cc
 depfc = all_cube_zonal_shear_fc
 mean_depcc = np.average(depcc)
 mean_depfc = np.average(depfc)
 correl = stat.ttest_ind(depcc,depfc,equal_var = False)
 Sigval = 1.0 - correl[1]
 print Sigval, correl[1]
 Sigval = float("{0:.3f}".format(Sigval))
 bins = np.linspace(np.min([np.min(depfc[:]),np.min(depcc[:])]), np.max([np.max(depfc[:]),np.max(depcc[:])]), 25)
 plt.title('(b) - Confidence Interval = '+str(Sigval), fontsize = 10)
# n, bins,patches = plt.hist(depcc[:],bins,normed = 1,facecolor = 'g',label = 'CC')
# n, bins,patches = plt.hist(depfc[:],bins,normed = 1,facecolor = 'b',alpha = 0.5,label = 'FC')
 n, bins,patches = plt.hist(depcc[:],bins,normed = 1,facecolor = 'g',label = 'mean = '+str(float("{0:.2f}".format(mean_depcc))))
 n, bins,patches = plt.hist(depfc[:],bins,normed = 1,facecolor = 'b',alpha = 0.5,label = 'mean = '+str(float("{0:.2f}".format(mean_depfc))))
 plt.scatter(mean_depcc, 0, c= 'g')
 plt.scatter(mean_depfc, 0 , c = 'b')
 plt.xlabel('1200 UTC mean zonal \n wind shear (m/s)')
 plt.legend(fontsize = 8)
 plt.subplot(3,4,3)
 depcc = all_cube_horizontal_shear_cc
 depfc = all_cube_horizontal_shear_fc
 mean_depcc = np.average(depcc)
 mean_depfc = np.average(depfc)
 correl = stat.ttest_ind(depcc,depfc,equal_var = False)
 Sigval = 1.0 - correl[1]
 print Sigval, correl[1]
 Sigval = float("{0:.3f}".format(Sigval))
 bins = np.linspace(np.min([np.min(depfc[:]),np.min(depcc[:])]), np.max([np.max(depfc[:]),np.max(depcc[:])]), 25)
 plt.title('(c) - Confidence Interval = '+str(Sigval), fontsize = 10)
# n, bins,patches = plt.hist(depcc[:],bins,normed = 1,facecolor = 'g',label = 'CC')
# n, bins,patches = plt.hist(depfc[:],bins,normed = 1,facecolor = 'b',alpha = 0.5,label = 'FC')
 n, bins,patches = plt.hist(depcc[:],bins,normed = 1,facecolor = 'g',label = 'mean = '+str(float("{0:.2f}".format(mean_depcc))))
 n, bins,patches = plt.hist(depfc[:],bins,normed = 1,facecolor = 'b',alpha = 0.5,label = 'mean = '+str(float("{0:.2f}".format(mean_depfc))))
 plt.scatter(mean_depcc, 0, c= 'g')
 plt.scatter(mean_depfc, 0 , c = 'b')
 plt.xlabel('1200 UTC mean wind \n shear magnitude (m/s)')
 plt.legend(fontsize = 8)
 plt.subplot(3,4,4)
 depcc = all_cube_TCW_1800_cc
 depfc = all_cube_TCW_1800_fc
 mean_depcc = np.average(depcc)
 mean_depfc = np.average(depfc)
 correl = stat.ttest_ind(depcc,depfc,equal_var = False)
 Sigval = 1.0 - correl[1]
 print Sigval, correl[1]
 Sigval = float("{0:.3f}".format(Sigval))
 bins = np.linspace(np.min([np.min(depfc[:]),np.min(depcc[:])]), np.max([np.max(depfc[:]),np.max(depcc[:])]), 25)
 plt.title('(d) - Confidence Interval = '+str(Sigval), fontsize = 10)
# n, bins,patches = plt.hist(depcc[:],bins,normed = 1,facecolor = 'g',label = 'CC')
# n, bins,patches = plt.hist(depfc[:],bins,normed = 1,facecolor = 'b',alpha = 0.5,label = 'FC')
 n, bins,patches = plt.hist(depcc[:],bins,normed = 1,facecolor = 'g',label = 'mean = '+str(float("{0:.2f}".format(mean_depcc))))
 n, bins,patches = plt.hist(depfc[:],bins,normed = 1,facecolor = 'b',alpha = 0.5,label = 'mean = '+str(float("{0:.2f}".format(mean_depfc))))
 plt.scatter(mean_depcc, 0, c= 'g')
 plt.scatter(mean_depfc, 0 , c = 'b')
 plt.xlabel('1800 UTC mean Total Column \n Water Vapor (kg/m2)')
 plt.legend(fontsize = 8)

 plt.subplot(3,4,5)
 depcc = np.ndarray.tolist(all_cube_CAPE_CC)
 depfc = np.ndarray.tolist(all_cube_CAPE_FC)
 mean_depcc = np.average(depcc)
 mean_depfc = np.average(depfc)
 correl = stat.ttest_ind(depcc,depfc,equal_var = False)
 Sigval = 1.0 - correl[1]
 print Sigval, correl[1]
 Sigval = float("{0:.3f}".format(Sigval))
 bins = np.linspace(np.min([np.min(depfc[:]),np.min(depcc[:])]), np.max([np.max(depfc[:]),np.max(depcc[:])]), 25)
 plt.title('(e) - Confidence Interval = '+str(Sigval), fontsize = 10)
# n, bins,patches = plt.hist(depcc[:],bins,normed = 1,facecolor = 'g',label = 'CC')
# n, bins,patches = plt.hist(depfc[:],bins,normed = 1,facecolor = 'b',alpha = 0.5,label = 'FC')
 n, bins,patches = plt.hist(depcc[:],bins,normed = 1,facecolor = 'g',label = 'mean = '+str(float("{0:.2f}".format(mean_depcc))))
 n, bins,patches = plt.hist(depfc[:],bins,normed = 1,facecolor = 'b',alpha = 0.5,label = 'mean = '+str(float("{0:.2f}".format(mean_depfc))))
 plt.scatter(mean_depcc, 0, c= 'g')
 plt.scatter(mean_depfc, 0 , c = 'b')
 plt.xlabel('1200 UTC mean MU-CAPE (J)')
 plt.legend(fontsize = 8)

 plt.subplot(3,4,6)
 depcc = np.ndarray.tolist(all_cube_CIN_CC)
 depfc = np.ndarray.tolist(all_cube_CIN_FC)
 mean_depcc = np.average(depcc)
 mean_depfc = np.average(depfc)
 correl = stat.ttest_ind(depcc,depfc,equal_var = False)
 Sigval = 1.0 - correl[1]
 print Sigval, correl[1]
 Sigval = float("{0:.3f}".format(Sigval))
 bins = np.linspace(np.min([np.min(depfc[:]),np.min(depcc[:])]), np.max([np.max(depfc[:]),np.max(depcc[:])]), 25)
 plt.title('(f) - Confidence Interval = '+str(Sigval), fontsize = 10)
# n, bins,patches = plt.hist(depcc[:],bins,normed = 1,facecolor = 'g',label = 'CC')
# n, bins,patches = plt.hist(depfc[:],bins,normed = 1,facecolor = 'b',alpha = 0.5,label = 'FC')
 n, bins,patches = plt.hist(depcc[:],bins,normed = 1,facecolor = 'g',label = 'mean = '+str(float("{0:.2f}".format(mean_depcc))))
 n, bins,patches = plt.hist(depfc[:],bins,normed = 1,facecolor = 'b',alpha = 0.5,label = 'mean = '+str(float("{0:.2f}".format(mean_depfc))))
 plt.scatter(mean_depcc, 0, c= 'g')
 plt.scatter(mean_depfc, 0 , c = 'b')
 plt.xlabel('1200 UTC mean MU-CIN (J)')
 plt.legend(fontsize = 8)

 plt.subplot(3,4,7)
 depcc = all_cube_rh_650_hpa_cc
 depfc = all_cube_rh_650_hpa_fc
 mean_depcc = np.average(depcc)
 mean_depfc = np.average(depfc)
 correl = stat.ttest_ind(depcc,depfc,equal_var = False)
 Sigval = 1.0 - correl[1]
 print Sigval, correl[1]
 Sigval = float("{0:.3f}".format(Sigval))
 bins = np.linspace(np.min([np.min(depfc[:]),np.min(depcc[:])]), np.max([np.max(depfc[:]),np.max(depcc[:])]), 25)
 plt.title('(g) - Confidence Interval = '+str(Sigval), fontsize = 10)
# n, bins,patches = plt.hist(depcc[:],bins,normed = 1,facecolor = 'g',label = 'CC')
# n, bins,patches = plt.hist(depfc[:],bins,normed = 1,facecolor = 'b',alpha = 0.5,label = 'FC')
 n, bins,patches = plt.hist(depcc[:],bins,normed = 1,facecolor = 'g',label = 'mean = '+str(float("{0:.2f}".format(mean_depcc))))
 n, bins,patches = plt.hist(depfc[:],bins,normed = 1,facecolor = 'b',alpha = 0.5,label = 'mean = '+str(float("{0:.2f}".format(mean_depfc))))
 plt.scatter(mean_depcc, 0, c= 'g')
 plt.scatter(mean_depfc, 0 , c = 'b')
 plt.xlabel('1200 UTC mean 700 hPa RH (%)')
 plt.legend(fontsize = 8)


 plt.subplot(3,4,8)
 depcc = all_cube_mean_OLR_cc
 depfc = all_cube_mean_OLR_fc
 mean_depcc = np.average(depcc)
 mean_depfc = np.average(depfc)
 correl = stat.ttest_ind(depcc,depfc,equal_var = False)
 Sigval = 1.0 - correl[1]
 print Sigval, correl[1]
 Sigval = float("{0:.3f}".format(Sigval))
 bins = np.linspace(np.min([np.min(depfc[:]),np.min(depcc[:])]), np.max([np.max(depfc[:]),np.max(depcc[:])]), 25)
 plt.title('(h) - Confidence Interval = '+str(Sigval), fontsize = 10)
# n, bins,patches = plt.hist(depcc[:],bins,normed = 1,facecolor = 'g',label = 'CC')
# n, bins,patches = plt.hist(depfc[:],bins,normed = 1,facecolor = 'b',alpha = 0.5,label = 'FC')
 n, bins,patches = plt.hist(depcc[:],bins,normed = 1,facecolor = 'g',label = 'mean = '+str(float("{0:.2f}".format(mean_depcc))))
 n, bins,patches = plt.hist(depfc[:],bins,normed = 1,facecolor = 'b',alpha = 0.5,label = 'mean = '+str(float("{0:.2f}".format(mean_depfc))))
 plt.scatter(mean_depcc, 0, c= 'g')
 plt.scatter(mean_depfc, 0 , c = 'b')
 plt.xlabel('1800 UTC mean OLR (W/m2)')
 plt.legend(fontsize = 8)

 plt.subplot(3,4,9)
 depcc = all_cube_min_omega_1800_cc
 depfc = all_cube_min_omega_1800_fc
 mean_depcc = np.average(depcc)
 mean_depfc = np.average(depfc)
 correl = stat.ttest_ind(depcc,depfc,equal_var = False)
 Sigval = 1.0 - correl[1]
 print Sigval, correl[1]
 Sigval = float("{0:.3f}".format(Sigval))
 bins = np.linspace(np.min([np.min(depfc[:]),np.min(depcc[:])]), np.max([np.max(depfc[:]),np.max(depcc[:])]), 25)
 plt.title('(i) - Confidence Interval = '+str(Sigval), fontsize = 10)
# n, bins,patches = plt.hist(depcc[:],bins,normed = 1,facecolor = 'g',label = 'CC')
# n, bins,patches = plt.hist(depfc[:],bins,normed = 1,facecolor = 'b',alpha = 0.5,label = 'FC')
 n, bins,patches = plt.hist(depcc[:],bins,normed = 1,facecolor = 'g',label = 'mean = '+str(float("{0:.2f}".format(mean_depcc))))
 n, bins,patches = plt.hist(depfc[:],bins,normed = 1,facecolor = 'b',alpha = 0.5,label = 'mean = '+str(float("{0:.2f}".format(mean_depfc))))
 plt.scatter(mean_depcc, 0, c= 'g')
 plt.scatter(mean_depfc, 0 , c = 'b')
 plt.xlabel('1800 UTC min omega (Pa/s)')
 plt.legend(fontsize = 8)

 plt.subplot(3,4,10)
 depcc = all_cube_precip_volume_cc
 depfc = all_cube_precip_volume_fc
 mean_depcc = np.average(depcc)
 mean_depfc = np.average(depfc)
 correl = stat.ttest_ind(depcc,depfc,equal_var = False)
 Sigval = 1.0 - correl[1]
 print Sigval, correl[1]
 Sigval = float("{0:.3f}".format(Sigval))
 bins = np.linspace(np.min([np.min(depfc[:]),np.min(depcc[:])]), np.max([np.max(depfc[:]),np.max(depcc[:])]), 25)
 plt.title('(j) - Confidence Interval = '+str(Sigval), fontsize = 10)
# n, bins,patches = plt.hist(depcc[:],bins,normed = 1,facecolor = 'g',label = 'CC')
# n, bins,patches = plt.hist(depfc[:],bins,normed = 1,facecolor = 'b',alpha = 0.5,label = 'FC')
 n, bins,patches = plt.hist(depcc[:],bins,normed = 1,facecolor = 'g',label = 'mean = '+str(float("{0:.2f}".format(mean_depcc))))
 n, bins,patches = plt.hist(depfc[:],bins,normed = 1,facecolor = 'b',alpha = 0.5,label = 'mean = '+str(float("{0:.2f}".format(mean_depfc))))
 plt.scatter(mean_depcc, 0, c= 'g')
 plt.scatter(mean_depfc, 0 , c = 'b')
 plt.xlabel('1800 UTC 1-hour total rainfall (kg)')
 plt.legend(fontsize = 8)
 

 plt.subplot(3,4,11)
 depcc = all_cube_area_cc*100*100/1000000
 depfc = all_cube_area_fc*100*100/1000000
 mean_depcc = np.average(depcc)
 mean_depfc = np.average(depfc)
 correl = stat.ttest_ind(depcc,depfc,equal_var = False)
 Sigval = 1.0 - correl[1]
 print Sigval, correl[1]
 Sigval = float("{0:.3f}".format(Sigval))
 bins = np.linspace(np.min([np.min(depfc[:]),np.min(depcc[:])]), np.max([np.max(depfc[:]),np.max(depcc[:])]), 25)
 plt.title('(k) - Confidence Interval = '+str(Sigval), fontsize = 10)
# n, bins,patches = plt.hist(depcc[:],bins,normed = 1,facecolor = 'g',label = 'CC')
# n, bins,patches = plt.hist(depfc[:],bins,normed = 1,facecolor = 'b',alpha = 0.5,label = 'FC')
 n, bins,patches = plt.hist(depcc[:],bins,normed = 1,facecolor = 'g',label = 'mean = '+str(float("{0:.2f}".format(mean_depcc))))
 n, bins,patches = plt.hist(depfc[:],bins,normed = 1,facecolor = 'b',alpha = 0.5,label = 'mean = '+str(float("{0:.2f}".format(mean_depfc))))
 plt.scatter(mean_depcc, 0, c= 'g')
 plt.scatter(mean_depfc, 0 , c = 'b')
 plt.xlabel("1800 UTC storm area ('000,000 km2)")
 plt.legend(fontsize = 8)

 plt.subplot(3,4,12)
 depcc = all_cube_cold_pool_cc
 depfc = all_cube_cold_pool_fc - 0.514998
 mean_depcc = np.average(depcc)
 mean_depfc = np.average(depfc)
 correl = stat.ttest_ind(depcc,depfc,equal_var = False)
 Sigval = 1.0 - correl[1]
 print Sigval, correl[1]
 Sigval = float("{0:.3f}".format(Sigval))
 bins = np.linspace(np.min([np.min(depfc[:]),np.min(depcc[:])]), np.max([np.max(depfc[:]),np.max(depcc[:])]), 25)
 plt.title('(l) - Confidence Interval = '+str(Sigval), fontsize = 10)
# n, bins,patches = plt.hist(depcc[:],bins,normed = 1,facecolor = 'g',label = 'CC')
# n, bins,patches = plt.hist(depfc[:],bins,normed = 1,facecolor = 'b',alpha = 0.5,label = 'FC')
 n, bins,patches = plt.hist(depcc[:],bins,normed = 1,facecolor = 'g',label = 'mean = '+str(float("{0:.2f}".format(mean_depcc))))
 n, bins,patches = plt.hist(depfc[:],bins,normed = 1,facecolor = 'b',alpha = 0.5,label = 'mean = '+str(float("{0:.2f}".format(mean_depfc))))
 plt.scatter(mean_depcc, 0, c= 'g')
 plt.scatter(mean_depfc, 0 , c = 'b')
 plt.xlabel('1800 UTC cold pool marker (K)')
 plt.legend(fontsize = 8)



 plt.subplots_adjust(hspace = 0.8, wspace = 0.3, top = 0.95)
 plt.show()

 # right next is to do the other scatter relationships


 plt.clf()
 plt.subplot(3,4,1)
 depcc =  all_cube_mean_OLR_cc
 depfc = all_cube_mean_OLR_fc
 indcc = all_cube_horizontal_shear_cc
 indfc = all_cube_horizontal_shear_fc
 plt.scatter(indcc,depcc, c='r', marker = '*', s = 5)
 plt.scatter(indfc, depfc, c='b', marker = 'v', s = 5)
 plt.xlabel('1200 UTC mean shear magnitude (m/s)')
 plt.ylabel('1800 UTC mean \n OLR (W/m2)')
 correlfirst = stat.pearsonr(depcc[:],indcc[:])
 ptest = correlfirst[1]
 correlfirst_cc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(indcc), np.poly1d(np.polyfit(indcc,depcc,1))(np.unique(indcc)),color ='r')
 else:
	 plt.plot(np.unique(indcc), np.poly1d(np.polyfit(indcc,depcc,1))(np.unique(indcc)),color ='r', linestyle = '--')
 correlfirst = stat.pearsonr(depfc[:],indfc[:])
 ptest = correlfirst[1]
 correlfirst_fc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(indfc), np.poly1d(np.polyfit(indfc,depfc,1))(np.unique(indfc)),color ='k')
 else:
	 plt.plot(np.unique(indfc), np.poly1d(np.polyfit(indfc,depfc,1))(np.unique(indfc)),color ='k', linestyle = '--')
 plt.title('(a) - CC correl = '+str(correlfirst_cc)+', FC correl = '+str(correlfirst_fc), loc = 'right', fontsize = 8)

 plt.subplot(3,4,2)
 depcc =  all_cube_min_omega_1800_cc
 depfc = all_cube_min_omega_1800_fc
 indcc = all_cube_horizontal_shear_cc
 indfc = all_cube_horizontal_shear_fc
 plt.scatter(indcc,depcc, c='r', marker = '*', s = 5)
 plt.scatter(indfc, depfc, c='b', marker = 'v', s = 5)
 plt.xlabel('1200 UTC mean shear magnitude (m/s)')
 plt.ylabel('1800 UTC minimum \n omega (Pa/s)')
 correlfirst = stat.pearsonr(depcc[:],indcc[:])
 ptest = correlfirst[1]
 correlfirst_cc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(indcc), np.poly1d(np.polyfit(indcc,depcc,1))(np.unique(indcc)),color ='r', label = 'CC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(indcc), np.poly1d(np.polyfit(indcc,depcc,1))(np.unique(indcc)),color ='r', linestyle = '--',label = 'CC Correl '+str(correlfirst))
 correlfirst = stat.pearsonr(depfc[:],indfc[:])
 ptest = correlfirst[1]
 correlfirst_fc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(indfc), np.poly1d(np.polyfit(indfc,depfc,1))(np.unique(indfc)),color ='k', label = 'FC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(indfc), np.poly1d(np.polyfit(indfc,depfc,1))(np.unique(indfc)),color ='k', linestyle = '--',label = 'FC Correl '+str(correlfirst))
 plt.title('(b) - CC correl = '+str(correlfirst_cc)+', FC correl = '+str(correlfirst_fc), loc = 'right', fontsize = 8)

 plt.subplot(3,4,3)
 depcc = np.ndarray.tolist(all_cube_CAPE_CC**0.5)
 depfc = np.ndarray.tolist(all_cube_CAPE_FC**0.5)
# depcc =  all_cube_min_omega_1800_cc
# depfc = all_cube_min_omega_1800_fc
 indcc = all_cube_horizontal_shear_cc
 indfc = all_cube_horizontal_shear_fc
 plt.scatter(indcc,depcc, c='r', marker = '*', s = 5)
 plt.scatter(indfc, depfc, c='b', marker = 'v', s = 5)
 plt.xlabel('1200 UTC mean shear magnitude (m/s)')
 plt.ylabel('1200 UTC square root \n MU-CAPE (J)')
 correlfirst = stat.pearsonr(depcc[:],indcc[:])
 ptest = correlfirst[1]
 correlfirst_cc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(indcc), np.poly1d(np.polyfit(indcc,depcc,1))(np.unique(indcc)),color ='r', label = 'CC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(indcc), np.poly1d(np.polyfit(indcc,depcc,1))(np.unique(indcc)),color ='r', linestyle = '--',label = 'CC Correl '+str(correlfirst))
 correlfirst = stat.pearsonr(depfc[:],indfc[:])
 ptest = correlfirst[1]
 correlfirst_fc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(indfc), np.poly1d(np.polyfit(indfc,depfc,1))(np.unique(indfc)),color ='k', label = 'FC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(indfc), np.poly1d(np.polyfit(indfc,depfc,1))(np.unique(indfc)),color ='k', linestyle = '--',label = 'FC Correl '+str(correlfirst))
 plt.title('(c) - CC correl = '+str(correlfirst_cc)+', FC correl = '+str(correlfirst_fc), loc = 'right', fontsize = 8)

 plt.subplot(3,4,4)
 depcc =  all_cube_min_omega_1800_cc
 depfc = all_cube_min_omega_1800_fc
 indcc = all_cube_mean_OLR_cc
 indfc = all_cube_mean_OLR_fc
 plt.scatter(indcc,depcc, c='r', marker = '*', s = 5)
 plt.scatter(indfc, depfc, c='b', marker = 'v', s = 5)
 plt.xlabel('1200 UTC mean OLR (W/m2)')
 plt.ylabel('1800 UTC minimum \n omega (Pa/s)')
 correlfirst = stat.pearsonr(depcc[:],indcc[:])
 ptest = correlfirst[1]
 correlfirst_cc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(indcc), np.poly1d(np.polyfit(indcc,depcc,1))(np.unique(indcc)),color ='r', label = 'CC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(indcc), np.poly1d(np.polyfit(indcc,depcc,1))(np.unique(indcc)),color ='r', linestyle = '--',label = 'CC Correl '+str(correlfirst))
 correlfirst = stat.pearsonr(depfc[:],indfc[:])
 ptest = correlfirst[1]
 correlfirst_fc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(indfc), np.poly1d(np.polyfit(indfc,depfc,1))(np.unique(indfc)),color ='k', label = 'FC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(indfc), np.poly1d(np.polyfit(indfc,depfc,1))(np.unique(indfc)),color ='k', linestyle = '--',label = 'FC Correl '+str(correlfirst))
 plt.title('(d) - CC correl = '+str(correlfirst_cc)+', FC correl = '+str(correlfirst_fc), loc = 'right', fontsize = 8)

 plt.subplot(3,4,5)
 depcc =  all_cube_min_omega_1800_cc
 depfc = all_cube_min_omega_1800_fc
 indcc = np.ndarray.tolist(all_cube_CAPE_CC**0.5)
 indfc = np.ndarray.tolist(all_cube_CAPE_FC**0.5)
 plt.scatter(indcc,depcc, c='r', marker = '*', s = 5)
 plt.scatter(indfc, depfc, c='b', marker = 'v', s = 5)
 plt.xlabel('1200 UTC square root MU-CAPE (J)')
 plt.ylabel('1800 UTC minimum \n omega (Pa/s)')
 correlfirst = stat.pearsonr(depcc[:],indcc[:])
 ptest = correlfirst[1]
 correlfirst_cc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(indcc), np.poly1d(np.polyfit(indcc,depcc,1))(np.unique(indcc)),color ='r', label = 'CC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(indcc), np.poly1d(np.polyfit(indcc,depcc,1))(np.unique(indcc)),color ='r', linestyle = '--',label = 'CC Correl '+str(correlfirst))
 correlfirst = stat.pearsonr(depfc[:],indfc[:])
 ptest = correlfirst[1]
 correlfirst_fc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(indfc), np.poly1d(np.polyfit(indfc,depfc,1))(np.unique(indfc)),color ='k', label = 'FC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(indfc), np.poly1d(np.polyfit(indfc,depfc,1))(np.unique(indfc)),color ='k', linestyle = '--',label = 'FC Correl '+str(correlfirst))
 plt.title('(e) - CC correl = '+str(correlfirst_cc)+', FC correl = '+str(correlfirst_fc), loc = 'right', fontsize = 8)

 plt.subplot(3,4,6)
 depcc =  all_cube_min_omega_1800_cc
 depfc = all_cube_min_omega_1800_fc
 indcc = all_cube_max_buoy_1800_cc
 indfc = all_cube_max_buoy_1800_fc
 plt.scatter(indcc,depcc, c='r', marker = '*', s = 5)
 plt.scatter(indfc, depfc, c='b', marker = 'v', s = 5)
 plt.xlabel('1800 UTC Max buoyancy (K)')
 plt.ylabel('1800 UTC minimum \n omega (Pa/s)')
 correlfirst = stat.pearsonr(depcc[:],indcc[:])
 ptest = correlfirst[1]
 correlfirst_cc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(indcc), np.poly1d(np.polyfit(indcc,depcc,1))(np.unique(indcc)),color ='r', label = 'CC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(indcc), np.poly1d(np.polyfit(indcc,depcc,1))(np.unique(indcc)),color ='r', linestyle = '--',label = 'CC Correl '+str(correlfirst))
 correlfirst = stat.pearsonr(depfc[:],indfc[:])
 ptest = correlfirst[1]
 correlfirst_fc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(indfc), np.poly1d(np.polyfit(indfc,depfc,1))(np.unique(indfc)),color ='k', label = 'FC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(indfc), np.poly1d(np.polyfit(indfc,depfc,1))(np.unique(indfc)),color ='k', linestyle = '--',label = 'FC Correl '+str(correlfirst))
 plt.title('(f) - CC correl = '+str(correlfirst_cc)+', FC correl = '+str(correlfirst_fc), loc = 'right', fontsize = 8)

 plt.subplot(3,4,8)
 indcc =  all_cube_rh_650_hpa_cc
 indfc = all_cube_rh_650_hpa_fc
 depcc = all_cube_cold_pool_cc
 depfc = all_cube_cold_pool_fc
 plt.scatter(indcc,depcc, c='r', marker = '*', s = 5)
 plt.scatter(indfc, depfc, c='b', marker = 'v', s = 5)
 plt.xlabel('1200 UTC mean 650 hPa RH (%)')
 plt.ylabel('1800 UTC cold pool \n marker (K)')
 correlfirst = stat.pearsonr(depcc[:],indcc[:])
 ptest = correlfirst[1]
 correlfirst_cc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(indcc), np.poly1d(np.polyfit(indcc,depcc,1))(np.unique(indcc)),color ='r', label = 'CC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(indcc), np.poly1d(np.polyfit(indcc,depcc,1))(np.unique(indcc)),color ='r', linestyle = '--',label = 'CC Correl '+str(correlfirst))
 correlfirst = stat.pearsonr(depfc[:],indfc[:])
 ptest = correlfirst[1]
 correlfirst_fc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(indfc), np.poly1d(np.polyfit(indfc,depfc,1))(np.unique(indfc)),color ='k', label = 'FC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(indfc), np.poly1d(np.polyfit(indfc,depfc,1))(np.unique(indfc)),color ='k', linestyle = '--',label = 'FC Correl '+str(correlfirst))
 plt.title('(h) - CC correl = '+str(correlfirst_cc)+', FC correl = '+str(correlfirst_fc), loc = 'right', fontsize = 8)


 plt.subplot(3,4,9)
# depcc = all_cube_10u_1800_cc
# depfc = all_cube_10u_1800_fc
 depcc = np.ndarray.tolist(all_cube_CAPE_CC)
 depfc = np.ndarray.tolist(all_cube_CAPE_FC)
# depcc =  all_cube_cold_pool_cc
# depfc = all_cube_cold_pool_fc
 indcc = all_cube_horizontal_shear_cc
 indfc = all_cube_horizontal_shear_fc
 plt.scatter(indcc,depcc, c='r', marker = '*', s = 5)
 plt.scatter(indfc, depfc, c='b', marker = 'v', s = 5)
 plt.ylabel('1200 UTC mean \n MU-CAPE (J)')
 plt.xlabel('1200 UTC mean shear magnitude (m/s)')
 correlfirst = stat.pearsonr(depcc[:],indcc[:])
 ptest = correlfirst[1]
 correlfirst_cc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(indcc), np.poly1d(np.polyfit(indcc,depcc,1))(np.unique(indcc)),color ='r', label = 'CC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(indcc), np.poly1d(np.polyfit(indcc,depcc,1))(np.unique(indcc)),color ='r', linestyle = '--',label = 'CC Correl '+str(correlfirst))
 correlfirst = stat.pearsonr(depfc[:],indfc[:])
 ptest = correlfirst[1]
 correlfirst_fc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(indfc), np.poly1d(np.polyfit(indfc,depfc,1))(np.unique(indfc)),color ='k', label = 'FC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(indfc), np.poly1d(np.polyfit(indfc,depfc,1))(np.unique(indfc)),color ='k', linestyle = '--',label = 'FC Correl '+str(correlfirst))
 plt.title('(i) - CC correl = '+str(correlfirst_cc)+', FC correl = '+str(correlfirst_fc), loc = 'right', fontsize = 8)



 plt.subplot(3,4,10)
 depcc =  all_cube_cold_pool_cc
 depfc = all_cube_cold_pool_fc
 indcc = np.ndarray.tolist(all_cube_CAPE_CC)
 indfc = np.ndarray.tolist(all_cube_CAPE_FC)
 plt.scatter(indcc,depcc, c='r', marker = '*', s = 5)
 plt.scatter(indfc, depfc, c='b', marker = 'v', s = 5)
 plt.ylabel('1800 UTC cold pool \n marker (K)')
 plt.xlabel('1200 UTC mean MU-CAPE (J)')
 correlfirst = stat.pearsonr(depcc[:],indcc[:])
 ptest = correlfirst[1]
 correlfirst_cc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(indcc), np.poly1d(np.polyfit(indcc,depcc,1))(np.unique(indcc)),color ='r', label = 'CC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(indcc), np.poly1d(np.polyfit(indcc,depcc,1))(np.unique(indcc)),color ='r', linestyle = '--',label = 'CC Correl '+str(correlfirst))
 correlfirst = stat.pearsonr(depfc[:],indfc[:])
 ptest = correlfirst[1]
 correlfirst_fc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(indfc), np.poly1d(np.polyfit(indfc,depfc,1))(np.unique(indfc)),color ='k', label = 'FC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(indfc), np.poly1d(np.polyfit(indfc,depfc,1))(np.unique(indfc)),color ='k', linestyle = '--',label = 'FC Correl '+str(correlfirst))
 plt.title('(j) - CC correl = '+str(correlfirst_cc)+', FC correl = '+str(correlfirst_fc), loc = 'right', fontsize = 8)

 plt.subplot(3,4,7)
 depcc =  all_cube_min_omega_1800_cc
 depfc = all_cube_min_omega_1800_fc
 indcc =  all_cube_cold_pool_cc
 indfc = all_cube_cold_pool_fc
 plt.scatter(indcc,depcc, c='r', marker = '*', s = 5)
 plt.scatter(indfc, depfc, c='b', marker = 'v', s = 5)
 plt.ylabel('1800 UTC minimum \n omega (Pa/s)')
 plt.xlabel('1800 UTC cold pool marker (%)')
 correlfirst = stat.pearsonr(depcc[:],indcc[:])
 ptest = correlfirst[1]
 correlfirst_cc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(indcc), np.poly1d(np.polyfit(indcc,depcc,1))(np.unique(indcc)),color ='r', label = 'CC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(indcc), np.poly1d(np.polyfit(indcc,depcc,1))(np.unique(indcc)),color ='r', linestyle = '--',label = 'CC Correl '+str(correlfirst))
 correlfirst = stat.pearsonr(depfc[:],indfc[:])
 ptest = correlfirst[1]
 correlfirst_fc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(indfc), np.poly1d(np.polyfit(indfc,depfc,1))(np.unique(indfc)),color ='k', label = 'FC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(indfc), np.poly1d(np.polyfit(indfc,depfc,1))(np.unique(indfc)),color ='k', linestyle = '--',label = 'FC Correl '+str(correlfirst))
 plt.title('(g) - CC correl = '+str(correlfirst_cc)+', FC correl = '+str(correlfirst_fc), loc = 'right', fontsize = 8)

 plt.subplot(3,4,12)
 depcc =  all_cube_max_buoy_1800_cc
 depfc = all_cube_max_buoy_1800_fc
 indcc = np.ndarray.tolist(all_cube_CAPE_CC)
 indfc = np.ndarray.tolist(all_cube_CAPE_FC)
 plt.scatter(indcc,depcc, c='r', marker = '*', s = 5)
 plt.scatter(indfc, depfc, c='b', marker = 'v', s = 5)
 plt.ylabel('1800 UTC maximum \n buoyancy (K)')
 plt.xlabel('1200 UTC mean MU-CAPE (J)')
 correlfirst = stat.pearsonr(depcc[:],indcc[:])
 ptest = correlfirst[1]
 correlfirst_cc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(indcc), np.poly1d(np.polyfit(indcc,depcc,1))(np.unique(indcc)),color ='r', label = 'CC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(indcc), np.poly1d(np.polyfit(indcc,depcc,1))(np.unique(indcc)),color ='r', linestyle = '--',label = 'CC Correl '+str(correlfirst))
 correlfirst = stat.pearsonr(depfc[:],indfc[:])
 ptest = correlfirst[1]
 correlfirst_fc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(indfc), np.poly1d(np.polyfit(indfc,depfc,1))(np.unique(indfc)),color ='k', label = 'FC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(indfc), np.poly1d(np.polyfit(indfc,depfc,1))(np.unique(indfc)),color ='k', linestyle = '--',label = 'FC Correl '+str(correlfirst))
 plt.title('(l) - CC correl = '+str(correlfirst_cc)+', FC correl = '+str(correlfirst_fc), loc = 'right', fontsize = 8)

 plt.subplot(3,4,11)
 depcc =  all_cube_cold_pool_cc
 depfc = all_cube_cold_pool_fc
 indcc = np.ndarray.tolist(all_cube_CIN_CC)
 indfc = np.ndarray.tolist(all_cube_CIN_FC)
 plt.scatter(indcc,depcc, c='r', marker = '*', s = 5)
 plt.scatter(indfc, depfc, c='b', marker = 'v', s = 5)
 plt.ylabel('1800 UTC cold pool \n marker (K)')
 plt.xlabel('1200 UTC mean MU-CIN (J)')
 correlfirst = stat.pearsonr(depcc[:],indcc[:])
 ptest = correlfirst[1]
 correlfirst_cc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(indcc), np.poly1d(np.polyfit(indcc,depcc,1))(np.unique(indcc)),color ='r', label = 'CC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(indcc), np.poly1d(np.polyfit(indcc,depcc,1))(np.unique(indcc)),color ='r', linestyle = '--',label = 'CC Correl '+str(correlfirst))
 correlfirst = stat.pearsonr(depfc[:],indfc[:])
 ptest = correlfirst[1]
 correlfirst_fc = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(indfc), np.poly1d(np.polyfit(indfc,depfc,1))(np.unique(indfc)),color ='k', label = 'FC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(indfc), np.poly1d(np.polyfit(indfc,depfc,1))(np.unique(indfc)),color ='k', linestyle = '--',label = 'FC Correl '+str(correlfirst))
 plt.title('(k) - CC correl = '+str(correlfirst_cc)+', FC correl = '+str(correlfirst_fc), loc = 'right', fontsize = 8)


 plt.subplots_adjust(hspace = 0.6, wspace = 0.6, top = 0.95, right = 0.95)
 plt.show()


 plt.clf()
 depcc = all_cube_precip_99p_cc/(all_cube_TCW_1800_cc*all_cube_min_omega_1800_cc)
 depfc = all_cube_precip_99p_fc/(all_cube_TCW_1800_fc*all_cube_min_omega_1800_fc)
 plt.subplot(1,2,1)
 indcc = all_cube_horizontal_shear_cc
 indfc = all_cube_horizontal_shear_fc
 plt.scatter(all_cube_horizontal_shear_cc, depcc, c = 'r', marker = '*', s = 5)
 plt.scatter(all_cube_horizontal_shear_fc, depfc, c = 'b', marker = '*', s = 5)
 correlfirst = stat.pearsonr(depcc[:],indcc[:])
 ptest = correlfirst[1]
 correlfirst = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(indcc), np.poly1d(np.polyfit(indcc,depcc,1))(np.unique(indcc)),color ='r', label = 'CC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(indcc), np.poly1d(np.polyfit(indcc,depcc,1))(np.unique(indcc)),color ='r', linestyle = '--',label = 'CC Correl '+str(correlfirst))
 correlfirst = stat.pearsonr(depfc[:],indfc[:])
 ptest = correlfirst[1]
 correlfirst = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(indfc), np.poly1d(np.polyfit(indfc,depfc,1))(np.unique(indfc)),color ='k', label = 'FC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(indfc), np.poly1d(np.polyfit(indfc,depfc,1))(np.unique(indfc)),color ='k', linestyle = '--',label = 'FC Correl '+str(correlfirst))
 plt.legend(fontsize = 9)
 plt.ylabel('precip/(TCWV x omega)')
 plt.xlabel('Horizontal wind shear (m/s)')

 plt.subplot(1,2,2)
 indcc = np.ndarray.tolist(all_cube_CAPE_CC)
 indfc = np.ndarray.tolist(all_cube_CAPE_FC)
 plt.scatter(indcc, depcc, c = 'r', marker = '*', s = 5)
 plt.scatter(indfc, depfc, c = 'b', marker = '*', s = 5)
 correlfirst = stat.pearsonr(depcc[:],indcc[:])
 ptest = correlfirst[1]
 correlfirst = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(indcc), np.poly1d(np.polyfit(indcc,depcc,1))(np.unique(indcc)),color ='r', label = 'CC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(indcc), np.poly1d(np.polyfit(indcc,depcc,1))(np.unique(indcc)),color ='r', linestyle = '--',label = 'CC Correl '+str(correlfirst))
 correlfirst = stat.pearsonr(depfc[:],indfc[:])
 ptest = correlfirst[1]
 correlfirst = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(indfc), np.poly1d(np.polyfit(indfc,depfc,1))(np.unique(indfc)),color ='k', label = 'FC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(indfc), np.poly1d(np.polyfit(indfc,depfc,1))(np.unique(indfc)),color ='k', linestyle = '--',label = 'FC Correl '+str(correlfirst))
 plt.legend(fontsize = 9)
 plt.ylabel('precip/(TCWV x omega)')
 plt.xlabel('SBCAPE')

 plt.show()

 TCW_bins = np.linspace(np.min([np.min(all_cube_TCW_1800_cc[:]),np.min(all_cube_TCW_1800_fc[:])]), np.max([np.max(all_cube_TCW_1800_cc[:]),np.max(all_cube_TCW_1800_fc[:])]), 25)
 cold_pool_bins = np.linspace(np.min([np.min(all_cube_cold_pool_cc[:]),np.min(all_cube_cold_pool_fc[:])]), np.max([np.max(all_cube_cold_pool_cc[:]),np.max(all_cube_cold_pool_fc[:])]), 25)
 omega_bins = np.linspace(np.min([np.min(all_cube_min_omega_1800_cc[:]),np.min(all_cube_min_omega_1800_fc[:])]), np.max([np.max(all_cube_min_omega_1800_cc[:]),np.max(all_cube_min_omega_1800_fc[:])]), 25)

#################################
 shear_bins = np.arange(0,25,1)
 u10_bins = np.arange(0,25,1)
# shear_bins = np.linspace(np.min([np.min(all_cube_horizontal_shear_cc[:]),np.min(all_cube_horizontal_shear_fc[:])]), np.max([np.max(all_cube_horizontal_shear_cc[:]),np.max(all_cube_horizontal_shear_fc[:])]), 25)
# u10_bins = np.linspace(np.min([np.min(all_cube_10u_1800_cc[:]),np.min(all_cube_10u_1800_fc[:])]), np.max([np.max(all_cube_10u_1800_cc[:]),np.max(all_cube_10u_1800_fc[:])]), 25)
 a = 25 
 both_climates_histogram = np.zeros((int(25),int(25),2),float)
 for element in range(0, len(all_cube_TCW_1800_cc)):
        OMEGA = all_cube_min_omega_1800_cc[element]
        TCW = all_cube_TCW_1800_cc[element]
        for x in range(0,int(a) - 1):
                for y in range(0,int(a) - 1):
                        if x < int(a) - 1 and y < int(a) - 1:
                                if omega_bins[x] <= OMEGA < omega_bins[x+1] and TCW_bins[y] <= TCW < TCW_bins[y+1]:
                                        both_climates_histogram[y,x,0] = both_climates_histogram[y,x,0] + 1
                                        both_climates_histogram[y,x,1] = both_climates_histogram[y,x,1] + all_cube_precip_99p_cc[element]
 for element in range(0, len(all_cube_TCW_1800_fc)):
        OMEGA = all_cube_min_omega_1800_fc[element]
        TCW = all_cube_TCW_1800_fc[element]
        strmid = all_cube_stormid_FC[element]
        for x in range(0,int(a) - 1):
                for y in range(0,int(a) - 1):
                        if x < int(a) - 1 and y < int(a) - 1:
                                if omega_bins[x] <= OMEGA < omega_bins[x+1] and TCW_bins[y] <= TCW < TCW_bins[y+1]:
                                        both_climates_histogram[y,x,0] = both_climates_histogram[y,x,0] + 1
                                        both_climates_histogram[y,x,1] = both_climates_histogram[y,x,1] + all_cube_precip_99p_fc[element]
 for x in range(0,int(a)):
        for y in range(0,int(a)):
                 if both_climates_histogram[x,y,0] >0: both_climates_histogram[x,y,1] = both_climates_histogram[x,y,1] / both_climates_histogram[x,y,0]
                 both_climates_histogram[x,y,0] = 100 * both_climates_histogram[x,y,0] / (len(all_cube_min_omega_1800_cc)+len(all_cube_min_omega_1800_fc))
 plt.subplot(2,2,1)
 pallette1 = plt.get_cmap('rainbow')
 pallette1.set_under('k', alpha = 0.3)
 pallette1.set_over('Gray')
 levels = np.linspace(1., np.max(both_climates_histogram[:,:,1]),20)
 #cd = plt.contourf(omega_bins,TCW_bins,both_climates_histogram[:,:,1],levels, cmap = plt.get_cmap('spectral'))
 cd = plt.pcolor(omega_bins,TCW_bins,both_climates_histogram[:,:,1],vmin = 1.,vmax = np.max(both_climates_histogram[:,:,1]), cmap = pallette1)
 cb = plt.colorbar(cd,orientation = 'vertical')
 cb.set_label('both climates \n precip rate (mm/hr)')
 plt.xlabel('1800 UTC minimum omega (Pa/s)')
 plt.ylabel('1800 UTC TCW (kg/m2)')
 plt.title('(a)', x = 0.01, fontsize = 10)

 both_climates_histogram = np.zeros((int(25),int(25),2),float)
 for element in range(0, len(all_cube_TCW_1800_cc)):
        OMEGA = all_cube_min_omega_1800_cc[element]
        TCW = all_cube_cold_pool_cc[element]
        for x in range(0,int(a) - 1):
                for y in range(0,int(a) - 1):
                        if x < int(a) - 1 and y < int(a) - 1:
                                if omega_bins[x] <= OMEGA < omega_bins[x+1] and cold_pool_bins[y] <= TCW < cold_pool_bins[y+1]:
                                        both_climates_histogram[y,x,0] = both_climates_histogram[y,x,0] + 1
                                        both_climates_histogram[y,x,1] = both_climates_histogram[y,x,1] + all_cube_precip_99p_cc[element]
 for element in range(0, len(all_cube_TCW_1800_fc)):
        OMEGA = all_cube_min_omega_1800_fc[element]
        TCW = all_cube_cold_pool_fc[element]
        strmid = all_cube_stormid_FC[element]
        for x in range(0,int(a) - 1):
                for y in range(0,int(a) - 1):
                        if x < int(a) - 1 and y < int(a) - 1:
                                if omega_bins[x] <= OMEGA < omega_bins[x+1] and cold_pool_bins[y] <= TCW < cold_pool_bins[y+1]:
                                        both_climates_histogram[y,x,0] = both_climates_histogram[y,x,0] + 1
                                        both_climates_histogram[y,x,1] = both_climates_histogram[y,x,1] + all_cube_precip_99p_fc[element]
 for x in range(0,int(a)):
        for y in range(0,int(a)):
                 if both_climates_histogram[x,y,0] >0: both_climates_histogram[x,y,1] = both_climates_histogram[x,y,1] / both_climates_histogram[x,y,0]
                 both_climates_histogram[x,y,0] = 100 * both_climates_histogram[x,y,0] / (len(all_cube_min_omega_1800_cc)+len(all_cube_min_omega_1800_fc))
 plt.subplot(2,2,2)
 pallette1 = plt.get_cmap('rainbow')
 pallette1.set_under('k', alpha = 0.3)
 pallette1.set_over('Gray')
 levels = np.linspace(1., np.max(both_climates_histogram[:,:,1]),20)
 cd = plt.pcolor(omega_bins,cold_pool_bins,both_climates_histogram[:,:,1],vmin = 1., vmax = np.max(both_climates_histogram[:,:,1]), cmap = pallette1)
# cd = plt.contourf(omega_bins,cold_pool_bins,both_climates_histogram[:,:,1],levels, cmap = plt.get_cmap('spectral'))
 cb = plt.colorbar(cd,orientation = 'vertical')
 cb.set_label('both climates \n precip rate (mm/hr)')
 plt.xlabel('1800 UTC minimum omega (Pa/s)')
 plt.ylabel('1800 UTC cold pool marker (K)')
 plt.title('(b)', x = 0.01, fontsize = 10)

 #plt.savefig('CC_FC_omega850_tcw_rainfall_percentile_'+str(threshold)+'_number_of_bins_'+str(a)+'_precip_pcolor_1800z_no_midday_rain_C2_C4_storms.png')

 both_climates_histogram = np.zeros((int(25),int(25),2),float)
 for element in range(0, len(all_cube_min_omega_1800_cc)):
        OMEGA = all_cube_horizontal_shear_cc[element]
        TCW = all_cube_10u_1800_cc[element]
        for x in range(0,int(a) - 1):
                for y in range(0,int(a) - 1):
                        if x < int(a) - 1 and y < int(a) - 1:
                                if shear_bins[x] <= OMEGA < shear_bins[x+1] and u10_bins[y] <= TCW < u10_bins[y+1]:
                                        both_climates_histogram[y,x,0] = both_climates_histogram[y,x,0] + 1
                                        both_climates_histogram[y,x,1] = both_climates_histogram[y,x,1] + all_cube_min_omega_1800_cc[element]
 for element in range(0, len(all_cube_min_omega_1800_fc)):
        OMEGA = all_cube_horizontal_shear_fc[element]
        TCW = all_cube_10u_1800_fc[element]
        strmid = all_cube_stormid_FC[element]
        for x in range(0,int(a) - 1):
                for y in range(0,int(a) - 1):
                        if x < int(a) - 1 and y < int(a) - 1:
                                if shear_bins[x] <= OMEGA < shear_bins[x+1] and u10_bins[y] <= TCW < u10_bins[y+1]:
                                        both_climates_histogram[y,x,0] = both_climates_histogram[y,x,0] + 1
                                        both_climates_histogram[y,x,1] = both_climates_histogram[y,x,1] + all_cube_min_omega_1800_fc[element]
 for x in range(0,int(a)):
        for y in range(0,int(a)):
                 if both_climates_histogram[x,y,0] >0: both_climates_histogram[x,y,1] = both_climates_histogram[x,y,1] / both_climates_histogram[x,y,0]
                 both_climates_histogram[x,y,0] = 100 * both_climates_histogram[x,y,0] / (len(all_cube_min_omega_1800_cc)+len(all_cube_min_omega_1800_fc))
 plt.subplot(2,2,3)
 pallette1 = plt.get_cmap('rainbow')
 pallette1.set_under('k', alpha = 0.3)
 pallette1.set_over('Gray')
 levels = [-110,-105,-100,-95,-90,-85,-80,-75,-70,-65,-60,-55,-50,-45,-40,-35,-30,-25,-20,-15,-10]
 cd = plt.pcolor(shear_bins,u10_bins,both_climates_histogram[:,:,1],vmin = -110,vmax = -10, cmap = pallette1)
 #cd = plt.contourf(shear_bins,u10_bins,both_climates_histogram[:,:,1],levels, cmap = plt.get_cmap('spectral'))
 cb = plt.colorbar(cd,orientation = 'vertical')
 cb.set_label('both climates 1800 UTC \n min omega (Pa/s)')
 plt.xlabel('1200 UTC mean horizontal shear (m/s)')
 plt.ylabel('1800 UTC 99th percentile \n 10-m wind speed (m/s)')
 plt.title('(c)', x = 0.01, fontsize = 10)
 plt.subplots_adjust(hspace = 0.4, wspace = 0.4)
 plt.show()

 # NOW  DO THE COLD POOL WORK
 depcc =  all_cube_cold_pool_cc
 depfc = all_cube_cold_pool_fc
 plt.subplot(2,2,1)
 plt.scatter(depcc, all_cube_precip_99p_cc, c='r', marker = '*', s = 5)
 plt.scatter(depfc, all_cube_precip_99p_fc, c='b', marker = 'v', s = 5)
 plt.title('(a)', x = 0.01, fontsize = 10)
 plt.xlabel('1800Z 1 perc T - 1200Z mean T (K)')
 plt.ylabel('99p precipitation (mm/hr)')
 correlfirst = stat.pearsonr(depcc[:],all_cube_precip_99p_cc[:])
 ptest = correlfirst[1]
 correlfirst = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(depcc), np.poly1d(np.polyfit(depcc,all_cube_precip_99p_cc,1))(np.unique(depcc)),color ='r', label = 'CC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(depcc), np.poly1d(np.polyfit(depcc,all_cube_precip_99p_cc,1))(np.unique(depcc)),color ='r', linestyle = '--',label = 'CC Correl '+str(correlfirst))
 correlfirst = stat.pearsonr(depfc[:],all_cube_precip_99p_fc[:])
 ptest = correlfirst[1]
 correlfirst = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(depfc), np.poly1d(np.polyfit(depfc,all_cube_precip_99p_fc,1))(np.unique(depfc)),color ='k', label = 'FC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(depfc), np.poly1d(np.polyfit(depfc,all_cube_precip_99p_fc,1))(np.unique(depfc)),color ='k', linestyle = '--',label = 'FC Correl '+str(correlfirst))
 plt.legend(fontsize = 9)

 depcc =  all_cube_cold_pool_cc
 depfc = all_cube_cold_pool_fc - 0.514998
 plt.subplot(2,2,2)
 plt.scatter(depcc, all_cube_precip_99p_cc, c='r', marker = '*', s = 5)
 plt.scatter(depfc, all_cube_precip_99p_fc, c='b', marker = 'v', s = 5)
 plt.title('(b)', x = 0.01, fontsize = 10)
 plt.xlabel('1800Z 1 perc T - 1200Z mean T \n [CC Corected] (K)')
# plt.ylabel('99p precipitation (mm/hr)')
 correlfirst = stat.pearsonr(depcc[:],all_cube_precip_99p_cc[:])
 ptest = correlfirst[1]
 correlfirst = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(depcc), np.poly1d(np.polyfit(depcc,all_cube_precip_99p_cc,1))(np.unique(depcc)),color ='r', label = 'CC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(depcc), np.poly1d(np.polyfit(depcc,all_cube_precip_99p_cc,1))(np.unique(depcc)),color ='r', linestyle = '--',label = 'CC Correl '+str(correlfirst))
 correlfirst = stat.pearsonr(depfc[:],all_cube_precip_99p_fc[:])
 ptest = correlfirst[1]
 correlfirst = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(depfc), np.poly1d(np.polyfit(depfc,all_cube_precip_99p_fc,1))(np.unique(depfc)),color ='k', label = 'FC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(depfc), np.poly1d(np.polyfit(depfc,all_cube_precip_99p_fc,1))(np.unique(depfc)),color ='k', linestyle = '--',label = 'FC Correl '+str(correlfirst))
 plt.legend(fontsize = 9)

 depcc =  all_cube_cold_pool_cc
 depfc = all_cube_cold_pool_fc
 indcc = all_cube_mean_OLR_cc
 indfc = all_cube_mean_OLR_fc
 plt.subplot(2,2,3)
 plt.scatter(depcc, indcc, c='r', marker = '*', s = 5)
 plt.scatter(depfc, indfc, c='b', marker = 'v', s = 5)
 plt.title('(c)', x = 0.01, fontsize = 10)
 plt.xlabel('1800Z 1 perc T - 1200Z mean T (K)')
 plt.ylabel('mean OLR (W/m2)')
 correlfirst = stat.pearsonr(depcc[:],indcc[:])
 ptest = correlfirst[1]
 correlfirst = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(depcc), np.poly1d(np.polyfit(depcc,indcc,1))(np.unique(depcc)),color ='r', label = 'CC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(depcc), np.poly1d(np.polyfit(depcc,indcc,1))(np.unique(depcc)),color ='r', linestyle = '--',label = 'CC Correl '+str(correlfirst))
 correlfirst = stat.pearsonr(depfc[:],indfc[:])
 ptest = correlfirst[1]
 correlfirst = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(depfc), np.poly1d(np.polyfit(depfc,indfc,1))(np.unique(depfc)),color ='k', label = 'FC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(depfc), np.poly1d(np.polyfit(depfc,indfc,1))(np.unique(depfc)),color ='k', linestyle = '--',label = 'FC Correl '+str(correlfirst))
 plt.legend(fontsize = 9)

 depcc =  all_cube_cold_pool_cc
 depfc = all_cube_cold_pool_fc - 0.514998
 plt.subplot(2,2,4)
 plt.scatter(depcc, indcc, c='r', marker = '*', s = 5)
 plt.scatter(depfc, indfc, c='b', marker = 'v', s = 5)
 plt.title('(f)', x = 0.01, fontsize = 10)
 plt.xlabel('1800Z 1 perc T - 1200Z mean T \n [CC corected] (K)')
# plt.ylabel('99p precipitation (mm/hr)')
 correlfirst = stat.pearsonr(depcc[:],indcc[:])
 ptest = correlfirst[1]
 correlfirst = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(depcc), np.poly1d(np.polyfit(depcc,indcc,1))(np.unique(depcc)),color ='r', label = 'CC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(depcc), np.poly1d(np.polyfit(depcc,indcc,1))(np.unique(depcc)),color ='r', linestyle = '--',label = 'CC Correl '+str(correlfirst))
 correlfirst = stat.pearsonr(depfc[:],indfc[:])
 ptest = correlfirst[1]
 correlfirst = float("{0:.3f}".format(correlfirst[0]))
 if ptest < 0.05:
	 plt.plot(np.unique(depfc), np.poly1d(np.polyfit(depfc,indfc,1))(np.unique(depfc)),color ='k', label = 'FC Correl '+str(correlfirst))
 else:
	 plt.plot(np.unique(depfc), np.poly1d(np.polyfit(depfc,indfc,1))(np.unique(depfc)),color ='k', linestyle = '--',label = 'FC Correl '+str(correlfirst))
 plt.legend(fontsize = 9)
 plt.show()

 # Now do the PDFs of cold pool factors
 plt.clf()
 plt.subplot(2,4,5)
 depcc = np.ndarray.tolist(all_cube_cold_pool_cc)
 depfc = np.ndarray.tolist(all_cube_cold_pool_fc)
 mean_depcc = np.average(depcc)
 mean_depfc = np.average(depfc)
 print 'T midday diff FC - CC = '+str(float(mean_depfc-mean_depcc))
 correl = stat.ttest_ind(depcc,depfc,equal_var = False)
 Sigval = 1.0 - correl[1]
 Sigval = float("{0:.3f}".format(Sigval))
 bins = np.linspace(np.min([np.min(depfc[:]),np.min(depcc[:])]), np.max([np.max(depfc[:]),np.max(depcc[:])]), 25)
 plt.title('(e) - Confidence Interval = '+str(Sigval), fontsize = 10)
 n, bins,patches = plt.hist(depcc[:],bins,normed = 1,facecolor = 'g',label = 'mean = '+str(float("{0:.2f}".format(mean_depcc))))
 n, bins,patches = plt.hist(depfc[:],bins,normed = 1,facecolor = 'b',alpha = 0.5,label = 'mean = '+str(float("{0:.2f}".format(mean_depfc))))
 plt.scatter(mean_depcc, 0, c= 'g')
 plt.scatter(mean_depfc, 0 , c = 'b')
 plt.xlabel('1800Z 1% 1.5-meter T \n - 1200Z mean 1.5-meter T (K)')
 plt.legend(fontsize = 8)
 plt.subplot(2,4,6)
 depcc = np.ndarray.tolist((all_cube_mslp_diff_cc+0.11185)/100.)
 depfc = np.ndarray.tolist(all_cube_mslp_diff_fc/100.)
 mean_depcc = np.average(depcc)
 mean_depfc = np.average(depfc)
 print 'p midday diff FC - CC = '+str(float(mean_depfc-mean_depcc))
 correl = stat.ttest_ind(depcc,depfc,equal_var = False)
 Sigval = 1.0 - correl[1]
 Sigval = float("{0:.3f}".format(Sigval))
 bins = np.linspace(np.min([np.min(depfc[:]),np.min(depcc[:])]), np.max([np.max(depfc[:]),np.max(depcc[:])]), 25)
 plt.title('(f) - Confidence Interval = '+str(Sigval), fontsize = 10)
 n, bins,patches = plt.hist(depcc[:],bins,normed = 1,facecolor = 'g',label = 'mean = '+str(float("{0:.2f}".format(mean_depcc))))
 n, bins,patches = plt.hist(depfc[:],bins,normed = 1,facecolor = 'b',alpha = 0.5,label = 'mean = '+str(float("{0:.2f}".format(mean_depfc))))
# n, bins,patches = plt.hist(depcc[:],bins,normed = 1,facecolor = 'g',label = 'CC')
# n, bins,patches = plt.hist(depfc[:],bins,normed = 1,facecolor = 'b',alpha = 0.5,label = 'FC')
 plt.scatter(mean_depcc, 0, c= 'g')
 plt.scatter(mean_depfc, 0 , c = 'b')
 plt.xlabel('1800Z 99th perc surface pressure \n - 1200Z mean surface pressure (hPa)')
 plt.legend(fontsize = 8)
 plt.subplot(2,4,7)
 depcc = np.ndarray.tolist(all_cube_10m_u_diff_cc)
 depfc = np.ndarray.tolist(all_cube_10m_u_diff_fc+0.38104)
 mean_depcc = np.average(depcc)
 mean_depfc = np.average(depfc)
 print 'u midday diff FC - CC = '+str(float(mean_depfc-mean_depcc))
 correl = stat.ttest_ind(depcc,depfc,equal_var = False)
 Sigval = 1.0 - correl[1]
 Sigval = float("{0:.3f}".format(Sigval))
 bins = np.linspace(np.min([np.min(depfc[:]),np.min(depcc[:])]), np.max([np.max(depfc[:]),np.max(depcc[:])]), 25)
 plt.title('(g) - Confidence Interval = '+str(Sigval), fontsize = 10)
 n, bins,patches = plt.hist(depcc[:],bins,normed = 1,facecolor = 'g',label = 'mean = '+str(float("{0:.2f}".format(mean_depcc))))
 n, bins,patches = plt.hist(depfc[:],bins,normed = 1,facecolor = 'b',alpha = 0.5,label = 'mean = '+str(float("{0:.2f}".format(mean_depfc))))
# n, bins,patches = plt.hist(depcc[:],bins,normed = 1,facecolor = 'g',label = 'CC')
# n, bins,patches = plt.hist(depfc[:],bins,normed = 1,facecolor = 'b',alpha = 0.5,label = 'FC')
 plt.scatter(mean_depcc, 0, c= 'g')
 plt.scatter(mean_depfc, 0 , c = 'b')
 plt.xlabel('1800Z 99% 10-m wind speed \n - 1200Z mean 10-m wind speed (m/s)')
 plt.legend(fontsize = 8)
 plt.subplot(2,4,8)
 depcc = np.ndarray.tolist(all_cube_10m_u_cubed_diff_cc)
 depfc = np.ndarray.tolist(all_cube_10m_u_cubed_diff_fc+33.2881)
 mean_depcc = np.average(depcc)
 mean_depfc = np.average(depfc)
 print 'u3 midday diff FC - CC = '+str(float(mean_depfc-mean_depcc))
 correl = stat.ttest_ind(depcc,depfc,equal_var = False)
 Sigval = 1.0 - correl[1]
 Sigval = float("{0:.3f}".format(Sigval))
 bins = np.linspace(np.min([np.min(depfc[:]),np.min(depcc[:])]), np.max([np.max(depfc[:]),np.max(depcc[:])]), 25)
 plt.title('(h) - Confidence Interval = '+str(Sigval), fontsize = 10)
 n, bins,patches = plt.hist(depcc[:],bins,normed = 1,facecolor = 'g',label = 'mean = '+str(float("{0:.2f}".format(mean_depcc))))
 n, bins,patches = plt.hist(depfc[:],bins,normed = 1,facecolor = 'b',alpha = 0.5,label = 'mean = '+str(float("{0:.2f}".format(mean_depfc))))
# n, bins,patches = plt.hist(depcc[:],bins,normed = 1,facecolor = 'g',label = 'CC')
# n, bins,patches = plt.hist(depfc[:],bins,normed = 1,facecolor = 'b',alpha = 0.5,label = 'FC')
 plt.scatter(mean_depcc, 0, c= 'g')
 plt.scatter(mean_depfc, 0 , c = 'b')
 plt.xlabel('1800Z mean 10-m w.s.c. \n - 1200Z mean 10-m w.s.c. (m3/s3)')
 plt.legend(fontsize = 8)
 plt.subplot(2,4,1)
 depcc = all_cube_T15_1800_cc
 depfc = all_cube_T15_1800_fc
 mean_depcc = np.average(depcc)
 mean_depfc = np.average(depfc)
 print 'T diff FC - CC = '+str(float(mean_depfc-mean_depcc))
 correl = stat.ttest_ind(depcc,depfc,equal_var = False)
 Sigval = 1.0 - correl[1]
 Sigval = float("{0:.3f}".format(Sigval))
 bins = np.linspace(np.min([np.min(depfc[:]),np.min(depcc[:])]), np.max([np.max(depfc[:]),np.max(depcc[:])]), 25)
 plt.title('(a) - Confidence Interval = '+str(Sigval), fontsize = 10)
 n, bins,patches = plt.hist(depcc[:],bins,normed = 1,facecolor = 'g',label = 'mean = '+str(float("{0:.2f}".format(mean_depcc))))
 n, bins,patches = plt.hist(depfc[:],bins,normed = 1,facecolor = 'b',alpha = 0.5,label = 'mean = '+str(float("{0:.2f}".format(mean_depfc))))
# n, bins,patches = plt.hist(depcc[:],bins,normed = 1,facecolor = 'g',label = 'CC')
# n, bins,patches = plt.hist(depfc[:],bins,normed = 1,facecolor = 'b',alpha = 0.5,label = 'FC')
 plt.scatter(mean_depcc, 0, c= 'g')
 plt.scatter(mean_depfc, 0 , c = 'b')
 plt.xlabel('1800Z 1% 1.5-meter \n Temperature (K)')
 plt.ylabel('Probability density')
 plt.legend(fontsize = 8)
 plt.subplot(2,4,2)
 depcc = all_cube_mslp_1800_cc/100.
 depfc = all_cube_mslp_1800_fc/100.
 mean_depcc = np.average(depcc)
 mean_depfc = np.average(depfc)
 print 'pressure diff FC - CC = '+str(float(mean_depfc-mean_depcc))
 correl = stat.ttest_ind(depcc,depfc,equal_var = False)
 Sigval = 1.0 - correl[1]
 Sigval = float("{0:.3f}".format(Sigval))
 bins = np.linspace(np.min([np.min(depfc[:]),np.min(depcc[:])]), np.max([np.max(depfc[:]),np.max(depcc[:])]), 25)
 plt.title('(b) - Confidence Interval = '+str(Sigval), fontsize = 10)
 n, bins,patches = plt.hist(depcc[:],bins,normed = 1,facecolor = 'g',label = 'mean = '+str(float("{0:.2f}".format(mean_depcc))))
 n, bins,patches = plt.hist(depfc[:],bins,normed = 1,facecolor = 'b',alpha = 0.5,label = 'mean = '+str(float("{0:.2f}".format(mean_depfc))))
# n, bins,patches = plt.hist(depcc[:],bins,normed = 1,facecolor = 'g',label = 'CC')
# n, bins,patches = plt.hist(depfc[:],bins,normed = 1,facecolor = 'b',alpha = 0.5,label = 'FC')
 plt.scatter(mean_depcc, 0, c= 'g')
 plt.scatter(mean_depfc, 0 , c = 'b')
 plt.xlabel('1800Z 99th perc \n surface pressure (hPa)')
 plt.legend(fontsize = 8)
 plt.subplot(2,4,3)
 depcc = all_cube_10u_1800_cc
 depfc = all_cube_10u_1800_fc
 mean_depcc = np.average(depcc)
 mean_depfc = np.average(depfc)
 print 'u diff FC - CC = '+str(float(mean_depfc-mean_depcc))
 correl = stat.ttest_ind(depcc,depfc,equal_var = False)
 Sigval = 1.0 - correl[1]
 Sigval = float("{0:.3f}".format(Sigval))
 bins = np.linspace(np.min([np.min(depfc[:]),np.min(depcc[:])]), np.max([np.max(depfc[:]),np.max(depcc[:])]), 25)
 plt.title('(c) - Confidence Interval = '+str(Sigval), fontsize = 10)
 n, bins,patches = plt.hist(depcc[:],bins,normed = 1,facecolor = 'g',label = 'mean = '+str(float("{0:.2f}".format(mean_depcc))))
 n, bins,patches = plt.hist(depfc[:],bins,normed = 1,facecolor = 'b',alpha = 0.5,label = 'mean = '+str(float("{0:.2f}".format(mean_depfc))))
# n, bins,patches = plt.hist(depcc[:],bins,normed = 1,facecolor = 'g',label = 'CC')
# n, bins,patches = plt.hist(depfc[:],bins,normed = 1,facecolor = 'b',alpha = 0.5,label = 'FC')
 plt.scatter(mean_depcc, 0, c= 'g')
 plt.scatter(mean_depfc, 0 , c = 'b')
 plt.xlabel('1800Z 99% 10-meter \n wind speed (m/s)')
 plt.legend(fontsize = 8)
 plt.subplot(2,4,4)
 depcc = all_cube_10u_cubed_1800_cc
 depfc = all_cube_10u_cubed_1800_fc
 mean_depcc = np.average(depcc)
 mean_depfc = np.average(depfc)
 print 'u3 diff FC - CC = '+str(float(mean_depfc-mean_depcc))
 correl = stat.ttest_ind(depcc,depfc,equal_var = False)
 Sigval = 1.0 - correl[1]
 Sigval = float("{0:.3f}".format(Sigval))
 bins = np.linspace(np.min([np.min(depfc[:]),np.min(depcc[:])]), np.max([np.max(depfc[:]),np.max(depcc[:])]), 25)
 plt.title('(d) - Confidence Interval = '+str(Sigval), fontsize = 10)
 n, bins,patches = plt.hist(depcc[:],bins,normed = 1,facecolor = 'g',label = 'mean = '+str(float("{0:.2f}".format(mean_depcc))))
 n, bins,patches = plt.hist(depfc[:],bins,normed = 1,facecolor = 'b',alpha = 0.5,label = 'mean = '+str(float("{0:.2f}".format(mean_depfc))))
# n, bins,patches = plt.hist(depcc[:],bins,normed = 1,facecolor = 'g',label = 'CC')
# n, bins,patches = plt.hist(depfc[:],bins,normed = 1,facecolor = 'b',alpha = 0.5,label = 'FC')
 plt.scatter(mean_depcc, 0, c= 'g')
 plt.scatter(mean_depfc, 0 , c = 'b')
 plt.xlabel('1800Z mean 10-meter \n wind speed cubed [w.s.c.] (m3/s3)')
 plt.legend(fontsize = 8)
 plt.subplots_adjust(hspace = 0.5, wspace = 0.4)
 plt.show()
# all_cube_T15_1800_fc = np.delete(all_cube_T15_1800_fc, killlist, axis = 0)
# all_cube_mslp_1800_fc = np.delete(all_cube_mslp_1800_fc, killlist, axis = 0)
# all_cube_10u_1800_fc = np.delete(all_cube_10u_1800_fc, killlist, axis = 0)
# all_cube_10u_cubed_1800_fc = np.delete_all_cube_10u_cubed_1800_fc, killlist, axis = 0)

if "__name__" == "__main__":
   main(xstart,xend,ystart,yend)
