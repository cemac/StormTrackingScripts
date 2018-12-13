"""Example Script

    How to use plotting tools
   :copyright: © 2018 University of Leeds.
   :license: BSD-2 Clause.

To use:

    python Plotting_Examples.py

.. CEMAC_stomtracking:
   https://github.com/cemac/StormTrackingScripts
"""

import StormScriptsPy3 as SSP3
import matplotlib.pyplot as plt

# Give file names
fc_csv = 'data/fc_test_standard.csv'
cc_csv = 'data/cc_test_standard.csv'
c = SSP3.S_Plotter.stormstats(fc_csv, cc_csv)
# Create extra variable required
df_fc = SSP3.S_Plotter.createvars(fc_csv)
df_cc = SSP3.S_Plotter.createvars(cc_csv)

# Define panel numbers, figletters, lables and variables to plot
panelno = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

figletter = ['a) ', 'b) ', 'c) ', 'd) ', 'e) ', 'f) ', 'g) ', 'h) ',
             'i) ', 'j) ', 'k) ']

xlab = ['1800 UTC 99p precipitation rate (mm/hr)',
        '1200 UTC mean zonal \n wind shear (m/s)',
        '1200 UTC mean wind \n shear magnitude (m/s)',
        '1800 UTC mean Total Column \n Water Vapor (kg/m2)',
        '1200 UTC mean MU-CAPE (J)', '1200 UTC mean MU-CIN (J)',
        '1200 UTC mean 700 hPa RH (%)', '1800 UTC mean OLR (W/m2)',
        '1800 UTC min omega (Pa/s)',
        '1800 UTC 1-hour total rainfall (kg)',
        "1800 UTC storm area ('000,000 km2)",
        '1800 UTC cold pool marker (K)']

var1c = [df_cc.precip99, df_cc.max_shear, df_cc.hor_shear,
         df_cc.shear_TCW_eve, df_cc.CAPE_CAPE, df_cc.CAPE_CIN,
         df_cc.Tephi_RH650, df_cc.OLRs, df_cc.omega_1800_1p,
         df_cc.precipvol, df_cc.area*100*100/1000000, df_cc.cold]

var1f = [df_fc.precip99, df_fc.max_shear, df_fc.hor_shear,
         df_fc.shear_TCW_eve, df_fc.CAPE_CAPE, df_fc.CAPE_CIN,
         df_fc.Tephi_RH650, df_fc.OLRs, df_fc.omega_1800_1p,
         df_fc.precipvol, df_fc.area*100*100/1000000, df_fc.cold]

# Use the required plotter
for i in panelno:
    SSP3.S_Plotter.shear_pdfs(i, figletter[i-1], var1c[i-1],
                              var1f[i-1], xlab[i-1])

plt.tight_layout()
plt.savefig(figname + '_PDF.png')
plt.clf()
