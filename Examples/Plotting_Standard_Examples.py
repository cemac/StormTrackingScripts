"""Example Script

    How to use plotting tools
   :copyright: © 2018 University of Leeds.
   :license: BSD-2 Clause.

To use:

    python Plotting_Standard_Examples.py

.. CEMAC_stomtracking:
   https://github.com/cemac/StormTrackingScripts
"""

import StormScriptsPy3 as SSP3
import matplotlib.pyplot as plt

# Give file names
fc_csv = 'data/fc_example_standard.csv'
cc_csv = 'data/cc_example_standard.csv'
c = SSP3.S_Plotter.stormstats(fc_csv, cc_csv)
# You can generate a standard plot simply by:
c.Standard_Correl2(c.df_fc, c.df_cc, 'Example')
c.Standard_pdfs(c.df_fc, c.df_cc, 'Example')
c.Standard_coldpool(c.df_fc, c.df_cc, 'Example')
c.Standard_histograms(c.df_fc, c.df_cc, 'Example')
c.Standard_Correl2(c.df_fc, c.df_cc, 'Example')
c.Standard_Correl1(c.df_fc, c.df_cc, 'Example')
c.Standard_shear_pdf(c.df_fc, c.df_cc, 'Example')
c.Standard_precip_correl(c.df_fc, c.df_cc, 'Example')
