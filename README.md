# StormTrackingScripts #

[![GitHub release](https://img.shields.io/badge/release-v.1.0-blue.svg)(
https://github.com/cemac/StormTrackingScripts/releases/tag/v.1.0)

**ORIGINAL VERSION**

## Description ##

Development of Storm Tracking Scripts from ICAS dynamics group to improve functionality and "genericise".

## Requirements ##

 * [Python 2.7](https://www.anaconda.com/download/)
 * [Iris](https://scitools.org.uk/iris/docs/latest)

## Usage ##

Currently run in series:
* run storm_in_box as fc_storm_in_box first to give you the storms over a given area.
* Then run the py codes in the dataminer and CAPE… folders.
* Then you run the c2c4…py file in the main rainfall_tracks directory
* run via
  ```python
  import __.main(345,375,10,18,5000)
  ```

## Acknowledgements ##

These scripts have been developed based off the work done by the [Institute of Climate and Atmospheric Science (ICAS)](http://www.see.leeds.ac.uk/research/icas/) [dynamics](http://www.see.leeds.ac.uk/research/icas/research-themes/atmosphere/) group and the University of Leeds. This work was originally part of the [African Monsoon Multidisciplinary Analysis](https://www.amma2050.org/) (AMMA-2050) project.
