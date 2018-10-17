# Starting Information #

These scripts are using information produced by Tracking code written by Thorwald Stein and updates by Julia Crook.

## Storm Information Files ##

<hr>

The tracking algorithm outputs a text file with a name related to the matching input filename and
containing details of the storms and cells within each storm identified in each timestep, eg.

storm 375 area=10 centroid=202.00,19.00 box=199.0,18.0,5,1 life=0 u=0.00 v=0.00 mean=0.000435 min=0.000307 max=0.000525 accreted=-999 parent=-999 child=-999 cell=441,442

cell 441 stormid=375 centroid=202.00,19.00 maxr=0.000525

<hr>

For example if we wanted to look at all the storms in the hourly IMPALA data 4km precipitation tracking 12km hourly data we might have 25000 files for each timestep each of those files may contain information listed in the manner above for a couple 1000 storms.

<hr>

The number following storm/cell is the storm/cell id

**area:**     the number of grid cells that met the threshold in this storm

**centroid:** the centre of the storm determined as the mean of [latix, lonix] of all grid cells making up that storm (or the centre of the cell)

**box:**      defines the rectangle around the storm [minlatix, minlonix, nlats, nlons]

**life:**     the number of timesteps this storm has been seen minus 1, i.e. if this is zero the storm
          has only been seen in one image so far; this should increment in each successive timestep until the storm disappears.
          If the storm was created by splitting from a parent, life will be the life of the parent when this storm was created
          and will increment thereafer.
u,v       the velocity at this timestep as determined from pattern correlation (not to be used as velocity of the storm)

**mean:**     the mean value in the image for this storm

**min:**      the minimum value in the image for this storm

**max:**      the maximum value in the image for this storm

**accreted:** the storm ids that merged with this one at this timestep

**parent:**   if this storm split from another storm at this timestep this is the id of the parent storm

**child:**    the ids of storms that split off this storm at this timestep.

**cell:**     the ids of any cells within this storm

Note that a value of -999 means an invalid value, eg no children, no parent etc.

**maxr:**     for a cell this is the minimum or maximum value depending on whether you are looking at data < or > threshold

<hr>

storminbox.py is hunting through storms to find information on storms over a certain size (area) at a certain location (centroid) over a certain time period (in the file name) and then stores a list of storms and small amount of information such as time, location, mean OLR.
