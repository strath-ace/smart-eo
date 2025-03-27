# Information regarding dataset

Releasing any of the data in this dataset publicly will require correct licensing with regards to the orignal datasets.

#### SENTINEL_raw_NO2 (Sheet 1)

Contains all assets with asset ownership details, coordinates and the raw NO2 values directly from European Space Agencies SENTINEL-5P satellite with no processing. The data is the L2 layer tropospheric NO2 values.

At the coordinates of each asset all cells within the original NO2 dataset are subdivided by 0.1 degrees around the coordinates unless a different asset is closer. Therefore, no cells are shared by oil rigs. The mean values for each month in these subdivide asset areas are given in this sheet document.

#### SENTINEL_normalised_NO2 (Sheet 2)

Contains all assets with asset ownership details, coordinates and the raw NO2 values directly from European Space Agencies SENTINEL-5P satellite with no processing. The data is the L2 layer tropospheric NO2 values.

At the coordinates of each asset all cells within the original NO2 dataset are subdivided by 0.1 degrees around the coordinates unless a different asset is closer. Therefore, no cells are shared by oil rigs. Each subvdivided asset area is normalised by:

1. Get mean and std of nearest 10 oil rigs not including the currently selected oil rig for each month
2. Normalise the current oil rig by this mean and std for each month:
```
current_oil_rig_per_month = (current_oil_rig_per_month - neighbour_mean_per_month)/neighbour_std_per_month
```
3. Crop normalised values by 3 standard deviations to remove extreme values
4. Multiply by global standard deviation and add global mean to convert back to realistic values

The mean values after this normalisation for each month in these subdivide asset areas are given in this sheet document.

#### FIRMS_counts (Sheet 3)

Contains all assets with asset ownership details, coordinates and the number of FIRMS/VIIRS satellite thermal anomoly observations. The specific satellite dataset from FIRMS/VIIRS is that created by the Suomi National Polar-Orbiting Partnership (S-NPP) satellite.

At the coordinates of each asset all anomolies within the original S-NPP dataset are subdivided by 0.1 degrees around the coordinates unless a different asset is closer. Therefore, no cells are shared by oil rigs. The number of anomolies detected per month are given in this sheet document.

#### FIRMS_counts (Sheet 4)

Contains all assets with asset ownership details, coordinates and the average brightness of flares of FIRMS/VIIRS satellite thermal anomoly observations. The specific satellite dataset from FIRMS/VIIRS is that created by the Suomi National Polar-Orbiting Partnership (S-NPP) satellite.

At the coordinates of each asset all anomolies within the original S-NPP dataset are subdivided by 0.1 degrees around the coordinates unless a different asset is closer. Therefore, no cells are shared by oil rigs. The mean brightness of anomolies detected per month are given in this sheet document.

#### OIL_price_monthly (Sheet 5)

Contains the Brent Oil Spot Price from [here](https://www.eia.gov/dnav/pet/hist/RBRTED.htm). Divided into monthly data to be aligned with all other sheets data.

#### Reasons for missing values

Missing values occur for a number of reasons:
- Satellites or datasets have temporarly stopped producing data
- At northern latitudes such as some of the more northern oil rigs in the north sea do not recieve enough light in the winter to correctly detect NO2 and therefore these are removed. Similarly during the summer to long a day means less thermal anomoly recordings from FIRMS/VIIRS due to the requirement of night.
- When an oil rig is surrounded by others it may not have enough datapoints to produce a valid dataset and therefore is empty in the excel.