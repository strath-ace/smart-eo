import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime, timedelta
import matplotlib.gridspec as gridspec
from shapely.geometry import Point
from cartopy.io import shapereader
from tqdm import tqdm
from rasterio.features import rasterize
import json
import scipy
from astropy.convolution import Gaussian2DKernel, interpolate_replace_nans

from commons import *

################### Set conditions #################

lon_min, lon_max = -2, 4
lat_min, lat_max = 52, 62

aspect = 1 / np.cos(np.deg2rad(((lat_max + lat_min)/2)))

start = datetime(2018, 1, 1)
end = datetime(2025, 1, 1)

all_chemicals = ["NO2", "CO", "HCHO" ,"SO2"] ## "CH4"

####################################################

for chemical in all_chemicals:

    print("Processing", chemical, "data")

    ################### Get data #################

    chemical_dataset = np.load("PROCESSED_DATA/all_days_"+chemical+".npz")
    data = np.array(chemical_dataset["data"])#, dtype=np.float64)
    dates = chemical_dataset["dates"]

    data[data > 100] = np.nan

    data = normalize(data)

    # data = np.power(data, -10)
    # data = np.sqrt(data)

    # data = scipy.ndimage.gaussian_filter(data, 1)

    data[data > np.nanmean(data)+3*np.nanstd(data)] = np.nan
    data[data < np.nanmean(data)-3*np.nanstd(data)] = np.nan

    # kernel = Gaussian2DKernel(x_stddev=1)
    # data_new = []
    # for x in tqdm(data[-50:]):
    #     # data_new.append(interpolate_replace_nans(x, kernel))
    #     data_new.append(normalize(kernel_process(x, 30)))

    # data = data_new

    # data[data > np.nanmean(data)+3*np.nanstd(data)] = np.nan
    # data[data < np.nanmean(data)-3*np.nanstd(data)] = np.nan

    # data_new = []
    # for i, x in enumerate(data):
    #     print(i, "of", len(data))
    #     temp = normalize(x)
    #     temp = land_mask(temp, lon_min, lon_max, lat_min, lat_max)
    #     temp = kernel_process(temp, 30)
    #     data_new.append(temp)

    # data_new = np.array(data_new)
    # derived = np.nanmean(data_new, axis=0)

    # plt.hist(data_new.flatten(), bins=100)
    # plt.yscale("log")
    # plt.savefig("test.png")

    # np.save("der", derived)



    derived = np.nanmean(data, axis=0)




    ################### Process into best viewing #################

    if chemical == "NO2":
        derived = land_mask(derived, lon_min, lon_max, lat_min, lat_max)
        derived = kernel_process(derived, 30, normalize=True)
        derived = normalize(derived)
        derived = np.power(10, derived)
    elif chemical == "HCHO":
        derived = land_mask(derived, lon_min, lon_max, lat_min, lat_max)
        derived = kernel_process(derived, 30, normalize=True)
        derived = normalize(derived)
        derived = np.power(10, derived)
    elif chemical == "SO2":
        derived = land_mask(derived, lon_min, lon_max, lat_min, lat_max)
        derived = kernel_process(derived, 30, normalize=True)
        derived = normalize(derived)
        derived = np.power(10, derived)
    elif chemical == "CO":
        derived = land_mask(derived, lon_min, lon_max, lat_min, lat_max)
        derived = kernel_process(derived, 30, normalize=True)
        derived = normalize(derived)
        derived = np.power(10, derived)
    elif chemical == "CH4":
        derived = land_mask(derived, lon_min, lon_max, lat_min, lat_max)
        derived = kernel_process(derived, 30, normalize=True)
        derived = normalize(derived)
        derived = np.power(10, derived)
        
    # Normalize
    derived = normalize(derived)

    np.save("PROCESSED_DATA/viewable_"+chemical, derived)

    ################### Plot emissions #################

    # Create figure and axis with a PlateCarree projection
    fig, ax = plt.subplots(figsize=(7, aspect*7), 
                        subplot_kw={'projection': ccrs.PlateCarree()},
                        layout="constrained")

    # Plot the array as an image
    img = ax.imshow(derived, extent=[lon_min, lon_max, lat_min, lat_max], 
                    origin='upper', transform=ccrs.PlateCarree(), 
                    cmap='Reds')

    # Add coastlines and grid lines
    ax.coastlines()
    ax.gridlines(draw_labels=True, linestyle='--', alpha=0)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')

    ax.set_aspect(1 / np.cos(np.deg2rad(((lat_max + lat_min)/2))))
    # ax.set_aspect("equal")

    # Add a colorbar
    cb = plt.colorbar(img, orientation='vertical')
    # cb.ax.tick_params(labelsize=16)

    ################### Plot Oil Platforms #################

    # plot_platforms(ax)

    ################### Set extent and save #################

    extra_degrees = 0.5
    ax.set_extent([lon_min-extra_degrees,
                    lon_max+extra_degrees,
                    lat_min-(extra_degrees/aspect),
                    lat_max+(extra_degrees/aspect)
                ], crs=ccrs.PlateCarree())

    plt.title(chemical + " Sentinel-5P overview of dataset")
    plt.savefig("results/map_"+chemical+".png", dpi=500)
    plt.clf()