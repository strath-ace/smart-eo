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

from commons import *

################### Set conditions #################

start = datetime(2018, 1, 1)
end = datetime(2025, 1, 1)

all_chemicals = ["NO2", "HCHO", "CH4", "SO2", "CO"]

################### Get data #################

edges = []
current = start
while current <= end:
    edges.append(current.date().isoformat())
    current += timedelta(days=1)

for chemical in all_chemicals:
    c=0
    all_imgs = []
    dates = []
    averages = []
    trigger = False
    for edge in tqdm(edges):
        if edge == "2018-04-29":
            trigger = True
        try:
            img = np.load("SENTINEL/numpy_"+chemical+"/"+edge+".npy")
            if trigger:
                averages.append(np.sum(np.isnan(img))/np.prod(np.shape(img)))
            all_imgs.append(img)
            dates.append(edge)
        except:
            # print(chemical, "missing data on", edge)
            c += 1
            pass
    all_imgs = np.array(all_imgs)

    print("###########")
    print(c, "missing for", chemical)
    print(1-np.nanmean(averages), "is average amount of missing in img")
    print("Gathered", len(all_imgs), "data points for", chemical)
    print("###########")

    np.savez("PROCESSED_DATA/all_days_"+chemical, data=all_imgs, dates=dates)
    del all_imgs