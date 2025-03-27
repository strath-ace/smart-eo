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
import csv
from collections import Counter

def load_json(file_name):
    with open(file_name) as f:
        output = json.load(f)
    return output

def save_json(file_name, data):
    with open(file_name,'w') as f:
        json.dump(data, f)

def csv_input(file_name):
    with open(file_name, "r") as f:
        read_obj = csv.reader(f)
        output = []
        for row in read_obj:
            output.append(row)   
    f.close()
    return output

def csv_output(file_name, data_send):
    # print(np.shape(np.array(data_send,dtype=str)))
    with open(file_name, "w", newline="") as f:
        # csv.writer(f?).writerows(data_send)
        for item in data_send:
            csv.writer(f).writerow(item)
        f.close()
        return

def to_timestamp(date_str):
    return int(datetime.strptime(date_str, "%Y-%m-%d").timestamp())


def normalize(data):
    if 0 in np.shape(data):
        return np.array(data)
    return (data - np.nanmin(data)) / (np.nanmax(data)-np.nanmin(data))


def land_mask(derived, lon_min, lon_max, lat_min, lat_max):

    lons = np.linspace(lon_min, lon_max, derived.shape[1])
    lats = np.linspace(lat_min, lat_max, derived.shape[0])

    land_shp = shapereader.natural_earth(resolution='10m', category='physical', name='land')
    land_polygons = list(shapereader.Reader(land_shp).geometries())

    lon_grid, lat_grid = np.meshgrid(lons, lats)

    land_mask = rasterize(
        [(poly, 1) for poly in land_polygons],  # Assign a value of 1 to land pixels
        out_shape=(len(lats), len(lons)),  # Output shape matches data resolution
        transform=[(lon_max - lon_min) / len(lons), 0, lon_min, 0, (lat_min - lat_max) / len(lats), lat_max]
    )

    derived[land_mask == 1] = np.nan
    return derived


def kernel_process(derived, kernel_size, normalize=True):
    derived2 = np.zeros_like(derived)
    derived2[derived2 == 0] = np.nan
    derived = np.pad(derived, kernel_size, 'constant', constant_values=np.nan)

    if normalize:
        # Normalize the kernel
        for x in tqdm(range(kernel_size,np.shape(derived2)[0]+kernel_size), desc="Kernel"):
            for y in range(kernel_size,np.shape(derived2)[1]+kernel_size):
                kernel = derived[x-kernel_size:x+kernel_size+1,y-kernel_size:y+kernel_size+1]
                derived2[x-kernel_size,y-kernel_size] = (derived[x,y] - np.nanmin(kernel)) / (np.nanmax(kernel)-np.nanmin(kernel))
    else:
        # Divide by mean of the kernel
        for x in tqdm(range(kernel_size,np.shape(derived2)[0]+kernel_size), desc="Kernel"):
            for y in range(kernel_size,np.shape(derived2)[1]+kernel_size):
                kernel = derived[x-kernel_size:x+kernel_size+1,y-kernel_size:y+kernel_size+1]
                derived2[x-kernel_size,y-kernel_size] = (derived[x,y] - np.nanmean(kernel)) / np.nanstd(kernel)
    return derived2


def plot_platforms(ax):

    all_data_rigs = load_json("NSTA/Surface_Points_(WGS84).geojson")
    data_rigs = [dat for dat in all_data_rigs["features"] if dat["properties"]["INF_TYPE"] == "PLATFORM" or dat["properties"]["INF_TYPE"] == "FPSO"]
    rig_lats = [entry["geometry"]["coordinates"][1] for entry in data_rigs]
    rig_lons = [entry["geometry"]["coordinates"][0] for entry in data_rigs]

    ax.scatter(rig_lons, rig_lats, alpha=0.3, marker="x")



def plot_many_firms_observations(file_name, filtered_rigs, before_date=None, after_date=None):

    num_rigs = len(filtered_rigs)
    cols = 10
    rows = int(np.ceil(num_rigs / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2), 
                            subplot_kw={'projection': ccrs.PlateCarree()},
                            layout="constrained")


    for ax, rig in zip(axes.flatten(), filtered_rigs):

        lats = []
        lons = []
        for hot_spot in rig["FIRMS_observations"]:
            if after_date != None:
                condition1 = to_timestamp(after_date) < to_timestamp(hot_spot["acq_date"])
            else:
                condition1 = True
            if before_date != None:
                condition2 = to_timestamp(hot_spot["acq_date"]) < to_timestamp(before_date)
            else:
                condition2 = True
            if condition1 and condition2:
                lats.append(hot_spot["latitude"])
                lons.append(hot_spot["longitude"])

        ax.scatter(lons, lats, c="red", alpha=0.2)

        # print(lats, lons)
        x,y = rig["coordinates"]
        ax.scatter(x, y, marker="x", c="blue")
        extra_degrees = 0.1
        aspect_temp = 1 / np.cos(np.deg2rad(y))
        ax.set_extent([x-extra_degrees,
                        x+extra_degrees,
                        y-(extra_degrees/aspect_temp),
                        y+(extra_degrees/aspect_temp)
                    ], crs=ccrs.PlateCarree())
        ax.set_aspect(1 / np.cos(np.deg2rad(y)))
        # ax.gridlines(draw_labels=True, linestyle='--', alpha=1)
        ax.set_title(rig["name"])

    plt.savefig(file_name, dpi=300)
    plt.clf()


def plot_many_firms_observations_time_series(file_name, filtered_rigs):

    num_rigs = len(filtered_rigs)
    cols = 10
    rows = int(np.ceil(num_rigs / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2),
                            layout="constrained")
    max_val = 0
    for ax, rig in zip(axes.flatten(), filtered_rigs):

        date_strings = [(hot_spot["acq_date"]) for hot_spot in rig["FIRMS_observations"]]

        timestamps = [int(datetime.strptime(date, "%Y-%m-%d").timestamp()) for date in date_strings]
        year_month_labels = [datetime.utcfromtimestamp(ts).strftime("%Y-%m") for ts in timestamps]

        counts = Counter(year_month_labels)
        
        years = range(2012, 2025)
        months = range(1, 13)

        all_months = [f"{year}-{month:02d}" for year in years for month in months]

        # Ensure all months exist in data (fill missing months with zero)
        histogram_data = [counts.get(month, 0) for month in all_months]

        ax.bar(all_months, histogram_data, color="red", alpha=0.7)

        ax.set_xlim([all_months[0], all_months[-1]])
        ax.set_xticks([all_months[0], all_months[-1]], [2018, 2025])
        if np.amax(histogram_data) > max_val:
            max_val = np.amax(histogram_data)

        ax.set_title(rig["name"])

    for ax in axes.flatten():
        ax.set_ylim([0, max_val])

    plt.savefig(file_name, dpi=300)
    plt.clf()


def plot_combined_firms_observations_time_series(file_name, filtered_rigs_list, rigs_list_names, colors):
    fig = plt.figure(figsize=(7,7), layout="constrained")
    all_data = []
    for filtered_rigs, name, color in zip(filtered_rigs_list, rigs_list_names, colors):
        date_strings = []
        for rig in filtered_rigs:
            for hot_spot in rig["FIRMS_observations"]:
                date_strings.append(hot_spot["acq_date"])

        timestamps = [int(datetime.strptime(date, "%Y-%m-%d").timestamp()) for date in date_strings]
        year_month_labels = [datetime.utcfromtimestamp(ts).strftime("%Y-%m") for ts in timestamps]
        counts = Counter(year_month_labels)
        # years = range(2012, 2026)
        years = range(2012, 2025)
        months = range(1, 13)
        all_months = [f"{year}-{month:02d}" for year in years for month in months]
        histogram_data = [counts.get(month, 0) for month in all_months]
        # all_data.append([all_months, np.array(histogram_data), name, color])
        if color == "black":
            alpha=0.7
        else:
            alpha = 1
        # plt.bar(all_months, np.array(histogram_data)/len(filtered_rigs), label=name, color=color, alpha=alpha)
        plt.bar(all_months, np.array(histogram_data), label=name, color=color, alpha=alpha)

    # plt.bar(all_data[0][0], all_data[0][1]+all_data[1][1]+all_data[2][1], label=all_data[0][2], color=all_data[0][3])
    # plt.bar(all_data[2][0], all_data[1][1]+all_data[2][1], alpha=0.7, label=all_data[2][2], color=all_data[2][3])
    # plt.bar(all_data[1][0], all_data[1][1], alpha=0.7, label=all_data[1][2], color=all_data[1][3])
    

    indices = np.arange(0, len(all_months), 12)
    all_months = np.array(all_months)[indices]
    display_month = []
    for month in all_months:
        display_month.append(month[:4])

    all_months = all_months.tolist()
    all_months.append(f"2025-01")
    display_month.append("2025")
    # all_months.append()
    # plt.xlim([0,144])
    plt.xticks(all_months, display_month)
    plt.legend()
    plt.xlabel("Time (Months)")
    plt.ylabel("Summed number of FIRMS observations")
    plt.title("Time series of summed FIRMS observations segregated by month")
    plt.savefig(file_name)
    plt.clf()



def trim_nan_edges(arr):
    mask = ~np.isnan(arr)
    if not mask.any():
        return np.array([])  # Return empty array if all NaN
    row_nonzero, col_nonzero = np.where(mask)
    row_min, row_max = row_nonzero.min(), row_nonzero.max()
    col_min, col_max = col_nonzero.min(), col_nonzero.max()
    return arr[row_min:row_max+1, col_min:col_max+1]