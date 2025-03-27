import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from commons import *
from datetime import datetime
from collections import Counter
import os
import seaborn as sns
import pandas as pd
import math
from scipy.stats import median_abs_deviation
from skimage.transform import resize

lon_min, lon_max = -2, 4
lat_min, lat_max = 52, 62
# lat_min, lat_max = 55.5, 62

aspect = 1 / np.cos(np.deg2rad(((lat_max + lat_min)/2)))

def point_in_bounds(coords, lon_min,lon_max,lat_min,lat_max):
    condition_lon = rig["coordinates"][0] > lon_min and rig["coordinates"][0] < lon_max
    condition_lat = rig["coordinates"][1] > lat_min and rig["coordinates"][1] < lat_max
    return condition_lon and condition_lat

# rigs = load_json("PROCESSED_DATA/surface_points_plus_FIRMS_count.json")
rigs = load_json("PROCESSED_DATA/surface_points_clean.json")
filtered_rigs = []
for rig in rigs:
    if point_in_bounds(rig["coordinates"], lon_min,lon_max,lat_min,lat_max):
    # if point_in_bounds(rig["coordinates"], -2,4,55.5,62):       # Only top of north sea
    # if point_in_bounds(rig["coordinates"], -2,4,52,55.5):       # Only bottom of north sea
        filtered_rigs.append(rig)

og_rigs = filtered_rigs.copy()
points = np.array([[rig["coordinates"][0], rig["coordinates"][1]] for rig in filtered_rigs])





# NOAA-20(JPSS-1)
all_data_a = load_json("FIRMS/DL_FIRE_J1V-C2_587751/fire_archive_J1V-C2_587751.json")
data_a = [dat for dat in all_data_a if dat["type"] == "3"]
# NOAA-21(JPSS-2)
# all_data_b = load_json("DL_FIRE_J2V-C2_587752/fire_nrt_J2V-C2_587752.json")
# data_b = [dat for dat in all_data_b if dat["type"] == "3"]
# S-NPP
all_data_c = load_json("FIRMS/DL_FIRE_SV-C2_587753/fire_archive_SV-C2_587753.json")
data_c = [dat for dat in all_data_c if dat["type"] == "3"]

datasets = [data_c]
dataset_names = [
    "Satellite dataset NOAA-20 (JPSS-1)",
    "Satellite dataset S-NPP"
]




def grid_to_real(i, j, width, height):
    real_x = lon_min + (i / (width - 1)) * (lon_max - lon_min)
    real_y = lat_min + (j / (height - 1)) * (lat_max - lat_min)
    return np.array([real_x, real_y])

def real_to_grid(x, y, height, width):
    i = int((x - lon_min) / (lon_max - lon_min) * (width - 1))
    j = int((y - lat_min) / (lat_max - lat_min) * (height - 1))
    return np.array([i, j])

def closest_point(p, points):
    distances = np.linalg.norm(points - p, axis=1)
    closest = np.argmin(distances)
    if distances[closest] <= 0.1155:
        return closest
    else:
        return np.nan
    

def closest_points_all(p, points):
    distances = np.linalg.norm(points - p, axis=1)
    indx = np.argsort(distances)
    return indx

for i in range(len(filtered_rigs)):
    filtered_rigs[i].update({"FIRMS_observations": []})

for data in datasets:
    lats = [entry["latitude"] for entry in data]
    lons = [entry["longitude"] for entry in data]

    for lat, lon, dat in zip(lats, lons, data):
        chosen_rig = closest_point([lon, lat], points)
        if not np.isnan(chosen_rig):
            chosen_rig = int(chosen_rig)
            filtered_rigs[chosen_rig]["FIRMS_observations"].append(dat)


save_json("PROCESSED_DATA/surface_points_plus_FIRMS.json", filtered_rigs)





















filtered_rigs_list = []

############### FIRMS OBSERVATION ALL RIGS ##################

filtered_rigs = og_rigs.copy()
filtered_rigs_list.append(filtered_rigs)

plot_many_firms_observations("results/FIRMS/FIRMS_all.png", filtered_rigs)
# plot_many_firms_observations_time_series("results/FIRMS/FIRMS_time_all.png", filtered_rigs)


############### FIRMS OBSERVATION PLC RIGS ##################

filtered_rigs = og_rigs.copy()

new_filtered_rigs = []
for rig in filtered_rigs:
    if rig["parent_company_type"] == "PLC":
        new_filtered_rigs.append(rig)
filtered_rigs = new_filtered_rigs
filtered_rigs_list.append(filtered_rigs)

plot_many_firms_observations("results/FIRMS/FIRMS_plc.png", filtered_rigs)#, after_date="2022-01-01")
# plot_many_firms_observations_time_series("results/FIRMS/FIRMS_time_plc.png", filtered_rigs)

############### FIRMS OBSERVATION LTD RIGS ##################

filtered_rigs = og_rigs.copy()

new_filtered_rigs = []
for rig in filtered_rigs:
    if rig["parent_company_type"] == "LTD":
        new_filtered_rigs.append(rig)
filtered_rigs = new_filtered_rigs
filtered_rigs_list.append(filtered_rigs)

plot_many_firms_observations("results/FIRMS/FIRMS_ltd.png", filtered_rigs)#, after_date="2022-01-01")
# plot_many_firms_observations_time_series("results/FIRMS/FIRMS_time_ltd.png", filtered_rigs)

############### FIRMS OBSERVATION SOE RIGS ##################

filtered_rigs = og_rigs.copy()

new_filtered_rigs = []
for rig in filtered_rigs:
    if rig["parent_company_type"] == "SOE":
        new_filtered_rigs.append(rig)
filtered_rigs = new_filtered_rigs
filtered_rigs_list.append(filtered_rigs)

plot_many_firms_observations("results/FIRMS/FIRMS_soe.png", filtered_rigs)#, after_date="2022-01-01")
# plot_many_firms_observations_time_series("results/FIRMS/FIRMS_time_soe.png", filtered_rigs)



############### FIRMS OBSERVATION Big time series ##################

rigs_list_names = ["All Rigs", "PLC Rigs", "LTD Rigs", "SOE Rigs"]
# indx = [3,1,2]
indx = [1,3,2]
colors = ["black", "skyblue", "sienna", "olivedrab"]
plot_combined_firms_observations_time_series("results/FIRMS/FIRMS_combined_time_series.png",
    [filtered_rigs_list[i] for i in indx],
    [rigs_list_names[i] for i in indx],
    [colors[i] for i in indx]
)

############### SENTINEL OBSERVATION Big time series ##################


filtered_rigs = og_rigs.copy()



chemical = "NO2"
chemical_dataset = np.load("PROCESSED_DATA/all_days_"+chemical+".npz")
SENTINEL_DATA = np.array(chemical_dataset["data"])#, dtype=np.float64)
date_strings = chemical_dataset["dates"]

# SENTINEL_DATA[SENTINEL_DATA > 100] = np.nan

height = np.shape(SENTINEL_DATA)[1]
width = np.shape(SENTINEL_DATA)[2]

print("Overall dataset size", height, width)

grid = np.zeros((height,width))
grid[grid == 0] = np.nan

points = np.array([[rig["coordinates"][0], rig["coordinates"][1]] for rig in filtered_rigs])

# Compute the closest point for each pixel
for y in range(height):
    for x in range(width):
        grid[y, x] = closest_point(grid_to_real(x, y, width, height), points)
grid = np.flip(grid,axis=0)


print(np.shape(SENTINEL_DATA))
SENTINEL_DATA_meaned = np.nanmean(SENTINEL_DATA, axis=0)
print(np.shape(SENTINEL_DATA_meaned))

data_partitioned = []
data_partitioned_shaped = []
for i in range(len(filtered_rigs)):
    data_partitioned.append(SENTINEL_DATA[:,grid == i])
    SENTINEL_DATA_temp = SENTINEL_DATA_meaned.copy()
    SENTINEL_DATA_temp[grid != i] = np.nan
    SENTINEL_DATA_temp = trim_nan_edges(SENTINEL_DATA_temp)
    data_partitioned_shaped.append(SENTINEL_DATA_temp)
del SENTINEL_DATA









date_strings_2 = np.array(date_strings.copy(), dtype="datetime64[D]")
months = date_strings_2.astype('datetime64[M]').astype(int)
years = date_strings_2.astype('datetime64[Y]').astype(int)
indices = months - months[0]

per_rig_emissions = []
for dat in tqdm(data_partitioned):
    temp = []
    for i in range(np.amax(indices)+1):
        # if (i)%12 != 11:
        temp.append(np.nanmean(dat[indices == i]))
    per_rig_emissions.append(temp)

per_rig_emissions = np.array(per_rig_emissions, dtype=np.float64)
# print("Per rig emissions", np.shape(per_rig_emissions))

per_rig_emissions_raw = per_rig_emissions.copy()

################## Normalize by nearby oil rigs ##################

og_rigs = filtered_rigs.copy()
points = np.array([[rig["coordinates"][0], rig["coordinates"][1]] for rig in filtered_rigs])

closest_rigs = []
for i, rig in enumerate(filtered_rigs):
    closest_rigs.append(closest_points_all(rig["coordinates"], points)[1:11])
closest_rigs = np.array(closest_rigs)

mean_val_overall = np.nanmean(per_rig_emissions)
std_val_overall = np.nanstd(per_rig_emissions)

per_rig_emissions_new = per_rig_emissions.copy()

copy_temp = []
copy_temp_norm = []
for i, rig in enumerate(filtered_rigs):
    all_dates = []
    for closest_i in closest_rigs:
        all_dates.append(per_rig_emissions[closest_i])
    all_dates = np.array(all_dates)
    mean_vals = np.nanmean(all_dates, axis=(0,1))
    std_vals = np.nanstd(all_dates, axis=(0,1))
    per_rig_emissions_new[i] = (per_rig_emissions[i] - mean_vals)/std_vals
    per_rig_emissions_new[i][per_rig_emissions[i] > 3] = np.nan
    per_rig_emissions_new[i][per_rig_emissions[i] < -3] = np.nan
    copy_temp_norm.append(per_rig_emissions[i])
    per_rig_emissions_new[i] = (per_rig_emissions_new[i]*std_val_overall)+mean_val_overall
    copy_temp.append(per_rig_emissions[i])
per_rig_emissions = per_rig_emissions_new.copy()


#####################################################


count = {"PLC": 0, "LTD": 0, "SOE": 0}
notice = []
for rig in filtered_rigs:
    count[rig["parent_company_type"]] +=1
    if rig["parent_company_type"] == "PLC":
        notice.append(0)
    if rig["parent_company_type"] == "LTD":
        notice.append(1)
    if rig["parent_company_type"] == "SOE":
        notice.append(2)
notice = np.array(notice)





# copy_temp = np.array(copy_temp)
# copy_temp_norm = np.array(copy_temp_norm)
# for i in range(np.shape(copy_temp)[1]):
#     binners = np.linspace(np.nanmin(copy_temp),np.nanmax(copy_temp),100)
#     plt.hist(copy_temp[notice==0,i][~np.isnan(copy_temp[notice==0,i])].flatten(), bins=binners, label="PLC", alpha=0.5)
#     plt.hist(copy_temp[notice==1,i][~np.isnan(copy_temp[notice==1,i])].flatten(), bins=binners, label="LTD", alpha=0.5)

#     # binners = np.linspace(-3,3,100)
#     # plt.hist(copy_temp_norm[notice==0,i][~np.isnan(copy_temp_norm[notice==0,i])].flatten(), bins=binners, label="PLC", alpha=0.5)
#     # plt.hist(copy_temp_norm[notice==1,i][~np.isnan(copy_temp_norm[notice==1,i])].flatten(), bins=binners, label="LTD", alpha=0.5)
    
#     plt.legend()
#     # plt.xlim([0,np.nanmax(copy_temp)])
#     # plt.ylim([0,10])
#     plt.savefig("temp/tester_"+str(i)+".png")
#     plt.clf()



plc_emissions_sum = np.nanmean(per_rig_emissions[notice == 0],axis=0)
ltd_emissions_sum = np.nanmean(per_rig_emissions[notice == 1],axis=0)
soe_emissions_sum = np.nanmean(per_rig_emissions[notice == 2],axis=0)
x = np.arange(len(plc_emissions_sum))

plt.bar(x, plc_emissions_sum+ltd_emissions_sum+soe_emissions_sum, label="All Rigs", color="black", alpha=0.7)
plt.bar(x, plc_emissions_sum, label="PLC Rigs", color="skyblue")
plt.bar(x, ltd_emissions_sum, label="LTD Rigs", color="sienna", alpha=1)
plt.bar(x, soe_emissions_sum, label="SOE Rigs", color="olivedrab", alpha=1)
plt.xlim([-72, 84])
plt.xticks([-72,-60,-48,-36,-24,-12,0,12,24,36,48,60,72,84],
    [2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025])
plt.xlabel("Time (Months)")
plt.ylabel("NO2 Emissions (molecules/cm^2)")
plt.title("Total NO2 Emissions per month")
plt.legend()
plt.savefig("results/emissions_summed.png")
plt.clf()




divisor = 2000

for rig_emission, rig in zip(per_rig_emissions, filtered_rigs):
    if rig["parent_company_type"] == "PLC":
        plt.plot(rig_emission, c="skyblue", alpha=count["PLC"]/divisor)
    if rig["parent_company_type"] == "LTD":
        plt.plot(rig_emission, c="sienna", alpha=count["LTD"]/divisor)
    if rig["parent_company_type"] == "SOE":
        plt.plot(rig_emission, c="olivedrab", alpha=count["SOE"]/divisor)

# per_rig_emissions[per_rig_emissions >= np.nanmean(per_rig_emissions)]



plc_emissions_mean = np.nanmean((per_rig_emissions[notice == 0]),axis=0)
ltd_emissions_mean = np.nanmean((per_rig_emissions[notice == 1]),axis=0)
soe_emissions_mean = np.nanmean((per_rig_emissions[notice == 2]),axis=0)
all_emissions_mean = np.nanmean((per_rig_emissions),axis=0)


plc_emissions_std = np.nanstd((per_rig_emissions[notice == 0]),axis=0)
ltd_emissions_std = np.nanstd((per_rig_emissions[notice == 1]),axis=0)
soe_emissions_std = np.nanstd((per_rig_emissions[notice == 2]),axis=0)
all_emissions_std = np.nanstd((per_rig_emissions),axis=0)



# Rolling average
def rolling_average(data, window_size):
    result = np.full(len(data) - window_size + 1, np.nan)  # Pre-allocate result array
    for i in range(len(result)):
        window = data[i : i + window_size]  # Extract window
        valid_values = window[~np.isnan(window)]  # Ignore NaNs
        if valid_values.size > 0:
            result[i] = np.mean(valid_values)  # Compute mean only on non-NaN values
    return result
plc_emissions_mean_new = rolling_average(plc_emissions_mean,12)# - all_emissions_mean, 12)
ltd_emissions_mean_new = rolling_average(ltd_emissions_mean,12)# - all_emissions_mean, 12)
soe_emissions_mean_new = rolling_average(soe_emissions_mean,12)# - all_emissions_mean, 12)
all_emissions_mean_new = rolling_average(all_emissions_mean,12)# - all_emissions_mean, 12)
x = np.arange(6,len(plc_emissions_mean_new)+6)

# plc_emissions_std = rolling_average(plc_emissions_std, 12)
# ltd_emissions_std = rolling_average(ltd_emissions_std, 12)
# soe_emissions_std = rolling_average(soe_emissions_std, 12)
# all_emissions_std = rolling_average(all_emissions_std, 12)


# plt.fill_between(x, plc_emissions_mean_new-plc_emissions_std, plc_emissions_mean_new+plc_emissions_std, color="skyblue", alpha=0.2)
# plt.fill_between(x, ltd_emissions_mean_new-ltd_emissions_std, ltd_emissions_mean_new+ltd_emissions_std, color="sienna", alpha=0.2)
# plt.fill_between(x, soe_emissions_mean_new-soe_emissions_std, soe_emissions_mean_new+soe_emissions_std, color="olivedrab", alpha=0.2)
# plt.fill_between(x, all_emissions_mean_new-all_emissions_std, all_emissions_mean_new+all_emissions_std, color="black", alpha=0.2)

plt.plot(x,plc_emissions_mean_new, c="skyblue", label="PLC")
plt.plot(x,ltd_emissions_mean_new, c="sienna", label="LTD")
plt.plot(x,soe_emissions_mean_new, c="olivedrab", label="SOE")
plt.plot(x,all_emissions_mean_new, c="black", label="Average")


display_indices, display_years = [], []
for indx, year in zip(indices, years):
    if indx % 12 == 0:
        display_indices.append(indx)
        display_years.append(year+1970)
plt.xticks(display_indices, display_years)

plt.xlabel("Time (Months)")
plt.ylabel("NO2 Emissions more or less than the average (molecules/cm^2)")
plt.legend()
plt.title("Rolling average NO2 Emissions for each rig more or less than average (2018-2025)")
plt.savefig("results/emissions_averaged.png")
plt.clf()
#### Description: Where 0 is the lowest emission ever recorded in the area and 1 is the highest emission ever recorded in the area
#### Description: The area is defined as 0.2 degrees longitude around each rig, and the values are normalised for north sea and south sea.
#### Description: So if you are above you emit more than the average by 0 to 1 level and opposite is true for below 0 to -1








x = np.linspace(np.pi, (14+1) * np.pi, len(x))  # Angle values (0 to 2π)

# Create the polar plot
fig, ax = plt.subplots(figsize=(7,7), subplot_kw={'projection': 'polar'}, layout="constrained")




for rig_emission, rig in zip(per_rig_emissions, filtered_rigs):
    if rig["parent_company_type"] == "PLC":
        plt.plot(x, rolling_average(rig_emission,12), c="skyblue", alpha=count["PLC"]/divisor)
    if rig["parent_company_type"] == "LTD":
        plt.plot(x, rolling_average(rig_emission,12), c="sienna", alpha=count["LTD"]/divisor)
    if rig["parent_company_type"] == "SOE":
        plt.plot(x, rolling_average(rig_emission,12), c="olivedrab", alpha=count["SOE"]/divisor)

ax.plot(x, plc_emissions_mean_new, c="skyblue", label="PLC Ownership")
ax.plot(x, ltd_emissions_mean_new, c="sienna", label="LTD Ownership")
ax.plot(x, soe_emissions_mean_new, c="olivedrab", label="SOE Ownership")
ax.plot(x, all_emissions_mean_new, c="black", label="All Rigs")

# x = np.arange(len(plc_emissions_mean))
# ax.plot(x, plc_emissions_mean, c="skyblue", label="PLC Ownership")
# ax.plot(x, ltd_emissions_mean, c="sienna", label="LTD Ownership")
# ax.plot(x, soe_emissions_mean, c="olivedrab", label="SOE Ownership")
# ax.plot(x, all_emissions_mean, c="black", label="All Rigs")

# ax.set_ylim([0, 0.00006])


ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_xticks(np.linspace(0,2*np.pi,13),
    ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec",""])

ax.set_yticks([])
ax.set_yticklabels([])
ax.grid(False)

ax.plot([0, 0], [0, 0.00006], color='black', linewidth=1.5, linestyle='--', label="y-values x10^-5")  # Vertical line at θ = 0

# Manually place y-axis labels along the vertical line
xloc = [0.1, 0.1/2, 0.1/3, 0.1/4, 0.1/5, 0.1/6]
yticks = [0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.000058]
# yticks = np.linspace(0.00001, 0.00005, 4)  # Define tick positions
for i, (x, tick) in enumerate(zip(xloc, yticks)):
    ax.text(0.2/(i+1), tick, str(int(round(tick*100000))), ha='center', va='bottom', fontsize=15, color='black')

# plt.xlabel("Time (Months)")
# plt.ylabel("Average NO2 Emissions (molecules/cm^2)")
plt.legend(
    loc="upper left",  # Position of the legend
    bbox_to_anchor=(0.8, 1.05)  # Adjust to place the legend outside the pie
)
plt.title("Rolling average NO2 Emissions for each rig averaged by ownership per month")
plt.savefig("results/NO2_emissions_averaged_with_assumptions.png", dpi=300)
plt.clf()







##################################################



months_past = -12

plc = (np.nanmean((per_rig_emissions[notice == 0,months_past:]), axis=1))
ltd = (np.nanmean(per_rig_emissions[notice == 1,months_past:], axis=1))
soe = (np.nanmean(per_rig_emissions[notice == 2,months_past:], axis=1))

# plc = np.nanmean(per_rig_emissions[notice == 0,months_past:], axis=1)
# ltd = np.nanmean(per_rig_emissions[notice == 1,months_past:], axis=1)
# soe = np.nanmean(per_rig_emissions[notice == 2,months_past:], axis=1)

plc = plc[~np.isnan(plc)]
ltd = ltd[~np.isnan(ltd)]
soe = soe[~np.isnan(soe)]

data = []
len_data = []
categories = []
colors = []
for x, name, color in zip([plc, ltd, soe], ["PLC", "LTD", "SOE"], ["skyblue", "sienna", "olivedrab"]):
    if len(x) > 0:
        data.append(x)
        len_data.append(len(x))
        categories.append(name)
        colors.append(color)

len_data = np.array(len_data)
len_data = 0.8*len_data/np.amax(len_data)

plt.figure(figsize=(7, 7), layout="constrained")
parts = plt.violinplot(data, widths=0.8*np.array(len_data)/np.amax(len_data), showmeans=True, showmedians=False, showextrema=False)

for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])  # Violin fill color
    pc.set_edgecolor("black")      # Violin edge color
    pc.set_alpha(1)              # Transparency
for key in ['cmeans']:
    parts[key].set(color="black", linewidth=1.2)

plt.xticks(ticks=range(1, len(categories) + 1), labels=categories, fontsize=12)
plt.title("Distribution of Oil Rigs with emissions for each Ownership type (2024 Only)")
plt.xlabel("Ownership Type")
plt.ylabel("NO2 Emissions (molecules/cm^2)")
plt.savefig("results/NO2_emissions_violin.png", dpi=300)
#### Description: Where 0 is the lowest emission ever recorded in the area and 1 is the highest emission ever recorded in the area
#### Description: The area is defined as 0.2 degrees longitude around each rig, and the values are normalised for north sea and south sea.

##################################################












############### PER RIG GRAPHICS ##################

filtered_rigs = og_rigs.copy()

num_rigs = len(filtered_rigs)
cols = 10
rows = int(np.ceil(num_rigs / cols))

data_csv_out_per_rig_emissions_raw = []
data_csv_out_per_rig_emissions_normalized = []
data_csv_out_per_rig_firms_counts = []
data_csv_out_per_rig_firms_intensity = []
data_csv_out_oil_price = []


dater = []
for i in range(84):
    date_point_year = str(int(np.floor(i/12)+2018))
    date_point_month = str((i%12)+1).zfill(2)
    dater.append(date_point_year+"-"+date_point_month)


temp = [
    "asset_name",
    "asset_owner",
    "parent_company",
    "ownership_type",
    "longitude",
    "latitude",
    *dater
]
data_csv_out_per_rig_emissions_raw.append(temp)
data_csv_out_per_rig_emissions_normalized.append(temp)
data_csv_out_per_rig_firms_counts.append(temp)
data_csv_out_per_rig_firms_intensity.append(temp)
data_csv_out_oil_price.append(temp)


plc_group_data_norm = []
plc_group_data_raw = []
ltd_group_data_norm = []
ltd_group_data_raw = []
soe_group_data_norm = []
soe_group_data_raw = []

plc_viirs_data_counts = []
ltd_viirs_data_counts = []
soe_viirs_data_counts = []

for i, rig in enumerate(filtered_rigs):
    rig_name = rig["name"]
    for bad in ["(", ")", " ", "-", "/"]:
        rig_name = rig_name.replace(bad, "_")
    save_location = "results/rigs/"+rig_name
    if not os.path.exists(save_location):
        os.mkdir(save_location)

    # fig, ax = plt.subplots(figsize=(5, 5), 
    #                      subplot_kw={'projection': ccrs.PlateCarree()},
    #                      layout="constrained")
    lats = [hot_spot["latitude"] for hot_spot in rig["FIRMS_observations"]]
    lons = [hot_spot["longitude"] for hot_spot in rig["FIRMS_observations"]]
    date = [hot_spot["acq_date"] for hot_spot in rig["FIRMS_observations"]]
    intensity = np.array([hot_spot["brightness"] for hot_spot in rig["FIRMS_observations"]])

    save_json(save_location+"/rig_data.json", rig)

    # ax.scatter(lons, lats, c="red", alpha=0.2)
    # x,y = rig["coordinates"]
    # ax.scatter(x, y, marker="x", c="blue")
    # extra_degrees = 0.1
    # aspect_temp = 1 / np.cos(np.deg2rad(y))
    # ax.set_extent([x-extra_degrees,
    #                 x+extra_degrees,
    #                 y-(extra_degrees/aspect_temp),
    #                 y+(extra_degrees/aspect_temp)
    #             ], crs=ccrs.PlateCarree())
    # ax.set_aspect(1 / np.cos(np.deg2rad(y)))
    # # ax.gridlines(draw_labels=True, linestyle='--', alpha=1)
    # ax.set_title(rig["name"])
    # plt.savefig(save_location+"/FIRMS_observations.png")
    # plt.clf()

    if len(date) > 0:
        date_strings_2 = np.array(date.copy(), dtype="datetime64[D]")
        months = date_strings_2.astype('datetime64[M]').astype(int)
        years = date_strings_2.astype('datetime64[Y]').astype(int)
        indices = months - months[0]

        counts = []
        intensities = []
        for j in range(144):
            counts.append(np.sum([indices == j]))
            if np.sum([indices == j]) > 0:
                intensities.append(np.nanmean(intensity[indices == j]))
            else:
                intensities.append(np.nan)



        # plt.figure(figsize=(10,10), layout="constrained")
        # plt.bar(np.arange(144), counts)
        # plt.title("Number of FIRMS observations per month")
        # plt.xlabel("Time (months)")
        # plt.ylabel("Number of FIRMS observation")
        # plt.savefig(save_location+"/FIRMS_times.png")
        # plt.clf()

    # plt.figure(figsize=(10,10), layout="constrained")
    # plt.plot(per_rig_emissions[i], label="Data accumulated per month")
    # rolled_average = rolling_average(per_rig_emissions[i], 12)
    # plt.plot(np.arange(6,len(rolled_average)+6), rolled_average, label="Rolling Average per Year")
    # plt.title("NO2 emissions per month")
    # plt.xlabel("Time (months)")
    # plt.ylabel("NO2 emissions")
    # plt.legend()
    # plt.xticks([0,12,24,36,48,60,72,84],[2018,2019,2020,2021,2022,2023,2024,2025])
    # plt.savefig(save_location+"/emissions.png")
    # plt.clf()

    # plt.figure(figsize=(10,10), layout="constrained")
    # plt.plot(per_rig_emissions[i]-np.nanmean(per_rig_emissions,axis=0), label="Data accumulated per month")
    # rolled_average = rolling_average(per_rig_emissions[i]-np.nanmean(per_rig_emissions,axis=0), 12)
    # plt.plot(np.arange(6,len(rolled_average)+6), rolled_average, label="Rolling Average per Year")
    # plt.title("NO2 emissions per month")
    # plt.xlabel("Time (months)")
    # plt.ylabel("NO2 emissions (Mean Removed)")
    # plt.legend()
    # plt.xticks([0,12,24,36,48,60,72,84],[2018,2019,2020,2021,2022,2023,2024,2025])
    # plt.savefig(save_location+"/emissions_mean_removed.png")
    # plt.clf()

    
    if rig["parent_company_type"] == "PLC":
        plc_group_data_norm.append(per_rig_emissions[i])
        plc_group_data_raw.append(per_rig_emissions_raw[i])
    if rig["parent_company_type"] == "LTD":
        ltd_group_data_norm.append(per_rig_emissions[i])
        ltd_group_data_raw.append(per_rig_emissions_raw[i])
    if rig["parent_company_type"] == "SOE":
        soe_group_data_norm.append(per_rig_emissions[i])
        soe_group_data_raw.append(per_rig_emissions_raw[i])

    if rig["parent_company_type"] == "PLC":
        plc_viirs_data_counts.append(counts)
        # plc_viirs_data_brightness.append(data_csv_out_per_rig_firms_counts[i])
    if rig["parent_company_type"] == "LTD":
        ltd_viirs_data_counts.append(counts)
        # ltd_group_data_raw.append(per_rig_emissions_raw[i])
    if rig["parent_company_type"] == "SOE":
        soe_viirs_data_counts.append(counts)
        # soe_group_data_raw.append(per_rig_emissions_raw[i])


    temp = [
        rig["name"],
        rig["asset_owner"],
        rig["parent_company"],
        rig["parent_company_type"],
        rig["coordinates"][0],
        rig["coordinates"][1],
        *per_rig_emissions_raw[i]
    ]
    data_csv_out_per_rig_emissions_raw.append(temp)

    temp = [
        rig["name"],
        rig["asset_owner"],
        rig["parent_company"],
        rig["parent_company_type"],
        rig["coordinates"][0],
        rig["coordinates"][1],
        *per_rig_emissions[i]
    ]
    data_csv_out_per_rig_emissions_normalized.append(temp)
    
    print(np.shape(counts))
    temp = [
        rig["name"],
        rig["asset_owner"],
        rig["parent_company"],
        rig["parent_company_type"],
        rig["coordinates"][0],
        rig["coordinates"][1],
        *np.array(counts)[-84:]
    ]
    data_csv_out_per_rig_firms_counts.append(temp)

    temp = [
        rig["name"],
        rig["asset_owner"],
        rig["parent_company"],
        rig["parent_company_type"],
        rig["coordinates"][0],
        rig["coordinates"][1],
        *np.array(intensities)[-84:]
    ]
    data_csv_out_per_rig_firms_intensity.append(temp)


oil_price_dataset = csv_input("oil_price/Brent_oil_monthly_spot.csv")[3:]
trigger = False
oil_prices = []
for row in oil_price_dataset:
    if row[0] == "15/01/2018":
        trigger = True
    if row[0] == "15/01/2025":
        trigger = False
    if trigger:
        oil_prices.append(row[1])
    

temp = ["OIL SPOT PRICE", "", "", "", "", "", *oil_prices]
data_csv_out_oil_price.append(temp)


df_emissions_raw = pd.DataFrame(data_csv_out_per_rig_emissions_raw[1:], columns=data_csv_out_per_rig_emissions_raw[0])
df_emissions_normalised = pd.DataFrame(data_csv_out_per_rig_emissions_normalized[1:], columns=data_csv_out_per_rig_emissions_normalized[0])
df_firms_counts = pd.DataFrame(data_csv_out_per_rig_firms_counts[1:], columns=data_csv_out_per_rig_firms_counts[0])
df_firms_average_brightness = pd.DataFrame(data_csv_out_per_rig_firms_intensity[1:], columns=data_csv_out_per_rig_firms_intensity[0])
df_oil_price = pd.DataFrame(data_csv_out_oil_price[1:], columns=data_csv_out_oil_price[0])

with pd.ExcelWriter('output.xlsx') as writer:  # doctest: +SKIP
    df_emissions_raw.to_excel(writer, sheet_name='SENTINEL_raw_NO2')
    df_emissions_normalised.to_excel(writer, sheet_name='SENTINEL_normalised_NO2')
    df_firms_counts.to_excel(writer, sheet_name='FIRMS_counts')
    df_firms_average_brightness.to_excel(writer, sheet_name='FIRMS_average_brightness')
    df_oil_price.to_excel(writer, sheet_name='OIL_price_monthly')







######################## STATISTICS TABLES NO2 ##########################

all_datasets = [
    np.array(plc_group_data_raw),
    np.array(ltd_group_data_raw),
    np.array(soe_group_data_raw),
    np.array(plc_group_data_norm),
    np.array(ltd_group_data_norm),
    np.array(soe_group_data_norm)
]

column1 = ["Raw", "Raw", "Raw", "Normalised", "Normalised", "Normalised"]
column2 = ["PLC", "LTD", "SOE", "PLC", "LTD", "SOE"]

statistics_table = []

row0 = [
    r"Pre/Post Normalisation",
    r"Ownership Type",
    r"Number of oil rigs in group",
    r"Missing datapoints per rig",
    r"Mean",
    r"Standard deviation",
    r"Median Absolute Deviation",
    r"Minimum",
    r"Maximum",
    r"Interquartile Range"
]
statistics_table.append(row0)

def round_sig(x, sig=4, multiplier = 100000):
    x = x * multiplier
    return str((np.round(x, sig - int(np.floor(np.log10(abs(x)))) - 1) if x != 0 else 0))

for row_data, c1, c2 in zip(all_datasets, column1, column2):
    flat_no_nan = row_data[~np.isnan(row_data)].flatten()
    row_pretty = [
        c1,
        c2,
        np.shape(row_data)[0],
        round_sig(np.sum(np.isnan(row_data[:,3:]))/(np.prod(np.shape(row_data[:,3:]))), multiplier=100),
        round_sig(np.nanmean(row_data)),
        round_sig(np.nanstd(row_data)),
        round_sig(median_abs_deviation(flat_no_nan)),
        round_sig(np.nanmin(row_data)),
        round_sig(np.nanmax(row_data)),
        round_sig(np.percentile(flat_no_nan, 75) - np.percentile(flat_no_nan, 25))
    ]
    statistics_table.append(row_pretty)


csv_output("table_for_paper_no2.csv", np.swapaxes(statistics_table,0,1))


######################## STATISTICS TABLES VIIRS ##########################

all_datasets = [
    np.array(plc_viirs_data_counts),
    np.array(ltd_viirs_data_counts),
    np.array(soe_viirs_data_counts)
]

column2 = ["PLC", "LTD", "SOE"]

statistics_table = []

row0 = [
    r"Ownership Type",
    r"Number of oil rigs in group",
    r"Percentage of each rig where no flaring detected",
    r"Mean",
    r"Standard deviation",
    r"Median Absolute Deviation",
    r"Minimum",
    r"Maximum",
    r"Interquartile Range"
]
statistics_table.append(row0)

for row_data, c2 in zip(all_datasets, column2):
    print("Shape", np.shape(row_data))
    flat_no_nan = row_data[~np.isnan(row_data)].flatten()
    row_pretty = [
        c2,
        np.shape(row_data)[0],
        round_sig((np.sum(flat_no_nan == 0)+np.sum(np.isnan(row_data)))/(np.prod(np.shape(row_data))),multiplier=100),
        round_sig(np.nanmean(row_data), multiplier=1),
        round_sig(np.nanstd(row_data), multiplier=1),
        round_sig(median_abs_deviation(flat_no_nan), multiplier=1),
        round_sig(np.nanmin(row_data), multiplier=1),
        round_sig(np.nanmax(row_data), multiplier=1),
        round_sig(np.percentile(flat_no_nan, 75) - np.percentile(flat_no_nan, 25), multiplier=1),
    ]
    statistics_table.append(row_pretty)


csv_output("table_for_paper_viirs.csv", np.swapaxes(statistics_table,0,1))



#####################################################################


filtered_rigs = og_rigs.copy()

num_rigs = len(filtered_rigs)
cols = 10
rows = int(np.ceil(num_rigs / cols))

fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2),
                         layout="constrained")

display_order = []
for dat in data_partitioned_shaped:
    display_order.append(np.nansum(dat))

display_order = np.array(display_order)
indx = np.flip(np.argsort(display_order[~np.isnan(display_order)]))

filtered_rigs = [filtered_rigs[i] for i in indx]
data_partitioned_shaped = [data_partitioned_shaped[i] for i in indx]



for i, (ax, rig, dat) in enumerate(zip(axes.flatten(), filtered_rigs, data_partitioned_shaped)):

    # print(np.shape(data_small))
    if np.shape(dat)[0] > 0 and np.shape(dat)[1] > 0:
        x,y = rig["coordinates"]
        aspect_temp = 1 / np.cos(np.deg2rad(y))
        ax.imshow(dat, cmap='Reds')
        ax.set_title(str(i) + " - " + rig["parent_company_type"])
    
    ax.axis("off")

plt.savefig("results/SENTINEL_all.png", dpi=300)
plt.clf()




data_partitoned_shaped_size = [np.sum(~np.isnan(dat)) for dat in data_partitioned_shaped]

print(data_partitoned_shaped_size)

plc_area_count = []
ltd_area_count = []
soe_area_count = []
for rig, dat in zip(filtered_rigs, data_partitoned_shaped_size):
    if rig["parent_company_type"] == "PLC":
        plc_area_count.append(dat/np.nanmax(data_partitoned_shaped_size))
    if rig["parent_company_type"] == "LTD":
        ltd_area_count.append(dat/np.nanmax(data_partitoned_shaped_size))
    if rig["parent_company_type"] == "SOE":
        soe_area_count.append(dat/np.nanmax(data_partitoned_shaped_size))



plt.figure(figsize=(7,7))
x = np.linspace(0,1, 10)
plt.hist(plc_area_count, bins=x, label="PLC", color="skyblue", alpha=0.5)
plt.hist(ltd_area_count, bins=x, label="LTD", color="sienna", alpha=0.5)
plt.hist(soe_area_count, bins=x, label="SOE", color="olivedrab", alpha=0.5)
plt.legend()
plt.xlabel("Portion of whole 0.1 degree area")
plt.ylabel("Number of oil rigs")
plt.title("Portion of segmented areas of the whole 0.1 degree area")
plt.savefig("results/area_count_histogram.png", dpi=300)
plt.clf()









cols = 9
rows = 1

fig, axes = plt.subplots(rows, cols, figsize=(10, 3), layout="constrained")

chosen_i_list = [46,125,192,  20,114,184,  52,109,198]

filtered_rigs = [filtered_rigs[i] for i in chosen_i_list]
data_partitioned_shaped = [data_partitioned_shaped[i] for i in chosen_i_list]

common_shape = (100, 100)


max_width = max(dat.shape[1] for dat in data_partitioned_shaped)
max_height = max(dat.shape[0] for dat in data_partitioned_shaped)


for i, (ax, rig, dat) in enumerate(zip(axes.flatten(), filtered_rigs, data_partitioned_shaped)):
    if np.shape(dat)[0] > 0 and np.shape(dat)[1] > 0:
        x,y = rig["coordinates"]
        aspect_temp = 1 / np.cos(np.deg2rad(y))
        resized_dat = resize(dat, common_shape, mode='reflect', anti_aliasing=True)
        image_width = dat.shape[1]
        image_height = dat.shape[0]
        center_x = (max_width - image_width) / 2
        center_y = (max_height - image_height) / 2
        cbar = ax.imshow(dat, cmap='Reds', extent=[center_x, center_x + image_width, center_y, center_y + image_height])
        ax.set_xlim(0, max_width)
        ax.set_ylim(0, max_height)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(rig["parent_company_type"])
    # ax.axis("off")

cbar_ax = fig.add_axes([0.2, 0.1, 0.6, 0.02])  # Position at bottom ([left, bottom, width, height])
cbar = plt.colorbar(cbar, cax=cbar_ax, orientation='horizontal')
tick_positions = np.linspace(cbar.vmin, cbar.vmax, num=2)  # Adjust number of ticks as needed
cbar.set_ticks(tick_positions)
cbar.set_ticklabels(["Local Minimum", "Local Maximum"])
plt.suptitle("Examples of areas selected around oil rigs averaged over all days in dataset")
plt.savefig("results/SENTINEL_selected.png", dpi=300)
plt.clf()