import json
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import csv
import pandas as pd
from commons import *

################# INPUT PARAMETERS #################

hex_size = 500

####################################################






# NOAA-20(JPSS-1)
all_data_a = load_json("FIRMS/DL_FIRE_J1V-C2_587751/fire_archive_J1V-C2_587751.json")
data_a = [dat for dat in all_data_a if dat["type"] == "3"]
# NOAA-21(JPSS-2)
# all_data_b = load_json("DL_FIRE_J2V-C2_587752/fire_nrt_J2V-C2_587752.json")
# data_b = [dat for dat in all_data_b if dat["type"] == "3"]
# S-NPP
all_data_c = load_json("FIRMS/DL_FIRE_SV-C2_587753/fire_archive_SV-C2_587753.json")
data_c = [dat for dat in all_data_c if dat["type"] == "3"]

all_data_rigs = load_json("NSTA/Surface_Points_(WGS84).geojson")
data_rigs = [dat for dat in all_data_rigs["features"] if dat["properties"]["INF_TYPE"] == "PLATFORM" or dat["properties"]["INF_TYPE"] == "FPSO"]

datasets = [data_a, data_c, data_rigs]
dataset_names = [
    "Satellite dataset NOAA-20 (JPSS-1)",
    "Satellite dataset S-NPP", 
    "Oil Platforms"
]


fig, axes = plt.subplots(1, len(datasets), figsize=(6*len(datasets), 6), subplot_kw={'projection': ccrs.PlateCarree()})


fig = plt.figure(figsize=(12, 18))
ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=1, projection=ccrs.PlateCarree())
ax2 = plt.subplot2grid((3, 2), (0, 1), colspan=1, projection=ccrs.PlateCarree())
ax3 = plt.subplot2grid((3, 2), (1, 0), colspan=1, projection=ccrs.PlateCarree())
ax4 = plt.subplot2grid((3, 2), (1, 1), colspan=1, projection=ccrs.PlateCarree())
ax5 = plt.subplot2grid((3, 2), (2, 0), colspan=2, projection=ccrs.PlateCarree())
axes = [ax1, ax2, ax3, ax4, ax5]


mins_maxs = [99999, 99999, -99999, -99999]

for data in datasets:
    try:
        lats = [entry["latitude"] for entry in data]
        lons = [entry["longitude"] for entry in data]
    except:
        lats = [entry["geometry"]["coordinates"][1] for entry in data]
        lons = [entry["geometry"]["coordinates"][0] for entry in data]

    # Define map extent
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)

    if min_lat < mins_maxs[0]:
        mins_maxs[0] = min_lat
    if min_lon < mins_maxs[1]:
        mins_maxs[1] = min_lon
    if max_lat > mins_maxs[2]:
        mins_maxs[2] = max_lat
    if max_lon > mins_maxs[3]:
        mins_maxs[3] = max_lon

vmin = 1
vmax = 10000



def plot_data(ax, lats, lons, data_name, bins, cmaper, extent, vmin, vmax, hexbin_plots=[], C_val=None):
    # Add features
    ax.add_feature(cfeature.LAND, edgecolor='black')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0)
    gl.xlabel_style = {'size': 5}
    gl.ylabel_style = {'size': 5}
    hb = ax.hexbin(lons, lats, C=C_val, 
                   gridsize=hex_size, cmap=cmaper, 
                   transform=ccrs.PlateCarree(), 
                   alpha=1, bins=bins, mincnt=1,
                   linewidths=0.1,
                   extent=extent, vmin=vmin, vmax=vmax
                   )    
    hexbin_plots.append(hb)
    ax.set_title(data_name)
    ax.set_extent([mins_maxs[1], mins_maxs[3], mins_maxs[0], mins_maxs[2]], crs=ccrs.PlateCarree())
    ax.set_aspect(1 / np.cos(np.deg2rad(((mins_maxs[0] + mins_maxs[2])/2))))
    return hexbin_plots




hexbin_plots = []






# data = datasets[0]
# data_name = dataset_names[0]
# lats = [entry["latitude"] for entry in data]
# lons = [entry["longitude"] for entry in data]
# bins = "log"
# cmaper = "autumn_r"
# extent = [mins_maxs[1], mins_maxs[3], mins_maxs[0], mins_maxs[2]]
# hexbin_plots = plot_data(axes[0], lats, lons, data_name, bins, cmaper, extent, vmin, vmax, hexbin_plots)





# data = datasets[1]
# data_name = dataset_names[1]
# lats = [entry["latitude"] for entry in data]
# lons = [entry["longitude"] for entry in data]
# bins = "log"
# cmaper = "autumn_r"
# extent = [mins_maxs[1], mins_maxs[3], mins_maxs[0], mins_maxs[2]]
# hexbin_plots = plot_data(axes[1], lats, lons, data_name, bins, cmaper, extent, vmin, vmax, hexbin_plots)





# # Combine both satellite dataset
# hexbin_map = {}
# for item in [hexbin_plots[0], hexbin_plots[1]]: #positions, values in zip(hexbin_plots, hexbin_plots):
#     for pos, val in zip(item.get_offsets(), item.get_array()):
#         key = tuple(pos)
#         hexbin_map[key] = hexbin_map.get(key, 0) + val
# # Convert combined data back to lists
# combined_positions = np.array(list(hexbin_map.keys()))
# combined_values = np.array(list(hexbin_map.values()))

# lats = combined_positions[:,1]
# lons = combined_positions[:,0]
# data_name = "Combined satellite data observations"
# bins = "log"
# cmaper = "autumn_r"
# extent = [mins_maxs[1], mins_maxs[3], mins_maxs[0], mins_maxs[2]]
# hexbin_plots = plot_data(axes[2], lats, lons, data_name, bins, cmaper, extent, vmin, vmax, hexbin_plots, C_val=combined_values)







# data = datasets[2]
# data_name = "Oil Platforms and FPSOs"
# lats = [entry["geometry"]["coordinates"][1] for entry in data]
# lons = [entry["geometry"]["coordinates"][0] for entry in data]
# bins = 1
# cmaper = "winter"
# extent = [mins_maxs[1], mins_maxs[3], mins_maxs[0], mins_maxs[2]]
# hexbin_plots = plot_data(axes[3], lats, lons, data_name, bins, cmaper, extent, vmin, vmax, hexbin_plots)






# # Combine both satellite dataset
# hexbin_map = {}
# item = hexbin_plots[3]
# for pos, val in zip(item.get_offsets(), item.get_array()):
#     key = tuple(pos)
#     hexbin_map[key] = -1
# item = hexbin_plots[2]
# for pos, val in zip(item.get_offsets(), item.get_array()):
#     key = tuple(pos)
#     if hexbin_map.get(key, 0) == -1:
#         hexbin_map[key] = val

# # Convert combined data back to lists
# combined_positions = np.array(list(hexbin_map.keys()))
# combined_values = np.array(list(hexbin_map.values()))

# combined_positions = combined_positions[combined_values >= 0]
# combined_values = combined_values[combined_values >= 0]

# lats = combined_positions[:,1]
# lons = combined_positions[:,0]
# data_name = "Flaring Observations at Oil Platforms and FPSOs"
# bins = "log"
# cmaper = "autumn_r"
# extent = [mins_maxs[1], mins_maxs[3], mins_maxs[0], mins_maxs[2]]
# hexbin_plots = plot_data(axes[4], lats, lons, data_name, bins, cmaper, extent, vmin, vmax, hexbin_plots, C_val=combined_values)








# cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.03])  # Position the colorbar underneath
# cb = plt.colorbar(hexbin_plots[2], cax=cbar_ax, orientation='horizontal', label='Log of Number of observations')

# plt.tight_layout(rect=[0, 0.1, 1, 1])


# plt.savefig("results/PLATFORMS_with_FIRMS_observations.png", dpi=500)








# hexbin = hexbin_plots[4]
# data = data_rigs
# lats = [entry["geometry"]["coordinates"][1] for entry in data]
# lons = [entry["geometry"]["coordinates"][0] for entry in data]

# point_map = {}
# for lon, lat, dat in zip(lons, lats, data):
#         distances = np.linalg.norm(hexbin.get_offsets() - np.array([lon, lat]), axis=1)
#         closest_hex = tuple(hexbin.get_offsets()[np.argmin(distances)])
#         if closest_hex not in point_map:
#             point_map[closest_hex] = []
#         point_map[closest_hex].append(dat)



# output_data = []
# json_output_data = {}
# for hexbin, value in hexbin_map.items():
#     # The == 1 removes alot of datapoints maybe change
#     if value != -1 and len(point_map.get(hexbin, [])) == 1:
#         # print(f"Hexbin {hexbin} - Value: {value}, Points: {len(point_map.get(hexbin, []))}")
#         platform = point_map.get(hexbin, [])[0]["properties"] | point_map.get(hexbin, [])[0]["geometry"]
#         platform["Observations"] = value
#         if len(output_data) == 0:
#             output_data.append(list(platform.keys()))
#         output_data.append(list(platform.values()))
#         json_output_data.update({platform["NAME"]: value})


# df = pd.DataFrame(output_data[1:], columns=output_data[0])

# df.to_csv("results/PLATFORMS_with_FIRMS_observations.csv")

# big_rig_data = load_json("PROCESSED_DATA/surface_points_clean.json")

# for rig in big_rig_data:
#     if rig["name"] in json_output_data.keys():
#         rig.update({"FIRMS_observations": json_output_data[rig["name"]]})
#     else:
#         rig.update({"FIRMS_observations": 0})

# save_json("PROCESSED_DATA/surface_points_plus_FIRMS_count.json", big_rig_data)


# smaller_columns = ["NAME", "REP_GROUP", "coordinates","Observations", "COMMENTS", "UPD_REAS"]
# df[smaller_columns].sort_values(by="Observations", ascending=False).to_csv("results/PLATFORMS_with_FIRMS_observations_reduced.csv", index=False)









plt.clf()

aspect = 1 / np.cos(np.deg2rad(((mins_maxs[0] + mins_maxs[2])/2)))
fig, ax = plt.subplots(figsize=(7, 7), 
                        subplot_kw={'projection': ccrs.PlateCarree()})

data = datasets[1]
data_name = dataset_names[1]
lats = [entry["latitude"] for entry in data]
lons = [entry["longitude"] for entry in data]
bins = "log"
cmaper = "autumn_r"
extent = [mins_maxs[1], mins_maxs[3], mins_maxs[0], mins_maxs[2]]
# hexbin_plots = plot_data(plt.gca(), lats, lons, data_name, bins, cmaper, extent, vmin, vmax, hexbin_plots)

ax.coastlines()
# ax.gridlines(draw_labels=True, linestyle='--', alpha=0)
ax.add_feature(cfeature.LAND, facecolor='lightgray')

gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0)
gl.xlabel_style = {'size': 5}
gl.ylabel_style = {'size': 5}
hb = ax.hexbin(lons, lats, 
                gridsize=150, cmap=cmaper, 
                transform=ccrs.PlateCarree(), 
                alpha=1, bins=bins, mincnt=1,
                linewidths=0.1,
                extent=extent
                )    
ax.set_title(data_name)
ax.set_extent([mins_maxs[1], mins_maxs[3], mins_maxs[0], mins_maxs[2]], crs=ccrs.PlateCarree())
ax.set_aspect(1 / np.cos(np.deg2rad(((mins_maxs[0] + mins_maxs[2])/2))))


fig.subplots_adjust(right=0.9)
# cbar_ax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
cb = plt.colorbar(hb, orientation='vertical', label='Log of Number of observations')
# cb.ax.tick_params(labelsize=20)

plt.savefig("results/FIRMS_S-NPP_data.png", dpi=300)

print((extent[1]-extent[0])/150)