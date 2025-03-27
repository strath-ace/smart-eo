import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from commons import *
from collections import Counter

lon_min, lon_max = -2, 4
lat_min, lat_max = 52, 62

aspect = 1 / np.cos(np.deg2rad(((lat_max + lat_min)/2)))

def point_in_bounds(coords, lon_min,lon_max,lat_min,lat_max):
    condition_lon = rig["coordinates"][0] > lon_min and rig["coordinates"][0] < lon_max
    condition_lat = rig["coordinates"][1] > lat_min and rig["coordinates"][1] < lat_max
    return condition_lon and condition_lat

# rigs = load_json("PROCESSED_DATA/surface_points_plus_FIRMS_count.json")
rigs = load_json("PROCESSED_DATA/surface_points_clean.json")
filtered_rigs = []
for rig in rigs:
    # if rig["FIRMS_observations"] > 0:
    if point_in_bounds(rig["coordinates"], lon_min,lon_max,lat_min,lat_max):
    # if point_in_bounds(rig["coordinates"], -2,4,55.5,62):
        filtered_rigs.append(rig)

points = np.array([[rig["coordinates"][0], rig["coordinates"][1]] for rig in filtered_rigs])

# chemical = "NO2"
# chemical_dataset = np.load("SENTINEL/numpy_"+chemical+"/2018-01-16.npy")
# data = np.array(chemical_dataset)
# height = np.shape(data)[0]
# width = np.shape(data)[1]

height = int(100*(lat_max-lat_min))
width = int(100*(lon_max-lon_min))

grid = np.zeros((height, width))
grid[grid == 0] = np.nan

# Plot the Voronoi diagram
fig, ax = plt.subplots(figsize=(6, aspect*7), 
                        subplot_kw={'projection': ccrs.PlateCarree()},
                        layout="constrained")

def grid_to_real(i, j):
    real_x = lon_min + (i / (width - 1)) * (lon_max - lon_min)
    real_y = lat_min + (j / (height - 1)) * (lat_max - lat_min)
    return np.array([real_x, real_y])

def real_to_grid(x, y):
    i = int((x - lon_min) / (lon_max - lon_min) * (width - 1))
    j = int((y - lat_min) / (lat_max - lat_min) * (height - 1))
    return [i, j]

def closest_point(p, points):
    distances = np.linalg.norm(points - grid_to_real(*p), axis=1)
    closest = np.argmin(distances)
    if distances[closest] < 0.1:
        return closest
    else:
        return np.nan

mapper = {"PLC": 0, "LTD": 1, "SOE": 2}
mapper_color = {"PLC": "skyblue", "LTD": "sienna", "SOE": "olivedrab"}

# Compute the closest point for each pixel
for y in range(height):
    for x in range(width):
        value = closest_point([x, y], points)
        if np.isnan(value):
            grid[y, x] = np.nan
        else:
            grid[y,x] = mapper[filtered_rigs[value]["parent_company_type"]]

grid = np.flip(grid,axis=0)

print("Number of sectors",len(np.unique(grid)))

ax.coastlines()
ax.gridlines(draw_labels=True, linestyle='--', alpha=0)
ax.add_feature(cfeature.LAND, facecolor='lightgray')

grid = land_mask(grid, lon_min, lon_max, lat_min, lat_max)

# ax.imshow(grid, extent=[lon_min,lon_max,lat_min,lat_max], cmap="tab10", vmin=0, vmax=10)
# ax.scatter(points[:,0], points[:,1], marker="x", c="red")

points_plc = []
points_ltd = []
points_soe = []
for point, rig in zip(points, filtered_rigs):
    if rig["parent_company_type"] == "PLC":
        points_plc.append(point)
    elif rig["parent_company_type"] == "LTD":
        points_ltd.append(point)
    elif rig["parent_company_type"] == "SOE":
        points_soe.append(point)
points_plc = np.array(points_plc)
points_ltd = np.array(points_ltd)
points_soe = np.array(points_soe)

dot_size = 100
ax.scatter(points_plc[:,0], points_plc[:,1], s=dot_size, c=mapper_color["PLC"], label="PLC Operated Oil Rig")
ax.scatter(points_ltd[:,0], points_ltd[:,1], s=dot_size, c=mapper_color["LTD"], label="LTD Operated Oil Rig")
ax.scatter(points_soe[:,0], points_soe[:,1], s=dot_size, c=mapper_color["SOE"], label="SOE Operated Oil Rig")

extra_degrees = 0.5
ax.set_extent([lon_min-extra_degrees,
                lon_max+extra_degrees,
                lat_min-(extra_degrees/aspect),
                lat_max+(extra_degrees/aspect)
            ], crs=ccrs.PlateCarree())
ax.set_aspect(1 / np.cos(np.deg2rad(((lat_max + lat_min)/2))))

plt.legend(
    handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) 
             for color in ["skyblue", "sienna", "olivedrab"]],
    labels=["PLC Operated Oil Rig", "LTD Operated Oil Rig", "SOE Operated Oil Rig"],
    title="Ownership Types",
    # loc="center left",  # Position of the legend
    # bbox_to_anchor=(1, 0.97)  # Adjust to place the legend outside the pie
)

plt.title("Oil Platforms and FPSO in UK Waters subdivided by Parent Company Ownership Type", wrap=True)
plt.savefig("results/oil_rig_location_map.png", dpi=300)
plt.clf()


##################################################


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


divisions = [np.sum([notice==i]) for i in range(3)]
# print(divisions)
fig = plt.figure(figsize=(10,10))
plt.pie(divisions, labels=["PLC Owned", "LTD Owned", "SOE Owned"],
         colors=["skyblue", "sienna", "olivedrab"], 
         wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
plt.title("")
plt.savefig("results/pie_number_of_ownership_types.png", dpi=300)
plt.clf()

















company_names = []
company_color = {}
for rig in filtered_rigs:
    company_names.append(rig["parent_company"])
    company_color.update({rig["parent_company"]: mapper_color[rig["parent_company_type"]]})

print(company_color)

company_count = Counter(company_names)

company_names = list(company_count.keys())
company_count = list(company_count.values())
indx = np.flip(np.argsort(company_count))
company_names = np.array([company_names[i] for i in indx])
company_count = np.array([company_count[i] for i in indx])


print(company_names, company_count)

cuttoff = 1
summer = np.sum(company_count[company_count <= cuttoff])

company_names = company_names[company_count > cuttoff]
company_count = company_count[company_count > cuttoff]


company_names = company_names.tolist()
company_count = company_count.tolist()

company_color_li = []
for name in company_names:
    company_color_li.append(company_color[name])

print(company_color_li)

company_names_replace = []
for name in company_names:
    temp = name
    if "China" in name:
        temp = "Government of China"
    if "Norway" in name:
        temp = "Government of Norway"
    company_names_replace.append(temp)
company_names = company_names_replace



company_names.append("Other")
company_count.append(summer)
company_color_li.append("grey")

max_length_name = max([len(name) for name in company_names])

print(max_length_name)

fig = plt.figure(figsize=(8,7), layout="constrained")
wedges, texts = plt.pie(
    company_count, 
    labels=company_names,
    colors=company_color_li,
    wedgeprops={'edgecolor': 'black', 'linewidth': 1.5},
    startangle=-8,
    labeldistance=1.7,
)

for i, text in enumerate(texts):
    # Set the rotation angle for each label
    angle = (wedges[i].theta2 + wedges[i].theta1) / 2  # Mid-angle of each wedge
    # angle = wedges[i].theta2 
    # multiplier = int(0.5*max_length_name/len(text.get_text()))
    if len(text.get_text()) < 10:
        multiplier = 1.1 
    else:
        multiplier = 1
    if angle > 90 and angle < 270:
        angle = angle - 180
        
        text.set_text(text.get_text().rjust(int(max_length_name*multiplier)))
        text.set_horizontalalignment('center')
    else:
        text.set_text(text.get_text().ljust(int(max_length_name*multiplier)))
        text.set_horizontalalignment('center')
    text.set_rotation(angle)
    text.set_verticalalignment('center')


plt.title("Division of oil rigs by parent company of "+str(len(filtered_rigs))+" total", pad=20)

plt.legend(
    handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) 
             for color in ["skyblue", "sienna", "olivedrab", "grey"]],
    labels=["PLC", "LTD", "SOE", "Mixed"],
    title="Ownership Types",
    loc="upper left",  # Position of the legend
    bbox_to_anchor=(1, 0.85)  # Adjust to place the legend outside the pie
)

plt.savefig("results/pie_number_of_rigs_per_company.png", dpi=300)
plt.clf()

