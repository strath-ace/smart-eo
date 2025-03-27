from commons import *
from collections import Counter

raw_data = load_json("NSTA/Surface_Points_(WGS84).geojson")
ownership_map = load_json("NSTA/who_owns_the_north_sea.json")

clean_data = []
for i in range(len(raw_data["features"])):
    temp = raw_data["features"][i]["properties"]
    if temp["STATUS"] == "ACTIVE" and temp["INF_TYPE"] in ["PLATFORM", "FPSO"]:
        temp.update({"coordinates": raw_data["features"][i]["geometry"]["coordinates"]})

        # Hardcode these values
        if temp["REP_GROUP"] == "REPSOL RESOURCES UK":
            temp["REP_GROUP"] = "REPSOL SINOPEC RESOURCES"
        elif temp["REP_GROUP"] == "CALENERGY GAS (HOLDINGS)":
            temp["REP_GROUP"] = "CALENERGY GAS HOLDINGS"
        elif temp["REP_GROUP"] == "INEOS FPS LTD":
            temp["REP_GROUP"] = "INEOS INDUSTRIES"
        elif temp["REP_GROUP"] == "WINTERSHALL NOORDZEE":
            temp["REP_GROUP"] = "WINTERSHALL B.V."
        
        # This one is 50/50 with harbour and dana and operated by ode
        elif temp["REP_GROUP"] == "KELLAS MIDSTREAM":
            temp["REP_GROUP"] = "HARBOUR ENERGY PLC"
        # This one is 50/50 with ping and hibiscus, was originally built by shell
        elif temp["REP_GROUP"] == "ANASURIA OPERATING COMPANY LIMITED":
            temp["REP_GROUP"] = "PING PETROLEUM LIMITED"

        ownership_temp = ownership_map[temp["REP_GROUP"]]
        ownership_temp.update({"asset_owner": temp["REP_GROUP"]})
        temp_organised = {
            "name": temp["NAME"],
            "asset_owner": temp["REP_GROUP"],
            "parent_company": ownership_temp["parent_company"],
            "parent_company_type": ownership_temp["parent_company_type"],
            "coordinates": temp["coordinates"],
            "ownership_details": ownership_temp,
            "NSTA_details": {k: temp[k] for k in temp.keys() - {"NAME", "coordinates"}}
        }

        clean_data.append(temp_organised)

save_json("PROCESSED_DATA/surface_points_clean.json", clean_data)










all_status = []
all_types = []
for i in range(len(raw_data["features"])):
    temp = raw_data["features"][i]["properties"]
    # all_status.append(temp["STATUS"])
    # all_types.append(temp["INF_TYPE"])
    all_types.append(temp["INF_TYPE"])

all_types = Counter(all_types)

all_types_name = list(all_types.keys())
all_types_counts = list(all_types.values())

indx = np.flip(np.argsort(all_types_counts))
all_types = {}
for i in indx:
    all_types.update({all_types_name[i]: all_types_counts[i]})


colors_map = {"ACTIVE": "#FFA07A", "NOT IN USE": "#FF8C00", "ABANDONED": "#D2691E", "REMOVED": "#D2691E"}

count_name = []
counter_all = []
counter_colors = []
for inf_type in all_types.keys():
    subtypes = []
    for i in range(len(raw_data["features"])):
        temp = raw_data["features"][i]["properties"]
        if temp["INF_TYPE"] == inf_type:
            subtypes.append(temp["STATUS"])
    subtypes = Counter(subtypes)
    for sub_name, sub_count in list(subtypes.items()):
        count_name.append(sub_name)
        counter_all.append(sub_count)
        counter_colors.append(colors_map[sub_name])


max_length_name = max([len(name) for name in all_types.keys()])


fig = plt.figure(figsize=(7,7), layout="constrained")
plt.pie(
    counter_all, 
    colors=counter_colors,
    startangle=0,
)


all_vals_names_with_other = np.array(list(all_types.keys()), dtype=str)
all_vals_with_other = np.array(list(all_types.values()))

summer = np.sum(all_vals_with_other[all_vals_with_other <= 2])

all_vals_names_with_other = all_vals_names_with_other[all_vals_with_other > 2].tolist()
all_vals_with_other = all_vals_with_other[all_vals_with_other > 2].tolist()

all_vals_names_with_other.append("Other")
all_vals_with_other.append(summer)


colors = [(1,1,1,0.1)]*len(all_vals_names_with_other)
wedges, texts = plt.pie(
    all_vals_with_other,
    labels=all_vals_names_with_other,
    colors=colors,
    wedgeprops={'edgecolor': 'black', 'linewidth': 1.5},
    startangle=0,
    labeldistance=1.3,
)

for i, text in enumerate(texts):
    # Set the rotation angle for each label
    angle = (wedges[i].theta2 + wedges[i].theta1) / 2  # Mid-angle of each wedge
    # angle = wedges[i].theta2 
    if angle > 90 and angle < 270:
        angle = angle - 180
        text.set_text(text.get_text().rjust(max_length_name))
        text.set_horizontalalignment('center')
    else:
        text.set_text(text.get_text().ljust(max_length_name))
        text.set_horizontalalignment('center')
    text.set_rotation(angle)
    text.set_verticalalignment('center')

plt.legend(
    handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) 
             for color in ["#FFA07A", "#FF8C00", "#D2691E"]],
    labels=["Active", "Not in Use", "Abandoned or Removed"],
    title="Status of Infrastructure",
    loc="upper left",  # Position of the legend
    bbox_to_anchor=(0.8, 0.95)  # Adjust to place the legend outside the pie
)

plt.title("Surface infrastructure in North Sea between recorded by the NSTA")
plt.savefig("results/pie_number_of_nsta_items.png", dpi=300)
plt.clf()







c_none = 0
c_stuff = 0
years = []

for x in clean_data:
    if x["NSTA_details"]["START_DATE"] == None:
        c_none += 1
    else:
        c_stuff += 1
        year = x["NSTA_details"]["START_DATE"][:4]  # Extract the year
        years.append(int(year))

print("########################")
print("Number of missing start dates:", c_none)
print("Number of included start dates:", c_stuff)
print("Total oil rigs", c_none+c_stuff)

plt.hist(years, bins=range(min(years), max(years) + 1))
plt.xlabel("Year")
plt.ylabel("Number of Oil Rigs")
plt.title("Oil Rig Creation Over Time")
plt.xticks(rotation=45)
plt.savefig("results/oil_rig_creation_histogram.png")
plt.clf()