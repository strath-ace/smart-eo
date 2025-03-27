# from sentinelhub import SentinelHubRequest, DataCollection, MimeType, bbox_to_dimensions, BBox, 
from sentinelhub import (
    SHConfig,
    CRS,
    BBox,
    DataCollection,
    DownloadRequest,
    MimeType,
    MosaickingOrder,
    SentinelHubDownloadClient,
    SentinelHubRequest,
    bbox_to_dimensions,
    SentinelHubCatalog,
)
from dotenv import load_dotenv
import os
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Get Sentinel Credentials
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(BASE_DIR, ".env"))
config = SHConfig()
config.sh_client_id = os.getenv("SENTINELHUB_client_id")
config.sh_client_secret = os.getenv("SENTINELHUB_client_secret")

# Define Area of Interest (AOI) as a bounding box (longitude/latitude)
bbox = BBox(bbox=[-2, 52, 4, 62], crs=CRS.WGS84)

catalog = SentinelHubCatalog(config=config)

# Take the eval script from sentinelhub website
evalscript = """
const band = "CH4";

function setup() {
  return {
    input: [band, "dataMask"],
    output: {
      bands: 2,
      sampleType: "FLOAT32",
    },
  };
}

function evaluatePixel(samples) {
  let ret = [samples[band]];
  ret.push(samples.dataMask);
  return ret;
}
"""



### Generate every day
start = datetime(2024, 1, 1)
end = datetime(2025, 1, 1)
edges = []
current = start
while current <= end:
    edges.append(current.date().isoformat())
    current += timedelta(days=1)
slots = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]



# print("Monthly time windows:\n")
# for slot in slots:
#     print(slot)






def get_results(time_interval):
    return SentinelHubRequest(
        data_folder="raw_data_CH4",
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL5P,
                time_interval=time_interval,
                # mosaicking_order=MosaickingOrder.LEAST_CC,
            )
        ],
        responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
        bbox=bbox,
        size=bbox_to_dimensions(bbox, resolution=1000),  # Set resolution (adjustable)
        config=config,
    )



list_of_requests = [get_results(slot) for slot in slots]
list_of_requests = [request.download_list[0] for request in list_of_requests]

# download data with multiple threads

for i in range(len(list_of_requests)):
  try:
    data = SentinelHubDownloadClient(config=config).download([list_of_requests[i]], max_threads=5)
    print(slots[i], "downloaded")
  except:
    print(slots[i], "not available")





















# for i, x in enumerate(data):
#     np.save("data/test_"+str(i), np.array(x))



# # some stuff for pretty plots
# ncols = 2
# nrows = 2
# aspect_ratio = bbox_to_dimensions(bbox, resolution=1000)[0] / bbox_to_dimensions(bbox, resolution=7000)[1]
# subplot_kw = {"xticks": [], "yticks": [], "frame_on": False}

# fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(5 * ncols * aspect_ratio, 5 * nrows), subplot_kw=subplot_kw)

# for idx, image in enumerate(data):
#     ax = axs[idx // ncols][idx % ncols]
#     ax.imshow(np.clip(image * 2.5 / 255, 0, 1))
#     ax.set_title(f"{slots[idx][0]}  -  {slots[idx][1]}", fontsize=10)

# plt.tight_layout()
# plt.savefig("test2.png")




# # Execute the request and save images
# response = request_true_color.get_data(save_data=True)
# print(response)
# for i, x in enumerate(response):
#     np.save("test_"+str(i), np.array(x))

# print("Images downloaded successfully.")
