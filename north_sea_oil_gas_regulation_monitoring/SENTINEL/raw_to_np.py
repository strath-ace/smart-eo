import os
import numpy as np
import rasterio
import json
from PIL import Image
from tqdm import tqdm

# Input
def load_json(file_name):
    with open(file_name) as f:
        output = json.load(f)
    return output


chemicals = ["HCHO", "NO2", "CH4", "SO2", "CO"]

for chem in chemicals:
    print("Proccesing", chem, "data")
    folder_path = "./raw_data_"+chem
    output_path = "./numpy_"+chem
    image_path = "./png_"+chem
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]


    for folder in tqdm(subfolders):

        date = load_json(folder_path+"/"+folder+"/request.json")["request"]["payload"]["input"]["data"][0]["dataFilter"]["timeRange"]["from"][:10]
        with rasterio.open(folder_path+"/"+folder+"/response.tiff") as src:
            array = src.read()

        value = array[0].copy()
        mask = array[1].copy()
        value[mask == 0] = np.nan
        np.save(output_path+"/"+date, value)

        img = array.copy()
        img = np.moveaxis(img, 0, -1)
        if (np.nanmax(img[:,:,0])-np.nanmin(img[:,:,0])) != 0:
            img[:,:,0] = 255*(img[:,:,0]-np.nanmin(img[:,:,0]))/(np.nanmax(img[:,:,0])-np.nanmin(img[:,:,0]))
            img[:,:,1] = img[:,:,1]*255

        image_pil = Image.fromarray(img.astype(np.uint8), mode="LA")
        image_pil.save(image_path+"/"+date+".png")