import os
import cv2
import zipfile
# import leafmap
import geopandas as gpd
import numpy as np
import rasterio
import glob
from rasterio.merge import merge
from rasterio.features import shapes
from shapely.geometry import shape
from samgeo import SamGeo, SamGeoPredictor, tms_to_geotiff
from segment_anything import sam_model_registry

def read_shapefile(filepath):
    """
    Reads a shapefile and extracts bounding boxes.

    Parameters:
        filepath (str): Path to the shapefile.

    Returns:
        List of bounding boxes.
    """
    gdf = gpd.read_file(filepath)
    condition = gdf['class'] != 0.0
    gdf = gdf[condition]
    return [list(geom.bounds) for geom in gdf.geometry]

def initialize_predictor(img_arr, checkpoint):
    """
    Initializes SamGeoPredictor.

    Parameters:
        img_arr (np.array): Image array.
        checkpoint (str): Path to the model checkpoint.

    Returns:
        An instance of SamGeoPredictor.
    """
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=checkpoint)
    predictor = SamGeoPredictor(sam)
    predictor.set_image(img_arr)

    return predictor

# def process_clip_boxes(image, clip_boxes, predictor):
#     """
#     Processes each clip box, creates geojson files, and adds them to the map.

#     Parameters:
#         image (str): Image file path.
#         clip_boxes (list): List of bounding boxes.
#         predictor (SamGeoPredictor): An instance of SamGeoPredictor.

#     Returns:
#         None
#     """
#     combined_gdf = gpd.GeoDataFrame()
#     src_files_to_mosaic = []

#     for i, clip_box in enumerate(clip_boxes):
#         print(i,clip_box)
#         masks, _, _ = predictor.predict(src_fp=image, geo_box=clip_box)
        
#         # Ensure masks are binary
#         masks = (masks > 0).astype('uint8')

#         # masks_img = os.path.join('/run/user/108646/gvfs/sftp:host=flexo/d/maboum/JSTARS/segmentation/checkpoints/test_samgeo', f"preds_{i}.tif")
#         masks_img = os.path.join('/d/maboum/test_samgeo', f"preds_{i}.tif")
#         predictor.masks_to_geotiff(image, masks_img, masks)

#         src = rasterio.open(masks_img)
#         src_files_to_mosaic.append(src)

#         # vector = os.path.join('/run/user/108646/gvfs/sftp:host=flexo/d/maboum/JSTARS/segmentation/checkpoints/test_samgeo', f"feats_{i}.geojson")
#         vector = os.path.join('/d/maboum/test_samgeo', f"feats_{i}.geojson")
#         temp_gdf = predictor.geotiff_to_geojson(masks_img, vector, bidx=1)
#         combined_gdf = combined_gdf.append(temp_gdf)
#         combined_gdf.set_geometry('geometry', inplace=True)

#     # Mosaic and write the mosaic raster to disk
#     mosaic, out_trans = merge(src_files_to_mosaic)
#     out_meta = src.meta.copy()

#     out_meta.update({"driver": "GTiff",
#                     "dtype": 'uint8',
#                     "height": mosaic.shape[1],
#                     "width": mosaic.shape[2],
#                     "count": 1,
#                     "transform": out_trans,
#                     "crs": src.crs})

#     with rasterio.open('mosaic_mask.tif', "w", **out_meta) as dest:
#         dest.write(mosaic[0], 1)

#     # Save polygons as separate features in a shapefile
#     combined_gdf['geometry'] = combined_gdf.geometry.buffer(0)
#     combined_gdf.to_file("separate_features.shp")
    

def process_clip_boxes(image, clip_boxes, predictor):
    """
    Processes each clip box, creates GeoJSON files, and combines them into a single GeoDataFrame.
    Also creates a mosaic raster from all masks and saves features as a shapefile.

    Parameters:
        image (str): Image file path.
        clip_boxes (list): List of bounding boxes.
        predictor (SamGeoPredictor): An instance of SamGeoPredictor.

    Returns:
        None
    """
    combined_gdf = gpd.GeoDataFrame()
    src_files_to_mosaic = []

    for i, clip_box in enumerate(clip_boxes):
        print(f"Processing clip box {i}: {clip_box}")
        
        # Predict masks for the given clip box
        masks, _, _ = predictor.predict(src_fp=image, geo_box=clip_box)
        
        # Ensure masks are binary
        masks = (masks > 0).astype('uint8')

        # Define file paths for masks and GeoJSON
        masks_img = os.path.join('/d/maboum/test_samgeo', f"preds_{i}.tif")
        vector = os.path.join('/d/maboum/test_samgeo', f"feats_{i}.geojson")

        # Save masks as GeoTIFF
        predictor.masks_to_geotiff(image, masks_img, masks)

        # Open the saved raster file
        src = rasterio.open(masks_img)
        src_files_to_mosaic.append(src)

        # Convert GeoTIFF masks to GeoJSON
        temp_gdf = predictor.geotiff_to_geojson(masks_img, vector, bidx=1)

        # Append temp_gdf to combined_gdf
        if not temp_gdf.empty:
            combined_gdf = combined_gdf.append(temp_gdf, ignore_index=True)
        else:
            print(f"Warning: GeoJSON for clip box {i} is empty. Skipping.")

    # Mosaic and write the mosaic raster to disk
    if src_files_to_mosaic:
        mosaic, out_trans = merge(src_files_to_mosaic)
        out_meta = src_files_to_mosaic[0].meta.copy()

        out_meta.update({
            "driver": "GTiff",
            "dtype": 'uint8',
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "count": 1,
            "transform": out_trans,
            "crs": src_files_to_mosaic[0].crs
        })

        with rasterio.open('mosaic_mask.tif', "w", **out_meta) as dest:
            dest.write(mosaic[0], 1)
    else:
        print("No valid rasters to mosaic. Skipping raster creation.")

    # Save polygons as separate features in a shapefile
    if not combined_gdf.empty:
        # Ensure geometry is valid
        combined_gdf['geometry'] = combined_gdf.geometry.buffer(0)
        combined_gdf.to_file("separate_features.shp")
    else:
        print("No valid GeoDataFrames to save. Skipping shapefile creation.")
# Switch to your image and shapefile instead
image = '/run/user/108646/gvfs/sftp:host=flexo/scratcht/OpenEarthMap/OpenEarthMap/OpenEarthMap_wo_xBD/accra/images/accra_3.tif'
shapefile = '/run/user/108646/gvfs/sftp:host=flexo/scratcht/OpenEarthMap/OpenEarthMap/OpenEarthMap_wo_xBD/accra/labels/bbox_accra_3.shp'

# SDefine SAM's model and path
# out_dir = os.path.join(os.path.expanduser("~"), "Downloads")
# os.makedirs(out_dir, exist_ok=True)
# out_dir = "/run/user/108646/gvfs/sftp:host=flexo/d/maboum/JSTARS/segmentation/checkpoints"
checkpoint = "/run/user/108646/gvfs/sftp:host=flexo/d/maboum/JSTARS/segmentation/checkpoints/sam_vit_h_4b8939.pth"

sam = SamGeo(
    model_type="vit_h",
    checkpoint=checkpoint,
    sam_kwargs=None,
)

# Read the image
img_arr = cv2.imread(image)

# Extract bounding boxes from the shapefile
clip_boxes = read_shapefile(shapefile)

# Initialize SamGeoPredictor
predictor = initialize_predictor(img_arr, checkpoint)

# Process each clip box and add vector to the map
process_clip_boxes(image, clip_boxes, predictor)

# Display the results
mosaic = 'mosaic_mask.tif'
features = 'separate_features.shp'
style={'color': '#a37aa9',}

# m = leafmap.Map(center=shapefile)
# m.add_raster(mosaic, layer_name="Mask Mosaic")
# m.add_vector(features, layer_name='Vector', opacity=0.5, style=style)




