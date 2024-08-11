from torchvision.utils import draw_segmentation_masks
import os
import torch
import rasterio
import torchvision.transforms.functional as F
from torchvision.ops import masks_to_boxes
import geopandas as gpd
from shapely.geometry import box

img_tif_path = "/run/user/108646/gvfs/sftp:host=flexo/scratcht/OpenEarthMap/OpenEarthMap/OpenEarthMap_wo_xBD/accra/images/accra_3.tif"
mask_tif_path = "/run/user/108646/gvfs/sftp:host=flexo/scratcht/OpenEarthMap/OpenEarthMap/OpenEarthMap_wo_xBD/accra/labels/accra_3.tif"
# Ouvrir le fichier d'image TIFF et lire les trois canaux en tant que tableau NumPy
with rasterio.open(img_tif_path) as src:
    img_np = src.read([1, 2, 3])
    img = torch.tensor(img_np, dtype=torch.uint8)

# Ouvrir le fichier de masque TIFF et lire le masque en tant que tableau NumPy
with rasterio.open(mask_tif_path) as src:
    mask_np = src.read(1)
    mask = torch.tensor(mask_np, dtype=torch.uint8)

# Convertir le masque en type float
mask = F.convert_image_dtype(mask, dtype=torch.uint8)

# Obtenir les ids uniques (objets)
obj_ids = torch.unique(mask)
print(obj_ids)

# Suppression du premier id (background)
obj_ids = obj_ids[1:]

# Créer des masques booléens pour chaque objet
masks = mask == obj_ids[:, None, None]
print(masks)
drawn_masks = []
for mask in masks:
    drawn_mask = draw_segmentation_masks(img, mask, alpha=0.8, colors="blue")
    drawn_masks.append(drawn_mask)
    
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
# Afficher les images avec les masques dessinés
for drawn_mask in drawn_masks:
    pil_img = F.to_pil_image(drawn_mask)
    plt.imshow(np.array(pil_img))
    plt.axis('off')
    plt.show()