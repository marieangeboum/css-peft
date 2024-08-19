from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import torch
import cv2
import os 
import matplotlib.cm as cm
from scipy.ndimage import label
import shapefile  # Bibliothèque pyshp pour manipuler des shapefiles

# Chemins vers les fichiers TIFF
img_tif_path = "/run/user/108646/gvfs/sftp:host=flexo/scratcht/OpenEarthMap/OpenEarthMap/OpenEarthMap_wo_xBD/accra/images/accra_3.tif"
mask_tif_path = "/run/user/108646/gvfs/sftp:host=flexo/scratcht/OpenEarthMap/OpenEarthMap/OpenEarthMap_wo_xBD/accra/labels/accra_3.tif"

# Extraire le nom de base du fichier de label pour nommer le shapefile
base_name = os.path.splitext(os.path.basename(mask_tif_path))[0]
shapefile_path = f'bbox_{base_name}.shp'

# Fonction pour obtenir les bounding boxes
def get_bounding_boxes(mask, class_label):
    labeled_mask, num_features = label(mask)
    
    bounding_boxes = []
    
    for label_num in range(1, num_features + 1):
        rows, cols = np.where(labeled_mask == label_num)
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()
        
        bounding_box = {
            'top_left': (min_row, min_col),
            'bottom_right': (max_row, max_col),
            'class': class_label
        }
        
        bounding_boxes.append(bounding_box)
    
    return bounding_boxes

# Lire le fichier d'image TIFF et les trois canaux en tant que tableau NumPy
with rasterio.open(img_tif_path) as src:
    img_np = src.read([1, 2, 3])
    img = torch.tensor(img_np, dtype=torch.uint8)

# Lire le fichier de masque TIFF et le masque en tant que tableau NumPy
with rasterio.open(mask_tif_path) as src:
    mask_np = src.read(1)
    mask = torch.tensor(mask_np, dtype=torch.uint8)

# Convertir le masque en tableau numpy
image_array = np.array(mask)

# Obtenir les classes uniques dans le masque
classes = np.unique(image_array)

# Créer un dictionnaire pour stocker les masques booléens pour chaque classe
masks = {}
for cls_ in classes:
    masks[cls_] = (image_array == cls_)

# Générer des couleurs dynamiques pour chaque classe
num_classes = len(classes)
colors = cm.get_cmap('hsv', num_classes)

# Créer un objet shapefile writer
w = shapefile.Writer(shapefile_path, shapefile.POLYGON)
w.field('class', 'N')

# Sauvegarder les masques booléens sous forme d'images TIFF
for i, (cls_, mask) in enumerate(masks.items()):
    mask_image = (mask.astype(np.uint8) * 255)  # Convertir le masque en image (0 et 255 pour booléens)
    
    # Convertir l'image en format compatible avec OpenCV (HxWxC) et de type uint8
    img_cv = np.moveaxis(img_np, 0, -1).astype(np.uint8)
    
    # Récupérer les bounding boxes
    bounding_boxes = get_bounding_boxes(mask, cls_)
    
    # Ajouter les bounding boxes au shapefile
    for bbox in bounding_boxes:
        top_left = bbox['top_left']
        bottom_right = bbox['bottom_right']
        class_label = bbox['class']
        
        # Définir les coordonnées du rectangle dans le shapefile
        polygon = [
            [top_left[1], top_left[0]],  # Haut gauche
            [top_left[1], bottom_right[0]],  # Bas gauche
            [bottom_right[1], bottom_right[0]],  # Bas droite
            [bottom_right[1], top_left[0]],  # Haut droite
            [top_left[1], top_left[0]]  # Fermer le polygone
        ]
        
        # Écrire le polygone et l'attribut de classe dans le shapefile
        w.poly([polygon])
        w.record(class_label)
    
    # Tracer les bounding boxes sur l'image d'origine avec Matplotlib
    plt.imshow(img_cv)
    for bbox in bounding_boxes:
        top_left = bbox['top_left']
        bottom_right = bbox['bottom_right']
        
        # Tracer le rectangle (bounding box)
        rect = plt.Rectangle((top_left[1], top_left[0]), 
                             bottom_right[1] - top_left[1], 
                             bottom_right[0] - top_left[0], 
                             linewidth=0.5, 
                             edgecolor=colors(i),  # Utiliser la couleur de la classe
                             facecolor='none')
        plt.gca().add_patch(rect)
    
    plt.title(f'Image avec Bounding Boxes pour la classe {cls_}')
    plt.axis('off')
    plt.show()

# Fermer et sauvegarder le shapefile
w.close()

# Créer le fichier .prj avec le même nom de base (optionnel)
prj_content = """GEOGCS["WGS 84",
    DATUM["WGS_1984",
        SPHEROID["WGS 84",6378137,298.257223563,
            AUTHORITY["EPSG","7030"]],
        AUTHORITY["EPSG","6326"]],
    PRIMEM["Greenwich",0,
        AUTHORITY["EPSG","8901"]],
    UNIT["degree",0.0174532925199433,
        AUTHORITY["EPSG","9122"]],
    AUTHORITY["EPSG","4326"]]"""
with open(f"bbox_{base_name}.prj", "w") as prj_file:
    prj_file.write(prj_content)

print(f"Shapefile {shapefile_path} et fichiers associés créés avec succès.")
import geopandas as gpd

# Chemin vers le fichier shapefile
shapefile_path = f'bbox_{base_name}.shp'

# Lire le fichier shapefile
try:
    gdf = gpd.read_file(shapefile_path)
    print("Fichier shapefile lu avec succès.")
    
    # Afficher les informations sur le GeoDataFrame
    print(gdf.info())
    print(gdf.head())
    
    # Afficher la carte
   
except Exception as e:
    print(f"Erreur lors de la lecture du fichier shapefile: {e}")
