import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import torch
import cv2
import matplotlib.cm as cm
from scipy.ndimage import label
import shapefile
import geopandas as gpd

class ShapefileCreator:
    def __init__(self, img_tif_path, mask_tif_path):
        self.img_tif_path = img_tif_path
        self.mask_tif_path = mask_tif_path
        self.base_name = os.path.splitext(os.path.basename(mask_tif_path))[0]
        self.shapefile_path = os.path.join(self.mask_tif_path.rsplit('/', 1)[0] + '/', f'bbox_{self.base_name}.shp')
        self.img_np = None
        self.mask_np = None

    def read_files(self):
        # Lire le fichier d'image TIFF et les trois canaux en tant que tableau NumPy
        with rasterio.open(self.img_tif_path) as src:
            self.img_np = src.read([1, 2, 3])
            self.img = torch.tensor(self.img_np, dtype=torch.uint8)

        # Lire le fichier de masque TIFF et le masque en tant que tableau NumPy
        with rasterio.open(self.mask_tif_path) as src:
            self.mask_np = src.read(1)
            self.mask = torch.tensor(self.mask_np, dtype=torch.uint8)

    def get_bounding_boxes(self, mask, class_label):
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

    def create_shapefile(self):
        image_array = np.array(self.mask)
        classes = np.unique(image_array)
        masks = {cls_: (image_array == cls_) for cls_ in classes}
        num_classes = len(classes)
        colors = cm.get_cmap('hsv', num_classes)

        # Créer un objet shapefile writer
        w = shapefile.Writer(self.shapefile_path, shapefile.POLYGON)
        w.field('class', 'N')

        for i, (cls_, mask) in enumerate(masks.items()):
            mask_image = (mask.astype(np.uint8) * 255)
            img_cv = np.moveaxis(self.img_np, 0, -1).astype(np.uint8)
            bounding_boxes = self.get_bounding_boxes(mask, cls_)

            for bbox in bounding_boxes:
                top_left = bbox['top_left']
                bottom_right = bbox['bottom_right']
                class_label = bbox['class']
                polygon = [
                    [top_left[1], top_left[0]],
                    [top_left[1], bottom_right[0]],
                    [bottom_right[1], bottom_right[0]],
                    [bottom_right[1], top_left[0]],
                    [top_left[1], top_left[0]]
                ]
                w.poly([polygon])
                w.record(class_label)

            # plt.imshow(img_cv)
            # for bbox in bounding_boxes:
            #     top_left = bbox['top_left']
            #     bottom_right = bbox['bottom_right']
            #     rect = plt.Rectangle(
            #         (top_left[1], top_left[0]),
            #         bottom_right[1] - top_left[1],
            #         bottom_right[0] - top_left[0],
            #         linewidth=0.5,
            #         edgecolor=colors(i),
            #         facecolor='none'
            #     )
            #     plt.gca().add_patch(rect)
            # plt.title(f'Image avec Bounding Boxes pour la classe {cls_}')
            # plt.axis('off')
            # plt.show()

        w.close()
        self.create_prj_file()
        print(f"Shapefile {self.shapefile_path} et fichiers associés créés avec succès.")

    def create_prj_file(self):
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
        with open(os.path.join(self.mask_tif_path.rsplit('/', 1)[0] + '/', f"bbox_{self.base_name}.prj"), "w") as prj_file:
            prj_file.write(prj_content)

    def verify_shapefile(self):
        try:
            gdf = gpd.read_file(self.shapefile_path)
            # print("Fichier shapefile lu avec succès.")
            # print(gdf.info())
            # print(gdf.head())
            
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier shapefile: {e}")

# # Utilisation de la classe
# data_path = "/run/user/108646/gvfs/sftp:host=flexo/scratcht/OpenEarthMap/OpenEarthMap/OpenEarthMap_wo_xBD/"
# locations_list = [s for s in os.listdir(data_path) if not (s.endswith('.csv') or s.endswith('.txt'))]
# print(locations_list)
# for lieu in locations_list :
#     print(lieu)
#     lbls_list = glob.glob(os.path.join(data_path,f"{lieu}/labels/{lieu}_*.tif"))
#     print(lbls_list)
#     if len(glob.glob(os.path.join(data_path,f"{lieu}/images/{lieu}_*.tif")))==0 :
#         continue
#     for lbl in lbls_list : 
#         mask_tif_path = lbl
#         img_tif_path = lbl.replace('/labels/', '/images/')
#         print(mask_tif_path, img_tif_path)
#         shapefile_creator = ShapefileCreator(img_tif_path, mask_tif_path)
#         shapefile_creator.read_files()
#         shapefile_creator.create_shapefile()
#         shapefile_creator.verify_shapefile()
