import os
import sys
from torch.utils.data import Dataset
import torch
from dl_toolbox.utils import get_tiles
from dl_toolbox.torch_datasets.utils import *
from dl_toolbox.torch_datasets.commons import minmax
from torchvision.transforms.functional import to_pil_image, to_tensor
import rasterio
import imagesize
import numpy as np
from rasterio.windows import Window, bounds, from_bounds, shape
from rasterio.plot import show

from dl_toolbox.utils import MergeLabels, OneHot
import matplotlib.pyplot as plt


def downsample_image(image, scale):
    # Determine the dimensions of the downsampled image
    downscaled_size = image.shape[0] // scale
    
    # Initialize a new array for the downsampled image
    downscaled_image = np.zeros((downscaled_size, downscaled_size), dtype=np.float32)
    
    # Divide the image into 8x8 patches
    patches = image.reshape(downscaled_size, scale, downscaled_size, scale)
    
    # Calculate the mode value in each patch
    for i in range(downscaled_size):
        for j in range(downscaled_size):
            patch = patches[i, :, j, :].flatten()
            counts = np.bincount(patch.astype(int))
            major_value = np.argmax(counts)
            downscaled_image[i, j] = major_value
    
    return downscaled_image

class FlairDs(Dataset):
    def __init__(self,image_path,tile,fixed_crops, crop_size,crop_step, img_aug,
                 label_path=None,binary = False,label_binary = None, interpolation =False, *args,**kwargs):

        self.image_path = image_path # path to image
        self.label_path = label_path # pth to corresponding label
        self.tile = tile # initializing a tile to be extracted from image
        self.crop_windows = list(get_tiles(
            nols=tile.width,
            nrows=tile.height, 
            size=crop_size,
            step=crop_step,
            row_offset=tile.row_off,
            col_offset=tile.col_off)) if fixed_crops else None # returns a list of tile these are crop extracted from the initial img
        #print("crop windows", self.crop_windows)
        self.crop_size = crop_size # initializing crop size
        self.img_aug = get_transforms(img_aug)
        self.binary = binary
        self.label_binary = label_binary
        self.interpolation = interpolation
    def read_label(self, label_path, window):
        pass

    def read_image(self, image_path, window):
        pass

    def __len__(self):
        # returns nb of cropped windows
        return len(self.crop_windows) if self.crop_windows else 1

    def __getitem__(self, idx):
        ''' Given index idx this function loads a sample  from the dataset based on index.
            It identifies image'location on disk, converts it to a tensor
        '''
        if self.crop_windows:# if self.windows is initialized correctly begin iteration on said list
            window = self.crop_windows[idx]
        else: # otherwise use Winodw method from rasterio module to initilize a window of size cx and cy
            cx = self.tile.col_off + np.random.randint(0, self.tile.width - self.crop_size + 1) # why add those randint ?
            cy = self.tile.row_off + np.random.randint(0, self.tile.height - self.crop_size + 1)
            window = Window(cx, cy, self.crop_size, self.crop_size)
        ## Not here --> # vizualise the window crops extracted from the input image
        with rasterio.open(self.image_path, 'r') as image_file:
            image_rasterio = image_file.read(window=window, out_dtype=np.float32) # read the cropped part of the image
            img_path_strings = self.image_path.split('/')
            domain_pattern = img_path_strings[-4]

        # converts image crop into a tensor more precisely returns a contiguous tensor containing the same data as self tensor.
        image = torch.from_numpy(image_rasterio).float().contiguous()
        # label = None
        if self.label_path:
            # label = self.read_label(
            #     label_path=self.label_path,
            #     window=window)
            with rasterio.open(self.label_path, 'r') as label_file:
                label = label_file.read(window=window, out_dtype=np.float32)

            # converts label crop into contiguous tensor
            label = torch.from_numpy(label).float().contiguous()
            # label = torch.where((label >= 13), 13, label)
            # Remap labels to classes 1-9 (ensuring in-place operations)
            label = torch.where((label == 1) | (label == 18), 1, label)  # Map 1 and 18 to 1
            label = torch.where(label == 2, 2, label)  # Map 2 to 2
            label = torch.where(label == 3, 3, label)  # Map 3 to 3
            label = torch.where((label == 4) | (label == 14), 4, label)  # Map 4 and 14 to 4
            label = torch.where((label == 5) | (label == 13), 5, label)  # Map 5 and 13 to 5
            label = torch.where((label == 6) | (label == 7) | (label == 16) | (label == 17), 6,
                                label)  # Map 6, 7, 16, 17 to 6
            label = torch.where((label == 8) | (label == 15), 7, label)  # Map 8 and 15 to 7
            label = torch.where((label == 9) | (label == 11) | (label == 12), 8, label)  # Map 9, 11, 12 to 8
            label = torch.where(label == 10, 9, label)  # Map 10 to 9
            label = torch.where(label == 19, 10, label)
            # After remapping, subtract 1 to bring the labels into 0-8 range
            multi_labels = label.float() - 1

            # Ensure that no labels are out of bounds
            # multi_labels = torch.clamp(multi_labels, min=0, max=8)  # Make sure the range is 0-8
            # assert torch.min(multi_labels) >= 0 and torch.max(multi_labels) <= 9, \
            #     f"Label out of bounds: min={torch.min(multi_labels)}, max={torch.max(multi_labels)}"
        if self.img_aug is not None:            
            final_image, final_mask = self.img_aug(img=image, label=multi_labels)

        else:
            final_image, final_mask = image, multi_labels
        return {'orig_image':image,
                'orig_mask': label,
                'id' : domain_pattern,
                'window':window,
                'image':final_image,
                'mask':final_mask}



















