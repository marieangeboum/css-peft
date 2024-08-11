#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 14:57:57 2024

@author: maboum
"""
import torch

batch_of_images = torch.randint(0, 13, size=(8, 1, 224, 224), dtype=torch.int64)
# images = target_np.astype(np.int64)
# scale = 8
# batch_size, height, width = images.shape
# print(images.shape)
# # Déterminer les dimensions de l'image downsamplée
# downscaled_size = height // scale
# print(downscaled_size)

# # Initialiser un tableau pour stocker les images downsamplées
# downscaled_images = np.zeros((batch_size, downscaled_size, downscaled_size), dtype=np.int64)

# # Parcourir chaque image dans le lot
# for b in range(batch_size):
    
#     # Parcourir chaque patch 8x8 dans l'image
#     for i in range(0, height, scale):
        
#         for j in range(0, width, scale):
            
#             # Récupérer le patch 8x8
#             patch = images[b, i:i+scale, j:j+scale]
#             print(patch)

#             # Déterminer la valeur majoritaire dans le patch
#             major_value = int(np.argmax(np.bincount(patch.flatten())))

#             # Assigner la valeur majoritaire à l'emplacement du patch dans l'image downsamplée
#             i_downsampled = i // scale
#             j_downsampled = j // scale
#             downscaled_images[b, i_downsampled, j_downsampled] = major_value
# print(downscaled_images)



# Créer un batch de tensors d'images de forme (2, 1, 224, 224) avec des entiers entre 0 et 12
# batch_of_images = torch.randint(0, 13, size=(2, 1, 224, 224), dtype=torch.int64)

# Appeler la fonction pour downsampler les images
# downsampled_images = downsample_images(batch_of_images, 8)

scale = 8
# images = batch_of_images = torch.randint(0, 13, size=(2, 1, 224, 224), dtype=torch.int64)
# images = target_np.astype(np.int64)
batch_size, channel, height, width = batch_of_images.shape

# Déterminer les dimensions de l'image downsamplée
downscaled_size = height // scale

# Initialiser un tableau pour stocker les images downsamplées
downscaled_images = torch.zeros((batch_size, 1, downscaled_size, downscaled_size), dtype=torch.int64)

# Diviser l'image en patches 8x8
patches = batch_of_images.unfold(2, scale, scale).unfold(3, scale, scale)

# # Calculer la valeur majoritaire dans chaque patch
# major_values, _ = patches.mode(dim=(4,5))

# # Assigner les valeurs majoritaires aux emplacements correspondants dans l'image downsamplée
# downscaled_images[:, :, :downscaled_size, :downscaled_size] = major_values

for b in range(batch_size):
       for i in range(downscaled_size):
           for j in range(downscaled_size):
               # Récupérer le patch 8x8
               patch = patches[b, :, i, j].flatten()
               
               # Compter les occurrences de chaque valeur dans le patch
               counts = torch.bincount(patch)
               
               # Trouver l'indice de la valeur avec le plus grand nombre d'occurrences
               major_value = torch.argmax(counts)
               
               # Assigner la valeur majoritaire à l'emplacement du patch dans l'image downsamplée
               downscaled_images[b, 0, i, j] = major_value
               
print(downscaled_images)


import torch

def downsample_images(images, scale):
    batch_size, channel, height, width = images.shape
    
    # Déterminer les dimensions de l'image downsamplée
    downscaled_size = height // scale
    
    # Initialiser un tableau pour stocker les images downsamplées
    downscaled_images = torch.zeros((batch_size, 1, downscaled_size, downscaled_size), dtype=torch.int64)
    
    # Diviser l'image en patches 8x8
    patches = images.unfold(2, scale, scale).unfold(3, scale, scale)
    
    # Calculer la valeur majoritaire dans chaque patch
    for b in range(batch_size):
        for i in range(downscaled_size):
            for j in range(downscaled_size):
                # Récupérer le patch 8x8
                patch = patches[b, :, i, j].flatten()
                
                # Compter les occurrences de chaque valeur dans le patch
                counts = torch.bincount(patch)
                
                # Trouver l'indice de la valeur avec le plus grand nombre d'occurrences
                major_value = torch.argmax(counts)
                
                # Assigner la valeur majoritaire à l'emplacement du patch dans l'image downsamplée
                downscaled_images[b, 0, i, j] = major_value
    
    return downscaled_images