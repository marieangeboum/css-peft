#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  2 13:52:20 2024

@author: maboum
"""
import numpy as np

def downsample_image(image, scale):
    # Déterminer les dimensions de l'image downsamplée
    downscaled_size = image.shape[0] // scale

    # Initialiser une nouvelle matrice pour l'image downsamplée
    downscaled_image = np.zeros((downscaled_size, downscaled_size))
    
    # Parcourir chaque patch 8x8 dans l'image
    for i in range(0, image.shape[0], scale):
        for j in range(0, image.shape[1], scale):
            # Récupérer le patch 8x8
            patch = image[i:i+scale, j:j+scale]
            print(type(patch[0,0]))
            # Déterminer la valeur majoritaire dans le patch
            major_value = float(int(np.argmax(np.bincount(patch.flatten()))))
            print(major_value)
            # Assigner la valeur majoritaire à l'emplacement du patch dans l'image downsamplée
            i_downsampled = i // scale
            j_downsampled = j // scale
            downscaled_image[i_downsampled, j_downsampled] = major_value

    return downscaled_image



def upsample_predictions(predictions, scale):
    # Taille de l'image upsamplée
    upscaled_size = predictions.shape[0] * scale

    # Initialiser une nouvelle matrice pour l'image upsamplée
    upscaled_predictions = np.zeros((upscaled_size, upscaled_size))

    # Parcourir chaque pixel dans les prédictions
    for i in range(predictions.shape[0]):
        for j in range(predictions.shape[1]):
            # Récupérer la valeur de la prédiction pour ce pixel
            prediction_value = predictions[i, j]

            # Calculer les coordonnées dans l'image upsamplée
            i_upsampled = i * scale
            j_upsampled = j * scale

            # Remplir le patch upsamplé avec la valeur de prédiction
            upscaled_predictions[i_upsampled:i_upsampled+scale, j_upsampled:j_upsampled+scale] = prediction_value

    return upscaled_predictions