# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Tue Sep 15 15:06:01 2020

@author: 20164798
"""
"""
this script read all training and test data and masks. 
prepare training data and masks and save them in a binary format.
prepare test data and test ids and save them in a binary format.

"""



import os
import numpy as np

from skimage.io import imsave, imread
from skimage.transform import resize

data_path = r"C:\Users\20164798\OneDrive - TU Eindhoven\UNI\BMT 3\BEP\data\prepared"
image_rows = 420
image_cols = 580

img_rows = 96
img_cols = 96

################################################################################

def create_data(data_path, folder_name, data_name, masks_present = True, save_path = data_path):
    """

    DESCRIPTION: 
    -------
    
    read data and masks, save them in seperate npy files
    """    
    dataPath = os.path.join(data_path, folder_name)
    images = os.listdir(dataPath)
    images = [image for image in images if image.find(".tif") > -1]
    total = len(images)
    
    if masks_present: total = int(total/2)

    imgs = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)
    if masks_present: imgs_mask = np.ndarray((total, image_rows, image_cols), dtype=np.uint8)

    i = 0
    print('-'*30)
    print('Creating images...')
    print('-'*30)
    
    for image_name in images:
        if 'mask' in image_name:
            continue
        
        img = imread(os.path.join(dataPath, image_name), as_gray=True)
        img = np.array([img])
        imgs[i] = img
        
        if masks_present: 
            image_mask_name = image_name.split('.')[0] + '_mask.tif'
            img_mask = imread(os.path.join(dataPath, image_mask_name), as_gray=True)
            img_mask = np.array([img_mask])
            imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save(os.path.join(save_path, data_name +' - imgs.npy'), imgs)
    if masks_present: np.save(os.path.join(save_path, data_name + ' - imgs_mask.npy'), imgs_mask)
    print('Saving to .npy files done.')


def load_data(data_path, imgs_data_name = "imgs.npy", masks_data_name = "imgs_mask.npy"):
    """
    load the saved masks and data

    Returns
    -------
    imgs_train : TYPE
        DESCRIPTION.
    imgs_mask_train : TYPE
        DESCRIPTION.

    """
    imgs_train = np.load(os.path.join(data_path, imgs_data_name))
    imgs_mask_train = np.load(os.path.join(data_path, masks_data_name))
    
    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)
    
    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std
    

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]
    
    return imgs_train, imgs_mask_train

def preprocess(imgs):
    """
    DESCRIPTION:
    -----------
    resize data
    
    Parameters
    ----------
    imgs : TYPE
        image data.

    Returns
    -------
    imgs_p : TYPE
        preprocessed data.

    """
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p
