"""
this script read all training and test data and masks. 
prepare training data and masks and save them in a binary format.
prepare test data and test ids and save them in a binary format.

"""

from __future__ import print_function

import os
import numpy as np
import csv

from skimage.io import imsave, imread
from skimage.transform import resize


image_rows = 420
image_cols = 580

img_rows = 96
img_cols = 96

################################################################################

def create_data(data_path, data_name, masks_present = True, save_path = None):
    """

    DESCRIPTION: 
    -------
    
    read data and masks, save them in seperate npy files
    """    
    if save_path is None: save_path = data_path
    images = os.listdir(data_path)
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
        
        img = imread(os.path.join(data_path, image_name), as_gray=True)
        img = np.array([img])
        imgs[i] = img
        
        if masks_present: 
            image_mask_name = image_name.split('.')[0] + '_mask.tif'
            img_mask = imread(os.path.join(data_path, image_mask_name), as_gray=True)
            img_mask = np.array([img_mask])
            imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print('Loading done.')

    np.save(os.path.join(save_path, data_name +' - imgs.npy'), imgs)
    if masks_present: np.save(os.path.join(save_path, data_name + ' - imgs_mask.npy'), imgs_mask)
    print('Saving to .npy files done.')


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

def load_data(data_path, imgs, msks):
    """
    load the saved masks and data

    Returns
    -------
    imgs_train : TYPE
        DESCRIPTION.
    imgs_mask_train : TYPE
        DESCRIPTION.

    """
    imgs_train = np.load(os.path.join(data_path, imgs))
    imgs_mask_train = np.load(os.path.join(data_path, msks))
    
    
    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)
    
    ###### data normalization ##########
    
    imgs_train = imgs_train.astype('float32')
    mean = np.mean(imgs_train)  # mean for data centering
    std = np.std(imgs_train)  # std for data normalization

    imgs_train -= mean
    imgs_train /= std
    
    #### 

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]
    
    return imgs_train, imgs_mask_train


def save_results(model_name, dice, time, elab=True):
    files = os.listdir()
    if elab:
        with open('results elaborate.csv', 'a', newline="") as file:
            writer = csv.writer(file, delimiter=';')
            if 'results.csv' not in files: writer.writerow(["Model_name", "Dice_score", "Time"])
            writer.writerow([model_name, dice, time])
            file.close()
    else:
        with open('results.csv', 'a', newline="") as file:
            writer = csv.writer(file,  delimiter=';')
            if 'results.csv' not in files: writer.writerow(["Model_name", "mean_dice_score", "std_dice_score", "mean_time", "std_time"])
            writer.writerow([model_name, np.mean(dice), np.std(dice), np.mean(time), np.std(time)])
            file.close()
        

        

