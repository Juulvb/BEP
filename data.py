# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 16:06:51 2020

@author: 20164798
"""

import os
import numpy as np

from skimage.io import imsave, imread
from keras.preprocessing.image import ImageDataGenerator
from skimage.transform import resize

import matplotlib.pyplot as plt
import numpy as np
from skimage.data import gravel
from skimage.filters import difference_of_gaussians, window
from scipy.fftpack import fftn, fftshift


image_rows = 420
image_cols = 580

img_rows = 96
img_cols = 96

def create_data_npy(data_path, folder_name, data_name, masks_present = True, save_path = None):
    """

    DESCRIPTION: 
    -------
    
    read data and masks, save them in seperate npy files
    """ 
    if save_path is None:
        save_path = data_path
        
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

def load_data_npy(data_path, imgs_data_name = "imgs.npy", masks_data_name = "imgs_mask.npy", preprocessing=False):
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

def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)

def getGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (96,96),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)
        
def load_generators(data_path, preprocessing = False, BATCH_SIZE = 32):    
    preprocessing_func = band_pass_filter if preprocessing == True else None
    data_gen_args = dict(rotation_range=0.2,
                        width_shift_range=0.05,
                        height_shift_range=0.05,
                        shear_range=0.05,
                        zoom_range=0.05,
                        horizontal_flip=True,
                        fill_mode='nearest', 
                        preprocessing_function = preprocessing_func)
    val_gen_args = dict(preprocessing_function = preprocessing_func)
    
    trainGene = getGenerator(BATCH_SIZE,data_path + '/train','train_frames','train_masks',data_gen_args,save_to_dir = None)
    valGene = getGenerator(BATCH_SIZE,data_path + '/val','val_frames','val_masks',val_gen_args, save_to_dir = None)
    
    return trainGene, valGene

def band_pass_filter(image):
    low_sigma=1
    high_sigma=12
    filtered_image = difference_of_gaussians(image, low_sigma, high_sigma)
    return filtered_image
    