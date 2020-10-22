"""
author: 
    J.P.A. van Boxtel
    j.p.a.v.boxtel@student.tue.nl
"""
from __future__ import print_function

import os
import numpy as np
import csv
import random

from skimage.io import imread
from skimage.transform import resize, downscale_local_mean
from skimage.filters import gaussian, prewitt

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def create_data(data_path, data_name, masks_present = True, save_path = None, image_rows = 420, image_cols = 580):
    """
    DESCRIPTION: read data and masks, save them in seperate npy files
    -------
    INPUTS: 
    data_path:      string, directory of the folder containing the images
    data_name:      string, name under which the npy files should be saved
    masks_present:  boolean, whether the data contains masks or not (so False for e.g. test data)
    save_path:      string, directory of the folder to save the npy files to
    image_rows:     number of pixel rows of input images
    image_cols:     number of pixel columns of input images
    -------
    OUTPUTS:
    1 npy file containing a numpy array with all given images
    *optional* 1 npy file containing a numpy array with all given masks
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

def elastic_transform(image, mask, alpha, sigma, random_state=None):
    """
    Description: Elastic deformation of image and corresponding mask as described in Simard et al. (2003).
    Code based on: https://www.kaggle.com/bguberfain/elastic-transform-for-data-augmentation 
    -------
    INPUTS: 
    image:          numpy array, the image to be transformed
    mask:           numpy array, the mask corresponding to the image
    alpha:          number, the scaling factor for the transformation
    sigma:          number, the standard deviation for the gaussian convolution
    random_state:   method used for generating random numbers for the deformation field
    -------
    OUTPUTS:
    1 numpy array containing the transformed image
    1 numpy array containing the transformed mask 
    """ 
    
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    #print(x.shape)
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    distored_mask = map_coordinates(mask, indices, order=1, mode='reflect')
    return distored_image.reshape(image.shape), distored_mask.reshape(mask.shape)

def image_transformation(images, masks, elastic_deform = None, low_pass = None, high_pass = None, prwt = False):
    """
    DESCRIPTION: Function containing all image pre-processing techniques applied to the images, including low-pass, high-pass, prewitt filters and elastic_deformation
    -------
    INPUTS: 
    images:         numpy array, containing all images to be transformed
    masks:          numpy array, containing all corresponding masks to the images
    elastic_deform: None or tuple of 2 numbers, (alpha, sigma) as described in function 'elastic_transform'
    low_pass:       None or int, giving the standard deviation used for the gaussian low-pass filter applied to the images
    high_pass:      None or int, giving the standard deviation used for the gaussian high_pass filter applied to the images
    prwt:           boolean, whether to apply a prewitt filter to the images or not
    -------
    OUTPUTS:
    1 numpy array containing all processed images
    1 numpy array containing all processed masks
    """ 

    im_tr = []
    msk_tr = []
    for i in range(len(images)):
        image = images[i]
        mask = masks[i]
        
        if low_pass is not None: 
            image = gaussian(image, low_pass)
        
        if high_pass is not None and not prewitt : 
            image = image - gaussian(image, high_pass)
        
        if prwt:
            image = prewitt(image)
        
        if elastic_deform is not None:
            alpha = random.uniform(1, elastic_deform[0])
            sigma = random.uniform(elastic_deform[1], 1)
            image, mask= elastic_transform(image, mask, image.shape[1]*alpha, image.shape[1]*sigma)
            
        im_tr.append(image)
        msk_tr.append(mask)
        
    return np.array(im_tr), np.array(msk_tr)

def reshape_imgs(imgs, img_rows, img_cols):
    """
    DESCRIPTION: reshapes the input images to the desired shapes
    -------
    INPUTS: 
    imgs:       numpy array containing all input images
    img_rows:   target number of rows for the images
    img_cols:   target number of columns for the images
    -------
    OUTPUTS:
    imgs_p:     numpy array with the reshaped images
    """ 

    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p

def load_data(data_path, imgs, msks, img_rows=96, img_cols=96, low_pass = None, high_pass = None, prwt = False):
    """
    DESCRIPTION:load the saved masks and data
    -------
    INPUTS:
    data_path:  string, directory of the folder containing the npy files with the images and masks
    imgs:       string, name of the npy file containing the images
    msks:       string, name of the npy file containing the masks
    img_rows:   int, desired number of rows for the images and masks to be reshaped to 
    img_cols:   int, desired number of columns for the images and masks to be reshaped to
    low_pass:   None or int, giving the standard deviation used for the gaussian low-pass filter applied to the images
    high_pass:  None or int, giving the standard deviation used for the gaussian high_pass filter applied to the images
    prwt:       boolean, whether to apply a prewitt filter to the images or not
    -------
    OUTPUTS:
    imgs_train:         numpy array with the preprocessed, reshaped, mean-centered, normalized images
    imgs_mask_train:    numpy array with the reshaped, normalized masks
    """
    
    imgs_train = np.load(os.path.join(data_path, imgs))
    imgs_mask_train = np.load(os.path.join(data_path, msks))
    
    
    if low_pass is not None or high_pass is not None or prwt: 
        print_func("Data Augmentation")
        imgs_train, imgs_mask_train = image_transformation(imgs_train, imgs_mask_train, low_pass=low_pass, high_pass=high_pass, prwt=prwt)

    imgs_train = reshape_imgs(imgs_train, img_rows, img_cols)
    imgs_mask_train = reshape_imgs(imgs_mask_train, img_rows, img_cols)
    
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


def save_results(model_name, dice, time, elab=True, file_total = 'results.csv', file_elab = 'results elaborate.csv'):
    """
    DESCRIPTION: helper function to easily save the model results to a csv file
    -------
    INPUTS:
    model_name: string, name to identify the model with
    dice:       number or list of numbers, the dice similarity score(s) of the model(s) 
    time:       number or list of numbers, the time(s) used for training the model(s)
    elab:       boolean, to determine whether to save the results as a result per fold in K-fold cross 
                validation (elab=True) or to save results as the means and standard deviations of all folds (elab=False)
    file_total: string, name of the file to save means and standard devitations to from all folds combined
    file_elab:  string, name of the file to save results to from each fold
    -------
    OUTPUTS:
    file_elab:  csv file containing the dice score and the train time per fold of the model (if file already exists, results will be appended)
    file_total: csv file containing the mean and standard deviation of the dice score and the train time per model (if file already exists, results will be appended)
    """
    
    files = os.listdir()
    if elab:
        with open(file_elab, 'a', newline="") as file:
            writer = csv.writer(file, delimiter=';')
            if file_elab not in files: writer.writerow(["Model_name", "Dice_score", "Time"])
            writer.writerow([model_name, dice, time])
            file.close()
    else:
        with open(file_total, 'a', newline="") as file:
            writer = csv.writer(file,  delimiter=';')
            if file_total not in files: writer.writerow(["Model_name", "mean_dice_score", "std_dice_score", "mean_time", "std_time"])
            writer.writerow([model_name, np.mean(dice), np.std(dice), np.mean(time), np.std(time)])
            file.close()

def downsample_image(images, n, img_cols = 96, img_rows = 96):
    """
    DESCRIPTION: helper function to prepare the data for M-net
    -------
    INPUTS:
    images: numpy array, all masks that need to be downsampled 
    n:      int, number of time the input needs to be downsampled (by factor 2)
    -------
    OUTPUTS:
    l: dictionary of n+1 keys and values, containing the original masks and each next element the downsampled by factor 2 version
    """
    l = {}
    l["o1"] = images
    for i in range(n):
        factor = 2**(i+1)
        arr = np.empty((len(images), images.shape[1], images.shape[2], images.shape[3]))
        for j in range(len(images)):
            ds = downscale_local_mean(images[j], (factor, factor, 1))
            arr[j] = resize(ds, (img_cols, img_rows, 1), preserve_range=True)
        l[f"o{i+2}"] = arr
    return l
                        
def print_func(str_in, c = '-', n=50):
    """
    DESCRIPTION: helper function to print information clearly in filled consoles
    -------
    INPUTS:
    str_in: string, the information that needs to be printed
    c:      string, the seperator characted to use
    n:      the number of seperator characters to print
    -------
    OUTPUTS:
    prints the input information with a leading and closing line of n seperator characters
    """
    print(c*n)
    print(str_in)
    print(c*n)