# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 14:30:23 2020

@author: 20164798
"""
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 11:25:09 2020

@author: 20164798
"""

from data import load_data, image_transformation
from model import Unet

import os
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping

#from tensorflow_addons.layers import InstanceNormalization, GroupNormalization, WeightNormalization
#from tensorflow.keras.layers import BatchNormalization, LayerNormalization

import csv
import numpy as np
from time import time


data_path = r"C:\Users\20164798\OneDrive - TU Eindhoven\UNI\BMT 3\BEP\data\prepared"
imgs = "train - imgs.npy"
msks = "train - imgs_mask.npy"

def train_model(data_path, imgs, msks, save_path = "models", model_name = "model",  
                batch_size = 32, nr_epochs=50, start_ch = 32, depth = 4, inc_rate = 2, kernel_size = (3, 3), learning_rate = 1e-5, 
                activation = 'relu', normalization = None, dropout = 0, verbose = 1, train_steps = None, val_steps = None, upconv = True, k = 5, small = False, 
                elastic_deform = False, band_pass_filter = False):
    print("Load data")
    images, masks = load_data(data_path, imgs, msks)
    if small: 
        images = images[:len(images)//100]
        masks = masks[:len(masks)//100]
    
    if elastic_deform or band_pass_filter: 
        print("Data Augmentation")
        images, masks = image_transformation(images, masks, elastic_deform, band_pass_filter)
    
    nro_imgs = len(images)
    VAL_DICE = []
    VAL_LOSS = []
    TIME = []
        
    for i in range(k):
        print("=" * 15 + " Fold Nr: " + str(i+1) + " " + "="*15 + "\nModel Name: " + model_name)
        start_split = int(i/k*nro_imgs)
        end_split = int((i+1)/k*nro_imgs)
        
        val_img = images[start_split:end_split]
        val_msks = masks[start_split:end_split]
        
        train_img = np.concatenate((images[:start_split], images[end_split:])) if start_split != 0 and end_split != nro_imgs else images[end_split:] if start_split == 0 else images[:start_split]
        train_msks = np.concatenate((masks[:start_split], images[end_split:])) if start_split != 0 and end_split != nro_imgs else masks[end_split:] if start_split == 0 else masks[:start_split]
        print(len(val_img), len(train_img))
            
        model = Unet(start_ch=start_ch, depth=depth, inc_rate=inc_rate, 
                      kernel_size = kernel_size, activation=activation, 
                      normalization=normalization, dropout=dropout, learning_rate = learning_rate, upconv = upconv)  

        weights_path = os.path.join(save_path, model_name + " K_" + str(i))
        csv_logger = CSVLogger(os.path.join(weights_path + ' log.out'), append=True, separator=';')
        earlystopping = EarlyStopping(monitor = 'val_loss', verbose = 1, min_delta = 0.0001, patience = 3, mode = 'auto', restore_best_weights = True)
        model_checkpoint = ModelCheckpoint(weights_path  + ' - weights.h5', monitor='val_loss', save_best_only=True)
        
        callbacks_list = [csv_logger, model_checkpoint, earlystopping]
     
        start_time = time()
        model.fit(train_img, train_msks, batch_size = batch_size, validation_data = (val_img, val_msks), epochs = nr_epochs, callbacks = callbacks_list, verbose = verbose, shuffle=True)
        train_time = start_time - time()
        
        model.load_weights(weights_path + ' - weights.h5')
        results = model.evaluate(val_img, val_msks)
        VAL_LOSS.append(results[0])
        VAL_DICE.append(results[1])
        TIME.append(train_time)
    
    saveresults(model_name, np.mean(VAL_DICE), np.std(VAL_DICE), 
            np.mean(VAL_LOSS), np.std(VAL_LOSS),
            np.mean(TIME), np.std(TIME))


def saveresults(model_name, mean_dice, std_dice, mean_loss, std_loss, mean_time, std_time):
    files = os.listdir()
    with open('results.csv', 'a', newline="") as file:
        writer = csv.writer(file,  delimiter=';')
        if 'results.csv' not in files: writer.writerow(["Model_name", "mean_dice_score", "std_dice_score", "mean_loss", "std_loss", "mean_time", "std_time"])
        writer.writerow([model_name, mean_dice, std_dice, mean_loss, std_loss, mean_time, std_time])
        file.close()
    
          
train_model(data_path, imgs, msks, nr_epochs=2, small=True, elastic_deform = True)          
        
   


