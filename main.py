# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 11:25:09 2020

@author: 20164798
"""

from data import load_generators, band_pass_filter
from model import Unet

import os
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger

from tensorflow_addons.layers import InstanceNormalization, GroupNormalization, WeightNormalization
from tensorflow.keras.layers import BatchNormalization, LayerNormalization

import random
import shutil

import csv
import numpy as np


data_path = r"C:\Users\20164798\OneDrive - TU Eindhoven\UNI\BMT 3\BEP\data\prepared k-cross"
save_path = ""
model_name = "Test Model"

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest', 
                    preprocessing_function = band_pass_filter)

val_gen_args = dict(preprocessing_function = band_pass_filter)

def train_model(data_path, save_path = "models", model_name = "model", data_gen_args=dict(), val_gen_args = dict(), 
                batch_size = 32, nr_epochs=2, start_ch = 32, depth = 4, inc_rate = 2, kernel_size = (3, 3), 
                activation = 'relu', normalization = None, dropout = 0, verbose = 1, train_steps = None, val_steps = None):
    
    trainGene, valGene = load_generators(data_path, data_gen_args, val_gen_args)
    
    no_training_imgs = len(os.listdir(data_path + '/train/train_frames'))
    no_val_imgs = len(os.listdir(data_path + '/val/val_frames'))
    train_steps = no_training_imgs//batch_size if train_steps is None else train_steps
    val_steps = no_val_imgs//batch_size if val_steps is None else val_steps
    
    model = Unet(start_ch=start_ch, depth=depth, inc_rate=inc_rate, 
                 kernel_size = kernel_size, activation=activation, 
                 normalization=normalization, dropout=dropout)  
    
    weights_path = os.path.join(save_path, model_name)
    csv_logger = CSVLogger(os.path.join(weights_path + ' log.out'), append=True, separator=';')
    model_checkpoint = ModelCheckpoint(weights_path  + ' - weights.h5', monitor='val_loss', save_best_only=True)
    
    callbacks_list = [csv_logger, model_checkpoint]
     
    model.fit(trainGene, steps_per_epoch = train_steps, validation_data = valGene, 
              validation_steps = val_steps, epochs = nr_epochs, callbacks = callbacks_list, verbose = verbose)
    
    model.load_weights(weights_path + ' - weights.h5')
    results = model.evaluate(valGene, steps = val_steps)
    results = dict(zip(model.metrics_names, results))
    
    return results, weights_path, model_name


def add_frames(source_path, target_path, image):
    source = source_path+'/'+image
    target = target_path+'/'+image
    shutil.move(source, target)
  
  
def k_cross_val(data_path, model = None, k = 5):
    
    VALIDATION_DICE = []
    VALIDATION_LOSS = []

    all_img = os.listdir(os.path.join(data_path, 'train', 'train_frames'))
    if 'val' not in os.listdir(data_path): os.makedirs(data_path + '/val')

    random.seed(230)
    random.shuffle(all_img)
    
    for i in range(k):
        
        print("=" * 15 + " Fold Nr: " + str(i+1) + " " + "="*15)
        start_split = int(i/k*len(all_img))
        end_split = int((i+1)/k*len(all_img))
        val_frames = all_img[start_split:end_split]
        val_masks = [frame[:-4] + "_mask" + frame[-4:] for frame in val_frames]
        
        folders = [(val_frames, 'val_frames', 'train_frames'), (val_masks, 'val_masks', 'train_masks')]
        

        
        forward = True
        for j in range(2):
            for folder in folders:
                
                if folder[1] not in os.listdir(data_path + '/val'): os.makedirs(data_path + '/val/' + folder[1])
        
                array = folder[0]
                target_path = len(array) * [os.path.join(data_path, 'val', folder[1])]
                source_path = len(array) * [os.path.join(data_path, 'train', folder[2])]
                
                if forward:
                    list(map(add_frames, source_path, target_path, array))
                else:
                    list(map(add_frames, target_path, source_path,  array))
                    os.rmdir(target_path[0])
            if forward == True and model is not None: 
                results, weights_path, model_name = model()
                shutil.move(weights_path + ' - weights.h5', weights_path + ' K_' + str(i) + ' - weights.h5')
                shutil.move(weights_path + ' log.out', weights_path + ' K_' + str(i) + ' log.out')
                VALIDATION_DICE.append(results['dice_coef'])
                VALIDATION_LOSS.append(results['loss'])           
            
            forward = False 
           
    os.rmdir(data_path + '/val')
    saveresults(model_name, np.mean(VALIDATION_DICE), np.std(VALIDATION_DICE), np.mean(VALIDATION_LOSS), np.std(VALIDATION_LOSS))

def saveresults(model_name, mean_dice, std_dice, mean_loss, std_loss):
    files = os.listdir()
    with open('results.csv', 'a', newline="") as file:
        writer = csv.writer(file,  delimiter=';')
        if 'results.csv' not in files: writer.writerow(["Model_name", "mean_dice_score", "std_dice_score", "mean_loss", "std_loss"])
        writer.writerow([model_name, mean_dice, std_dice, mean_loss, std_loss])
        file.close()
    
def helperfunc(data_path = data_path):
    folders = [('val_frames', 'train_frames'), ('val_masks', 'train_masks')]
    
    for folder in folders:
        all_img = os.listdir(os.path.join(data_path, 'val', folder[0]))
        target_path = len(all_img) * [os.path.join(data_path, 'train', folder[1])]
        source_path = len(all_img) * [os.path.join(data_path, 'val', folder[0])]
        
        list(map(add_frames, source_path, target_path, all_img))
        os.rmdir(source_path[0])
    os.rmdir(data_path + '/val')
               
            
        
k_cross_val(data_path, lambda: train_model(data_path, train_steps=10, val_steps=5, nr_epochs=2),  k=5)    


