# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 11:25:09 2020

@author: 20164798
"""

from data import load_generators, band_pass_filter
from model import Unet

import os
from keras.callbacks import ModelCheckpoint

from tensorflow_addons.layers import InstanceNormalization, GroupNormalization, WeightNormalization
from tensorflow.keras.layers import BatchNormalization, LayerNormalization


data_path_npy = r"C:\Users\20164798\OneDrive - TU Eindhoven\UNI\BMT 3\BEP\data\prepared"
data_path_generators = r"C:\Users\20164798\OneDrive - TU Eindhoven\UNI\BMT 3\BEP\data\prepared - generator"
imgs_data_name = 'train - imgs.npy'
masks_data_name = 'train - imgs_mask.npy'
model_name = "M1"
batch_size = 32
save_path = ""
nr_epochs = 2
verbose = 1

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest', 
                    preprocessing_function = band_pass_filter)

val_gen_args = dict(preprocessing_function = band_pass_filter)


trainGene, valGene = load_generators(data_path_generators, data_gen_args, val_gen_args)

no_training_imgs = len(os.listdir(data_path_generators + '/train/train_frames'))
no_val_imgs = len(os.listdir(data_path_generators + '/val/val_frames'))
train_steps = no_training_imgs//batch_size
val_steps = no_val_imgs//batch_size 

model = Unet()  
model_checkpoint = ModelCheckpoint(os.path.join(save_path, model_name + ' - weights.h5'), monitor='val_loss', save_best_only=True)
    
model.fit(trainGene, steps_per_epoch = train_steps, validation_data = valGene, validation_steps = val_steps, epochs = nr_epochs,
              callbacks = [model_checkpoint])



