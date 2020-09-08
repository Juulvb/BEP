# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 11:25:09 2020

@author: 20164798
"""

from data import load_generators, band_pass_filter
from model import Unet

import os
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger

from tensorflow_addons.layers import InstanceNormalization, GroupNormalization, WeightNormalization
from tensorflow.keras.layers import BatchNormalization, LayerNormalization

data_path = r"C:\Users\20164798\OneDrive - TU Eindhoven\UNI\BMT 3\BEP\data\prepared - generator"
save_path = ""
model_name = "M2"

batch_size = 32
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


trainGene, valGene = load_generators(data_path, data_gen_args, val_gen_args)

no_training_imgs = len(os.listdir(data_path + '/train/train_frames'))
no_val_imgs = len(os.listdir(data_path + '/val/val_frames'))
train_steps = no_training_imgs//batch_size
val_steps = no_val_imgs//batch_size 

model = Unet(start_ch=32, depth=4, inc_rate=2, 
             kernel_size = (3, 3), activation='relu', 
             normalization=BatchNormalization, dropout=0)  

csv_logger = CSVLogger(os.path.join(save_path, model_name + ' log.out'), append=True, separator=';')
model_checkpoint = ModelCheckpoint(os.path.join(save_path, model_name + ' - weights.h5'), 
                                   monitor='val_loss', save_best_only=True)

callbacks_list = [csv_logger, model_checkpoint]
 
model.fit(trainGene, steps_per_epoch = train_steps, validation_data = valGene, 
          validation_steps = val_steps, 
          epochs = nr_epochs, 
          callbacks = callbacks_list)



