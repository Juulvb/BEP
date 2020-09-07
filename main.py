# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 11:25:09 2020

@author: 20164798
"""

from data import *
from model import *

data_path_npy = r"C:\Users\20164798\OneDrive - TU Eindhoven\UNI\BMT 3\BEP\data\prepared"
data_path_generators = r"C:\Users\20164798\OneDrive - TU Eindhoven\UNI\BMT 3\BEP\data\prepared - generator"
imgs_data_name = 'train - imgs.npy'
masks_data_name = 'train - imgs_mask.npy'
model_name = "M1"
batch_size = 32
save_path = ""
nr_epochs = 2
verbose = 1
val_split = 0.15

data_created = True


if not data_created: 
    create_data_npy(data_path_npy, 'train', 'train')
    create_data_npy(data_path_npy, 'test - self', 'test (own)')
    create_data_npy(data_path_npy, 'test - final', 'test (final)', masks_present = False)


def data_from_npy():
    print("preparing data")
    imgs_train, imgs_mask_train = load_data_npy(data_path_npy, imgs_data_name, masks_data_name, preprocessing=True)
    print("train model")  
    model.fit(imgs_train, imgs_mask_train, batch_size=batch_size, epochs=nr_epochs, verbose=verbose, shuffle=True,
              validation_split=val_split,
              callbacks=[model_checkpoint])  

def data_from_generator():
    print("preparing data from generators")
    trainGene, valGene = load_generators(data_path_generators, preprocessing=True)
    print("train model")  
    no_training_imgs = len(os.listdir(data_path_generators + '/train/train_frames'))
    no_val_imgs = len(os.listdir(data_path_generators + '/val/val_frames'))
    train_steps = no_training_imgs//batch_size
    val_steps = no_val_imgs//batch_size 
    
    model.fit(trainGene, steps_per_epoch = train_steps, validation_data = valGene, validation_steps = val_steps, epochs = nr_epochs,
              callbacks = [model_checkpoint])

model = Unet()  
model_checkpoint = ModelCheckpoint(os.path.join(save_path, model_name + ' - weights.h5'), monitor='val_loss', save_best_only=True)
data_from_generator()