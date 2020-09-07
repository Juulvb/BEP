# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 11:12:04 2020

@author: 20164798
"""

import os
import random
import re
import shutil
from PIL import Image

DATA_PATH = r"C:\Users\20164798\OneDrive - TU Eindhoven\UNI\BMT 3\BEP\data\complete\train"
TARGET_PATH = r"C:\Users\20164798\OneDrive - TU Eindhoven\UNI\BMT 3\BEP\data\prepared - generator"
    
all_img = os.listdir(DATA_PATH)
all_frames = [img for img in all_img if (img.find("mask") == -1) and (img.find(".tif") > -1) ]

random.seed(230)
random.shuffle(all_frames)

train_split = int(0.65*len(all_frames))
val_split = int(0.75*len(all_frames))

train_frames = all_frames[:train_split]
val_frames = all_frames[train_split:val_split]
test_frames = all_frames[val_split:]

train_masks = [frame[:-4] + "_mask" + frame[-4:] for frame in train_frames]
val_masks = [frame[:-4] + "_mask" + frame[-4:] for frame in val_frames]
test_masks = [frame[:-4] + "_mask" + frame[-4:] for frame in test_frames]

def add_frames(dir_name, image):
  source = DATA_PATH+'/'+image
  target = TARGET_PATH+'/{}'.format(dir_name)+'/'+image
  shutil.copyfile(source, target)
  
  
all_folders = [(train_frames, 'train_frames'), (val_frames, 'val_frames'), 
                 (test_frames, 'test_frames'), (train_masks, 'train_masks'), (val_masks, 'val_masks'), 
                (test_masks, 'test_masks')]

for folder in all_folders:
    
    if folder[1] not in os.listdir(TARGET_PATH): os.makedirs(TARGET_PATH + '/' + folder[1])
    
    array = folder[0]
    name = [folder[1]] * len(array)

    list(map(add_frames, name, array))
    print(folder[1] + " move completed")


    
    
                       