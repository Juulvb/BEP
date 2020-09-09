# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 14:06:33 2020

@author: 20164798
"""
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
TARGET_PATH = r"C:\Users\20164798\OneDrive - TU Eindhoven\UNI\BMT 3\BEP\data\prepared k-cross"
    
all_img = os.listdir(DATA_PATH)
all_frames = [img for img in all_img if (img.find("mask") == -1) and (img.find(".tif") > -1) ]

random.seed(230)
random.shuffle(all_frames)

train_split = int(0.75*len(all_frames))

train_frames = all_frames[:train_split]
test_frames = all_frames[train_split:]

train_masks = [frame[:-4] + "_mask" + frame[-4:] for frame in train_frames]
test_masks = [frame[:-4] + "_mask" + frame[-4:] for frame in test_frames]

def add_frames(source_path, target_path, image):
  source = source_path+'/'+image
  target = target_path+'/'+image
  shutil.copyfile(source, target)
  
  
all_folders = [((train_frames, 'train_frames'), (train_masks, 'train_masks'), 'train'),
               ((test_frames, 'test_frames'), (test_masks, 'test_masks'), 'test')]

for subfolder in all_folders:
    
    if subfolder[2] not in os.listdir(TARGET_PATH): os.makedirs(TARGET_PATH + '/' + subfolder[2])
    
    for folder in subfolder:
        if type(folder)==str: continue
            
        if folder[1] not in os.listdir(TARGET_PATH + '/' + subfolder[2]): os.makedirs(TARGET_PATH + '/' + subfolder[2] + '/' + folder[1])

        array = folder[0]
        target_path = [TARGET_PATH + '/' + subfolder[2] + '/' + folder[1]] * len(array)
        source_path = [DATA_PATH] * len(array)
    
        list(map(add_frames, source_path, target_path, array))
        print(folder[1] + " copy completed")


    
    
                       