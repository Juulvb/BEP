# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 11:27:45 2020

@author: 20164798
"""
from Model import Unet
from Data import load_data, save_results, image_transformation
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
import os
from time import time


data_path = r"/home/jpavboxtel/data"
save_path = r"/home/jpavboxtel/code/final_models"

def test_model(exp_name, imgs, msks, test_imgs, test_msks, elastic_deform = None, data_path = r"/home/jpavboxtel/data", save_path = r"/home/jpavboxtel/code/final_models"):
    print("load test_images")
    test_images, test_masks = load_data(data_path, test_imgs, test_msks)
    print("load train_images")
    images, masks = load_data(data_path, imgs, msks)

    if elastic_deform is not None: 
        print('-'*30)
        print("Data Augmentation")
        print('-'*30)
        images, masks = image_transformation(images, masks, elastic_deform)
    
    dice_list = []
    time_list = []
    for i in range(10):
        print(f"train model run {i}/10 of test: {exp_name}")
        model = Unet(start_ch=32, depth=4, inc_rate=2, kernel_size = (5,5), activation='relu', normalization = None, dropout = 0.4,learning_rate = 1e-5, upconv = False)
        
        save_dir = f"{save_path}/{exp_name} N_{i}"
        model_checkpoint = ModelCheckpoint(save_dir + ' weights.h5', monitor='loss', save_best_only=True)
        csv_logger = CSVLogger(os.path.join(save_dir + ' log.out'), append=True, separator=';')
        earlystopping = EarlyStopping(monitor = 'loss', verbose = 1, min_delta = 0.0001, patience = 5, mode = 'auto', restore_best_weights = True)
        
        callbacks_list = [csv_logger, model_checkpoint, earlystopping]
        start_time = time()
        model.fit(images, masks, batch_size=16, epochs=50, verbose=1, shuffle=True, callbacks=callbacks_list)
        train_time = int(time()-start_time)
        
        score = model.evaluate(test_images, test_masks, verbose=1)
        dice_list.append(score[1])
        time_list.append(train_time)
        save_results(exp_name + f"N_{i}", score[1], train_time, file_elab = "test results elaborate.csv")
    
    save_results(exp_name, dice_list, time_list, False, file = "test results.csv")

# images = np.load(os.path.join(data_path, "test (own) - imgs.npy"))
# masks = np.load(os.path.join(data_path, "test (own) - imgs_mask.npy"))
#img, msk, tstimg, tstmsk = test_model("True - 4", "True - imgs.npy", "True - imgs_mask.npy", "True test - imgs.npy", "True test - imgs_mask.npy", data_path=data_path, save_path=save_path)
 