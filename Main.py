from __future__ import print_function

import os

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, LearningRateScheduler
from tensorflow.keras import backend as K
from tensorflow_addons.layers import InstanceNormalization, GroupNormalization, WeightNormalization
from tensorflow.keras.layers import BatchNormalization, LayerNormalization
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

from sklearn.model_selection import KFold
from time import time

from Data import load_data, save_results, image_transformation, print_func, downsample_image
from Model import Unet, Mnet, eval_Mnet, schedule


data_path = r"C:\Users\20164798\OneDrive - TU Eindhoven\UNI\BMT 3\BEP\data\prepared"
imgs = "train - imgs.npy"
msks = "train - imgs_mask.npy"
tstimgs = "test (own) - imgs.npy"
tstmsks = "test (own) - imgs_mask.npy"


def train_model(data_path=data_path, imgs=imgs, msks=msks, tstimgs="", tstmsks="", model_name="model", save_path = "models", num_folds=5, batch_size=32, learning_rate=1e-5, up=False, nr_epochs=50, verbosity=1, start_ch=32, depth=4, inc_rate=2, kernel_size=(3, 3), activation='relu', normalization=None, dropout=0, elastic_deform = None, low_pass = None, high_pass = None, prwt = False, lr_decay = False, model_net = Unet, final_test=False, monitor="val_loss"):
    
    ##### load data and optional test data #####
    print_func('Loading and preprocessing train data...')
    images, masks = load_data(data_path, imgs, msks, low_pass=low_pass, high_pass=high_pass, prwt=prwt)
    
    if final_test: 
        print_func('Loading and preprocessing test data...')
        test_images, test_masks = load_data(data_path, tstimgs, tstmsks, low_pass=low_pass, high_pass=high_pass, prwt=prwt)
        monitor = "loss"
        num_folds = 10
    
    ##### save arguments for the model to dictionairy #####
    arg_dict_model = {"start_ch": start_ch, "depth": depth, "inc_rate": inc_rate, "kernel_size": kernel_size, "activation": activation, "normalization": normalization, "dropout": dropout, "learning_rate": learning_rate, "up": up}
    
    ##### prepare for k-fold cross validation #####
    kfold = KFold(n_splits=num_folds, shuffle=True)
    fold_no = 1
    dice_per_fold, time_per_fold = [], []

    for train, val in kfold.split(images, masks):
        print_func(f'Training for fold {fold_no} (of {num_folds}) ... \nModel name: {model_name}')
        
        ##### prepare datasets #####
        if not final_test: #seperate data into train and validation set according to k-fold
            train_im, train_msk, val_im, val_msk = images[train], masks[train], images[val], masks[val] 
        else: #use all train data (no validation set) and test on test data
            train_im, train_msk, val_im, val_msk = images, masks, test_images, test_masks 
        
        if elastic_deform is not None: train_im, train_msk = image_transformation(train_im, train_msk, elastic_deform)
        
        if model_net == Mnet: #M-net has multiple outputs, the train and validation masks are downsampled to match the outputs of the model
            print_func("prepare data for Mnet")
            train_msk = downsample_image(train_msk, depth-1)
            val_msk = downsample_image(val_msk, depth-1)
        
        ##### load model with random initialized weights ######
        model = model_net(**arg_dict_model)
        
        ##### load callbacks #####
        save_dir = save_path + '/' + model_name + " K_" + str(fold_no)
        callbacks_list = []
        callbacks_list.append(ModelCheckpoint(save_dir + ' weights.h5', monitor=monitor, save_best_only=True))
        callbacks_list.append(CSVLogger(os.path.join(save_dir + ' log.out'), append=True, separator=';'))
        callbacks_list.append(EarlyStopping(monitor = monitor, verbose = 1, min_delta = 0.0001, patience = 5, mode = 'auto', restore_best_weights = True))
        
        
        if lr_decay:
            lr_schedule = LearningRateScheduler(schedule, verbose = 1)
            callbacks_list.append(lr_schedule)
        
        ##### fit model #####
        arg_dict_fit = {"x": train_im, "y": train_msk, "validation_data": (val_im, val_msk), "batch_size": batch_size, "epochs": nr_epochs, "verbose": verbosity, "shuffle": True}
        if final_test: del arg_dict_fit["validation_data"]
        
        start_time = time()
        model.fit(callbacks=callbacks_list, **arg_dict_fit)
        train_time = int(time()-start_time)
        
        ##### evaluate model #####
        if model_net == Mnet:
            scores = eval_Mnet(val_im, val_msk['o1'], model, verbose=1)
        else:
            scores = model.evaluate(val_im, val_msk, verbose=0)
        
        ##### save scores of fold #####
        print_func(f"Scores \nDice: {scores[1]} \nTime: {train_time}")
        dice_per_fold.append(scores[1])
        time_per_fold.append(train_time)
        save_results(model_name + f' K_{fold_no}', scores[1], train_time)
        
        fold_no += 1 
    
    ##### save scores of model #####
    save_results(model_name, dice_per_fold, time_per_fold, False) 

    
if __name__ == '__main__':
    result = train_model()
    
    
