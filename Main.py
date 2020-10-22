from __future__ import print_function

import os

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, LearningRateScheduler
from tensorflow.keras import backend as K
from tensorflow_addons.layers import InstanceNormalization, GroupNormalization, WeightNormalization
from tensorflow.keras.layers import BatchNormalization, LayerNormalization

from Data import load_data, save_results, image_transformation, print_func, downsample_image
from sklearn.model_selection import KFold
from time import time

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

from Model import Unet, Mnet, eval_Mnet, schedule


data_path = r"/home/jpavboxtel/data"

data_path = r"C:\Users\20164798\OneDrive - TU Eindhoven\UNI\BMT 3\BEP\data\prepared"
imgs = "True - imgs.npy"
msks = "True - imgs_mask.npy"
tstimgs = "True test - imgs.npy"
tstmsks = "True test - imgs_mask.npy"


def train_model(data_path=data_path, imgs=imgs, msks=msks, tstimgs="", tstmsks="", model_name="model", save_path = "models", num_folds=5, batch_size=32, learning_rate=1e-5, upconv=False, nr_epochs=50, verbosity=1, start_ch=32, depth=4, inc_rate=2, kernel_size=(3, 3), activation='relu', normalization=None, dropout=0, elastic_deform = None, low_pass = None, high_pass = None, prwt = False, lr_decay = False, model_net = Unet, final_test=False, monitor="val_loss"):
    
    print_func('Loading and preprocessing train data...')
    
    images, masks = load_data(data_path, imgs, msks, low_pass=low_pass, high_pass=high_pass, prwt=prwt)
    
    if final_test: 
        print_func('Loading and preprocessing test data...')
        test_images, test_masks = load_data(data_path, tstimgs, tstmsks, low_pass=low_pass, high_pass=high_pass, prwt=prwt)
        monitor = "loss"
        num_folds = 10
    
    arg_dict_model = {"start_ch": start_ch, "depth": depth, "inc_rate": inc_rate, "kernel_size": kernel_size, "activation": activation, "normalization": normalization, "dropout": dropout, "learning_rate": learning_rate, "upconv": upconv}
     
    kfold = KFold(n_splits=num_folds, shuffle=True)
    dice_per_fold = [] 
    time_per_fold = []

    fold_no = 1
    for train, val in kfold.split(images, masks):
        if not final_test:
            train_im, train_msk, val_im, val_msk = images[train], masks[train], images[val], masks[val] 
        else:
            train_im, train_msk, val_im, val_msk = images, masks, test_images, test_masks 
        
        if elastic_deform is not None: train_im, train_msk = image_transformation(train_im, train_msk, elastic_deform)

        print_func(f'Training for fold {fold_no} (of {num_folds}) ... \nModel name: {model_name}')

        model = model_net(**arg_dict_model)
        
        save_dir = save_path + '/' + model_name + " K_" + str(fold_no)
        #model_checkpoint = ModelCheckpoint(save_dir + ' weights.h5', monitor=monitor, save_best_only=True)
        csv_logger = CSVLogger(os.path.join(save_dir + ' log.out'), append=True, separator=';')
        earlystopping = EarlyStopping(monitor = monitor, verbose = 1, min_delta = 0.0001, patience = 5, mode = 'auto', restore_best_weights = True)
        
        callbacks_list = [csv_logger, earlystopping] #,model_checkpoint]
        
        if lr_decay:
            lr_schedule = LearningRateScheduler(schedule, verbose = 1)
            callbacks_list.append(lr_schedule)
            
        if model_net == Mnet:
            print_func("prepare data for Mnet")
            train_msk = downsample_image(train_msk, 3)
            val_msk = downsample_image(val_msk, 3)
            
        arg_dict_fit = {"x": train_im, "y": train_msk, "validation_data": (val_im, val_msk), "batch_size": batch_size, "epochs": nr_epochs, "verbose": verbosity, "shuffle": True}
        
        if final_test: del arg_dict_fit["validation_data"]
        
        start_time = time()
        model.fit(callbacks=callbacks_list, **arg_dict_fit)
        train_time = int(time()-start_time)
        
        if model_net == Mnet:
            scores = eval_Mnet(val_im, val_msk['o1'], model, verbose=1)
        else:
            scores = model.evaluate(val_im, val_msk, verbose=0)
            
        print_func(f"Scores \nDice: {scores[1]} \nTime: {train_time}")
        dice_per_fold.append(scores[1])
        time_per_fold.append(train_time)
        save_results(model_name + f' K_{fold_no}', scores[1], train_time)
        
        fold_no = fold_no + 1
        
    
    save_results(model_name, dice_per_fold, time_per_fold, False) 
    
if __name__ == '__main__':
    train_model(nr_epochs=1, model_net=Mnet, model_name="test Mnet")
    
    
