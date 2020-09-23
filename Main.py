from __future__ import print_function

import os

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from tensorflow.keras import backend as K
from Data import load_data, save_results, image_transformation
from sklearn.model_selection import KFold
from time import time

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

from Model import Unet


data_path = r"/home/jpavboxtel/data"

data_path = r"C:\Users\20164798\OneDrive - TU Eindhoven\UNI\BMT 3\BEP\data\prepared"
imgs = "train - imgs.npy"
msks = "train - imgs_mask.npy"

def train_model(data_path=data_path, imgs=imgs, msks=msks, model_name="model", save_path = "modelskl", num_folds=5, batch_size=32, learning_rate=1e-5, upconv=False, nr_epochs=50, verbosity=1, start_ch=32, depth=4, inc_rate=2, kernel_size=(3, 3), activation='relu', normalization=None, dropout=0, elastic_deform = None, low_pass = None, high_pass = None):
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    
    images, masks = load_data(data_path, imgs, msks)
            
    kfold = KFold(n_splits=num_folds, shuffle=True)
    dice_per_fold = []
    time_per_fold = []

    fold_no = 1
    for train, val in kfold.split(images, masks):
        train_im = images[train]
        train_msk = masks[train]
        
        if elastic_deform is not None or low_pass is not None or high_pass is not None: 
            print("Data Augmentation")
            train_im, train_msk = image_transformation(train_im, train_msk, elastic_deform, low_pass, high_pass)
        
        print('-'*30)
        print(f'Training for fold {fold_no} (of {num_folds}) ...')
        print(f'Model name: {model_name}')
        print('-'*30)
        
        model = Unet(start_ch=start_ch, depth=depth, inc_rate=inc_rate, 
                      kernel_size = kernel_size, activation=activation, 
                      normalization=normalization, dropout=dropout, learning_rate = learning_rate, upconv = upconv)
        
        save_dir = save_path + '/' + model_name + " K_" + str(fold_no)
        model_checkpoint = ModelCheckpoint(save_dir + ' weights.h5', monitor='val_loss', save_best_only=True)
        csv_logger = CSVLogger(os.path.join(save_dir + ' log.out'), append=True, separator=';')
        earlystopping = EarlyStopping(monitor = 'val_loss', verbose = 1, min_delta = 0.0001, patience = 5, mode = 'auto', restore_best_weights = True)
        
        callbacks_list = [csv_logger, model_checkpoint, earlystopping]
        
        start_time = time()
        model.fit(train_im, train_msk, validation_data = (images[val], masks[val]), batch_size=batch_size, epochs=nr_epochs, verbose=verbosity, shuffle=True, callbacks=callbacks_list)
        train_time = int(time()-start_time)
        
        scores = model.evaluate(images[val], masks[val], verbose=0)
        dice_per_fold.append(scores[1])
        time_per_fold.append(train_time)
        save_results(model_name + f' K_{fold_no}', scores[1], train_time)
        
        fold_no = fold_no + 1
        
    
    save_results(model_name, dice_per_fold, time_per_fold, False) 
    
if __name__ == '__main__':
    train_model()
    
    
    
