from __future__ import print_function
from sklearn.model_selection import KFold
from time import time

import os
import copy
import pandas as pd
import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, LearningRateScheduler
from tensorflow.keras.layers import BatchNormalization, LayerNormalization
from tensorflow.keras.models import model_from_json
from tensorflow_addons.layers import InstanceNormalization, GroupNormalization, WeightNormalization

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

from Data import load_data, save_results, image_transformation, print_func, downsample_image, post_process_openingbyreconstruction, post_process_thresholding, post_process_smoothingedges
from Model import Unet, Mnet, eval_Mnet, schedule, dice_coef_pred


data_path = r"C:\Users\20164798\OneDrive - TU Eindhoven\UNI\BMT 3\BEP\data\prepared"
imgs = "train - imgs.npy"
msks = "train - imgs_mask.npy"
tstimgs = "test (own) - imgs.npy"
tstmsks = "test (own) - imgs_mask.npy"


def train_model(data_path=data_path, imgs=imgs, msks=msks, tstimgs="", tstmsks="", model_name="model", save_path = "models", num_folds=5, batch_size=32, learning_rate=1e-5, nr_epochs=50, verbosity=1, up=False, start_ch=32, depth=4, inc_rate=2, kernel_size=(3, 3), activation='relu', normalization=None, dropout=0, elastic_deform = None, low_pass = None, high_pass = None, prwt = False, lr_decay = False, lr_schedule=None, model_net = Unet, final_test=False, monitor="val_loss"):
    '''
    DESCRIPTION: Function to load the data, load the model, fit the model and evaluate the model
    -------
    INPUTS:
    data_path:      string, directory of the folder containing the images
    imgs:           string, name of the npy file containing the images
    msks:           string, name of the npy file containing the masks
    tstimgs:        string, name of the npy file containing the test images
    tstmsks:        string, name of the npy file containing the test masks
    model_name:     string, name to identify the model
    save_path:      string, directory of the folder to which to save the results of the callbacks to
    num_folds:      int, number of folds to use for K-fold cross validation
    batch_size:     int, batch size to use in fitting the model
    learning_rate:  float, learning rate to use in compiling the model
    nr_epochs:      int, number of epochs to use in fitting the model
    verbosity:      boolean, whether to print the progress of fitting or not
    up:             boolean, True for using upsampling, False for using Transposed convolution
    start_ch:       int, the number of filters for the first convolutional layers
    depth:          int, the number of convolutional layers
    inc_rate:       number, the factor with which the number of filters is incremented per convolutional layer
    kernel_size:    int or tuple of 2 integers, the kernel size to be used in the convolution layers  
    activation:     string, which activation function to use in the convolution layers
    normalization:  function, normalization function. In case of Groupnormalization a tuple of the function and the desired group size
    dropout:        float between 0-1, the dropout rate to be used
    elastic_deform: None or tuple of 2 numbers, (alpha, sigma) with alpha being the scaling factor for the transformation and sigma being the standard deviation for the gaussian convolution
    low_pass:       None or int, giving the standard deviation used for the gaussian low-pass filter applied to the images
    high_pass:      None or int, giving the standard deviation used for the gaussian high_pass filter applied to the images
    prwt:           boolean, whether to apply a prewitt filter to the images or not
    lr_decay:       boolean, whether to use a scheduled learning rate using the schedule from 'Model.py' or not
    lr_schedule:    function, the schedule to be used for the learning rate
    model_net:      function, which model architecture to use (from Model.py: Unet or Mnet)
    final_test:     boolean, stating if the model should be optimized (k-fold = 5 folds, validation split is used, no test data is needed) or if the model performance should be tested (in 10 fold, training on all data en testing on test data)
    monitor:        string, which output of the model to monitor by the callbacks, in case final_test=True it will automaticly be set to "loss"
    -------
    OUTPUTS:
    A .h5 file per fold containing the model with its trained weights
    A .out file per fold containing the log of the training process in CSV format
    A .csv file where each row contains the DSC and train time per fold of the model, if there is already a file present the results are appended
    A .csv file where each row contains the mean and standard deviation of the DSCs and times of all folds of a model, if there is already a file present the results are appended
    '''
    
    ##### load data and optional test data #####
    images, masks = load_data(data_path, imgs, msks, low_pass=low_pass, high_pass=high_pass, prwt=prwt)
    
    if final_test: 
        print_func('Test Data:')
        test_images, test_masks = load_data(data_path, tstimgs, tstmsks, low_pass=low_pass, high_pass=high_pass, prwt=prwt)
        if model_net == Mnet: test_masks = downsample_image(test_masks, depth-1)
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
        
        train_im, train_msk, val_im, val_msk = images[train], masks[train], images[val], masks[val]  
        
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
            lr_sched = LearningRateScheduler(schedule, verbose = 1) if lr_schedule is None else LearningRateScheduler(lr_schedule, verbose = 1) 
            callbacks_list.append(lr_sched)
        
        ##### fit model #####
        arg_dict_fit = {"x": train_im, "y": train_msk, "validation_data": (val_im, val_msk), "batch_size": batch_size, "epochs": nr_epochs, "verbose": verbosity, "shuffle": True}
        
        start_time = time()
        model.fit(callbacks=callbacks_list, **arg_dict_fit)
        train_time = int(time()-start_time)
        
        if final_test: val_im, val_msk = test_images, test_masks        
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

def post_process(model_file, weights_path, data_path, imgs, msks, model_name = "", n = 10, m = False, low_pass = None, high_pass = None, threshold=0.95, disk_size=5, smooth_sigma=1, smooth_trsh=0.5):
    
    images, masks = load_data(data_path, imgs, msks, low_pass=low_pass, high_pass=high_pass)
    
    json_file = open(model_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    
    weights = os.listdir(weights_path)
    weights = [weight for weight in weights if weight.find(".h5")>-1]
    
    scores = pd.DataFrame()    
    
    for weight in weights:
        model.load_weights(os.path.join(weights_path, weight))

        predictions = model.predict(images[:n], verbose=1)
        if m: predictions = sum(predictions)/len(predictions)
        sc0 = np.mean(list(map(dice_coef_pred, masks[:n], predictions)))
        
        pp1, sc1 = [], []
        if type(threshold) != list: threshold = [threshold]
        for trsh in threshold: pp1.append(post_process_thresholding(copy.copy(predictions), trsh))
        for res in pp1: sc1.append(np.mean(list(map(dice_coef_pred, masks[:n], res))))
        max_idx1 = sc1.index(max(sc1))
        
        pp2, sc2 = [], []
        if type(disk_size) != list: disk_size = [disk_size]
        for ds in disk_size: pp2.append(post_process_openingbyreconstruction(copy.copy(pp1[max_idx1]), ds))
        for res in pp2: sc2.append(np.mean(list(map(dice_coef_pred, masks[:n], res))))
        max_idx2 = sc2.index(max(sc2))
        
        pp3, sc3 = [], []
        if type(smooth_sigma) != list: smooth_sigma = [smooth_sigma]
        if type(smooth_trsh) != list: smooth_trsh = [smooth_trsh]
        for ss in smooth_sigma: 
            for st in smooth_trsh: pp3.append(post_process_smoothingedges(copy.copy(pp2[max_idx2]), ss, st))
        for res in pp3: sc3.append(np.mean(list(map(dice_coef_pred, masks[:n], res))))        
        
        scores = scores.append([[sc0, *sc1, *sc2, *sc3]], ignore_index=True)        

    for key, value in scores.iteritems():
        m_name = model_name + '.'
        if key>0 and key<=len(threshold): m_name = m_name + "trsh" + str(threshold[key-1])
        elif key>len(threshold) and key<=len(threshold)+len(disk_size): m_name = m_name + "disk" + str(disk_size[key-1-len(threshold)])
        elif key>len(threshold)+len(disk_size):m_name + "smooth_sigma" + str(key-1-len(threshold)-len(disk_size))
        save_results(m_name, value, 0, False)
   
    
if __name__ == '__main__':
    result = train_model()
    
