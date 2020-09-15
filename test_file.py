# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 08:42:34 2020

@author: 20164798
"""
from main import *
import random


data_path = r"/home/jpavboxtel/data"

#data_path = r"C:\Users\20164798\OneDrive - TU Eindhoven\UNI\BMT 3\BEP\data\prepared"
imgs = "train - imgs.npy"
msks = "train - imgs_mask.npy"

#%%grid search
def grid_search():
    decoder_upconv = [False, True]
    kernels = [(3, 3), (5, 5), (7, 7)]
    depths = [2, 3, 4, 5]
    start_nr_neurons = [16, 32, 64]
    batch_sizes = [1, 32, 64]
    learning_rates = [1e-2, 1e-3, 1e-5]
    
    for upconv in decoder_upconv:
        for kernel in kernels:
            for depth in depths:
                for neurons in start_nr_neurons:
                    for batch_size in batch_sizes:
                        for learning_rate in learning_rates:
                            model_name = str(int(upconv))+"."+str(kernel[0])+"."+str(depth)+"."+str(neurons)+"."+str(batch_size)+"."+str(learning_rate)
                            try: 
                                k_cross_val(data_path, lambda: train_model(data_path, model_name = model_name, train_steps=10, val_steps=5, nr_epochs=2, upconv = upconv, kernel_size = kernel, depth = depth, start_ch = neurons, batch_size = batch_size, learning_rate = learning_rate),  k=5) 
                            except: 
                                print("aborted training: " + model_name)
                                restore_dir()
#%%
def random_search(nr_options=50):
    options = []
    i = 0
    while i < nr_options:
        i += 1
        depth = random.randint(3, 6)
        batch_size = 2**random.randint(0, 8)
        learning_rate = 1*10**-random.randint(1, 6)
        model_name = str(depth)+"."+str(batch_size)+"."+str(learning_rate)
        option = (model_name, depth, batch_size, learning_rate) 
        if option in options:
            print('tis zover')
            i -= 1
            continue              
        options.append(option)

    for option in options:
        try:
            train_model(data_path, imgs, msks, model_name=option[0], depth=option[1], batch_size=option[2], learning_rate = option[3])
        except:
            saveresults(model_name, "", "" , "" , "" , "", "")
    return options

options = random_search()
#k_cross_val(data_path, lambda: train_model(data_path, model_name = 'test', upconv = True, kernel_size = (3,3), depth = 4, start_ch = 64, batch_size = 64, learning_rate = 0.01),  k=5) 
 

