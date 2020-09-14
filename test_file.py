# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 08:42:34 2020

@author: 20164798
"""
from main import *
import random


data_path = r"C:\Users\20164798\OneDrive - TU Eindhoven\UNI\BMT 3\BEP\data\prepared k-cross"

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
def random_search(nr_options=100):
    options = []
    for i in range(nr_options):
        upconv = bool(random.getrandbits(1))
        ker_size = random.randrange(1, 11, 2)
        kernel = (ker_size, ker_size)
        depth = random.randint(1, 5)
        neurons = 2**random.randint(1, 7)
        batch_size = 2**random.randint(0, 7)
        learning_rate = 1*10**-random.randint(1, 6)
        model_name = str(int(upconv))+"."+str(kernel[0])+"."+str(depth)+"."+str(neurons)+"."+str(batch_size)+"."+str(learning_rate)
        option = (model_name, upconv, kernel, depth, neurons, batch_size, learning_rate)               
        options.append(option)

    for option in options:
        k_cross_val(data_path, lambda: train_model(data_path, model_name = option[0], train_steps=2, val_steps=1, nr_epochs=1, upconv = option[1], kernel_size = option[2], depth = option[3], start_ch = option[4], batch_size = option[5], learning_rate = option[6]),  k=3) 
 

random_search(3)
'''
0.7.5.8.8.1e-6
       
'''
