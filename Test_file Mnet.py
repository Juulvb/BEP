# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 12:38:07 2020

@author: 20164798
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 08:42:34 2020

@author: 20164798
"""
from Main import train_model, post_process
from Data import save_results
from Model import Unet, Mnet
from Final_test import test_model
from itertools import product
import random
import pandas as pd
import os

from tensorflow_addons.layers import InstanceNormalization, GroupNormalization, WeightNormalization
from tensorflow.keras.layers import BatchNormalization, LayerNormalization


data_path = r"/home/jpavboxtel/data"
save_path = r"/home/jpavboxtel/code/models_Mnet"

imgs = "train - imgs.npy"
msks = "train - imgs_mask.npy"

#%%

if not os.path.exists(save_path): os.mkdir(save_path)
# data_path = r"C:\Users\20164798\OneDrive - TU Eindhoven\UNI\BMT 3\BEP\data\prepared"
# save_path = "results"
# imgs = "test (own) - imgs.npy"
# msks = "test (own) - imgs_mask.npy"

#%%
def random_search(exp_name, exp_list, func_list, nr_options=50, prev_results = False, res_file = ""):
    options = [] if not prev_results else read_results(exp_name, exp_list, res_file)
    i = len(options)
    while i < nr_options:
        i += 1
        
        arg_dict = {}
        model_name = ""
        for j in range(len(exp_list)):
            var = exp_list[j]
            val = func_list[j](*var[1])
            model_name += str(val) + "."
            arg_dict[var[0]] = val
        if 'model_name' not in arg_dict: arg_dict['model_name'] = exp_name + '_' + model_name
        if arg_dict in options:
            i -= 1
            continue              
        options.append(arg_dict)
        if "imgs" not in arg_dict: arg_dict["imgs"] = imgs
        if "msks" not in arg_dict: arg_dict["msks"] = msks
        try:
            print(f"start training model {i} out of {nr_options}") 
            print(arg_dict)
            train_model(data_path, save_path = save_path, **arg_dict)
        except:
            print("saving model failed")
            save_results(model_name, 0, 0, elab=False)
            i -= 1
    
def read_results(exp_name, exp_list, res_file):
    results = pd.read_csv(res_file, sep=';')
    options = []
    for result in results['Model_name']:
        arg_dict = {}
        split_result = result.split("_")
        if exp_name == split_result[0]:
            split_result = split_result[1].split(".")
            split_result.remove('')
            i = 0
            while i < len(split_result):
                if split_result[i][-1] == '0' and i < len(split_result)-1 and split_result[i+1][0].isdigit(): 
                    split_result[i:i+2] = ['.'.join(split_result[i:i+2])]
                    i -= 1
                arg_dict[exp_list[i][0]] = eval(str(split_result[i]))
                i += 1    
            arg_dict['model_name'] = result
            options.append(arg_dict)
    return options


def grid_search(exp_name, exp_list, prev_results = False, res_file = ""):
    options = [] if not prev_results else read_results(exp_name, exp_list, res_file)
    var = [var[1] for var in exp_list]
    nr_options = len(list(product(*var)))
    for items in product(*var):
        arg_dict = {}
        model_name = ""
        for i in range(len(items)):
            arg_dict[exp_list[i][0]] = eval(str(items[i]))
            model_name += str(items[i]) + "."
        if 'model_name' not in arg_dict: arg_dict['model_name'] = exp_name + '_' + model_name   
        if not arg_dict in options:             
            options.append(arg_dict)
            if "imgs" not in arg_dict: arg_dict["imgs"] = imgs
            if "msks" not in arg_dict: arg_dict["msks"] = msks
            try:
                print(f"start training model {len(options)} out of {nr_options}") 
                print(arg_dict)
                train_model(data_path, save_path = save_path, **arg_dict)
            except:
                print("saving model failed")
                save_results(model_name, 0, 0, elab=False)
        else: 
            print(f"{arg_dict} is already trained")


#%%
def schedule1(epoch, lr):
    if epoch in [10, 20, 30, 40]: lr = lr/10
    return lr
def schedule2(epoch, lr):
    if epoch in [10, 15, 20, 25]: lr = lr/10
    return lr
def schedule3(epoch, lr):
    if epoch in [10, 12, 14, 16]: lr = lr/10
    return lr
def schedule4(epoch, lr):
    if epoch in [10, 11, 12, 13]: lr = lr/10
    return lr

#%%
# exp11a = [("model_name", ["exp_name + str('_Unet')"]), ("depth", [4]), ("batch_size", [16]), ("learning_rate", [1e-4]), ("kernel_size", [(5, 5)]), ("start_ch", [32]), ("dropout", [0.4]), ("low_pass", [None]), ("high_pass", [20]), ("elastic_deform", [(8, 0.05)]), ("normalization", ["(GroupNormalization, 8)"]), ("lr_decay", [True]), ("tstimgs", ["str('test (own) - imgs.npy')"]), ("tstmsks", ["str('test (own) - imgs_mask.npy')"]), ("final_test", [True]), ("imgs", ["str('train - imgs.npy')"]), ("msks", ["str('train - imgs_mask.npy')"])]
# grid_search("test - all data", exp11a, prev_results=True, res_file="results.csv")
# exp11b = [("model_name", ["exp_name + str('_Unet')"]), ("depth", [4]), ("batch_size", [16]), ("learning_rate", [1e-4]), ("kernel_size", [(5, 5)]), ("start_ch", [32]), ("dropout", [0.4]), ("low_pass", [None]), ("high_pass", [20]), ("elastic_deform", [(8, 0.05)]), ("normalization", ["(GroupNormalization, 8)"]), ("lr_decay", [True]), ("tstimgs", ["str('patient test - imgs.npy')"]), ("tstmsks", ["str('patient test - imgs_mask.npy')"]), ("final_test", [True]), ("imgs", ["str('patient - imgs.npy')"]), ("msks", ["str('patient - imgs_mask.npy')"])]
# grid_search("test - patient", exp11a, prev_results=True, res_file="results.csv")
# exp11c = [("model_name", ["exp_name + str('_Unet')"]), ("depth", [4]), ("batch_size", [16]), ("learning_rate", [1e-4]), ("kernel_size", [(5, 5)]), ("start_ch", [32]), ("dropout", [0.4]), ("low_pass", [None]), ("high_pass", [20]), ("elastic_deform", [(8, 0.05)]), ("normalization", ["(GroupNormalization, 8)"]), ("lr_decay", [True]), ("tstimgs", ["str('True test - imgs.npy')"]), ("tstmsks", ["str('True test - imgs_mask.npy')"]), ("final_test", [True]), ("imgs", ["str('True - imgs.npy')"]), ("msks", ["str('True - imgs_mask.npy')"])]
# grid_search("test - true", exp11a, prev_results=True, res_file="results.csv")

# exp1a = [("model_name", ["exp_name + str('_Mnet')"]), ("model_net", ["Mnet"]), ("batch_size", [16]), ("dropout", [0.2]), ("tstimgs", ["str('test (own) - imgs.npy')"]), ("tstmsks", ["str('test (own) - imgs_mask.npy')"]), ("final_test", [True]), ("imgs", ["str('train - imgs.npy')"]), ("msks", ["str('train - imgs_mask.npy')"])]
# grid_search("test_All", exp1a)
# exp1b = [("model_name", ["exp_name + str('_Mnet')"]), ("model_net", ["Mnet"]), ("batch_size", [16]), ("dropout", [0.2]), ("tstimgs", ["str('patient test - imgs.npy')"]), ("tstmsks", ["str('patient test - imgs_mask.npy')"]), ("final_test", [True]), ("imgs", ["str('patient - imgs.npy')"]), ("msks", ["str('patient - imgs_mask.npy')"])]
# grid_search("test_patient", exp1b, prev_results=True, res_file="results.csv")
# exp1c = [("model_name", ["exp_name + str('_Mnet')"]), ("model_net", ["Mnet"]), ("batch_size", [16]), ("dropout", [0.2]), ("tstimgs", ["str('True test - imgs.npy')"]), ("tstmsks", ["str('True test - imgs_mask.npy')"]), ("final_test", [True]), ("imgs", ["str('True - imgs.npy')"]), ("msks", ["str('True - imgs_mask.npy')"])]
# grid_search("test_true", exp1c, prev_results=True, res_file="results.csv")

# exp2a = [("model_net", ["Mnet"]), ("batch_size", [16]), ("dropout", [0.2]), ("low_pass", [None, 0.5, 1]), ("prwt", [False, True]), ("elastic_deform", [None, (8, 0.05), (4, 0.1)])]
# exp2b = [("model_net", ["Mnet"]), ("batch_size", [16]), ("dropout", [0.2]), ("low_pass", [None, 0.5, 1]), ("high_pass", [None, 10, 20]), ("elastic_deform", [None, (8, 0.05), (4, 0.1)])]
# grid_search("exp2", exp2a, prev_results=True, res_file="results.csv")
# grid_search("exp2", exp2b, prev_results=True, res_file="results.csv")

# exp12a = [("model_name", ["exp_name + str('_Unet_orig')"]), ("depth", [4]), ("batch_size", [64]), ("learning_rate", [1e-2]), ("kernel_size", [(3, 3)]), ("start_ch", [64]), ("dropout", [0.2]), ("tstimgs", ["str('test (own) - imgs.npy')"]), ("tstmsks", ["str('test (own) - imgs_mask.npy')"]), ("final_test", [True]), ("imgs", ["str('train - imgs.npy')"]), ("msks", ["str('train - imgs_mask.npy')"])]
# grid_search("test_All", exp12a, prev_results=True, res_file="results.csv")
# exp12b = [("model_name", ["exp_name + str('_Unet_orig')"]), ("depth", [4]), ("batch_size", [64]), ("learning_rate", [1e-2]), ("kernel_size", [(3, 3)]), ("start_ch", [64]), ("dropout", [0.2]), ("tstimgs", ["str('patient test - imgs.npy')"]),  ("tstmsks", ["str('patient test - imgs_mask.npy')"]), ("final_test", [True]), ("imgs", ["str('patient - imgs.npy')"]), ("msks", ["str('patient - imgs_mask.npy')"])]
# grid_search("test_patient", exp12b, prev_results=True, res_file="results.csv")
# exp12c = [("model_name", ["exp_name + str('_Unet_orig')"]), ("depth", [4]), ("batch_size", [64]), ("learning_rate", [1e-2]), ("kernel_size", [(3, 3)]), ("start_ch", [64]), ("dropout", [0.2]), ("tstimgs", ["str('True test - imgs.npy')"]), ("tstmsks", ["str('True test - imgs_mask.npy')"]), ("final_test", [True]), ("imgs", ["str('True - imgs.npy')"]), ("msks", ["str('True - imgs_mask.npy')"])]
# grid_search("test_true", exp12c, prev_results=True, res_file="results.csv")

# exp3 = [("model_net", ["Mnet"]), ("batch_size", [16]), ("dropout", [0.2]), ("low_pass", [0.5]), ("prwt", [False]), ("elastic_deform", [(8, 0.05), None]), ("normalization", ["(GroupNormalization, 8)", "(GroupNormalization, 16)", "LayerNormalization", "InstanceNormalization", "BatchNormalization"])]
# grid_search("exp2", exp3, prev_results=True, res_file="results.csv")

# exp4 = [("model_net", ["Mnet"]), ("batch_size", [16]), ("learning_rate", [1e-5]), ("dropout", [0.2]), ("low_pass", [0.5]), ("prwt", [False]), ("elastic_deform", [None]), ("lr_decay", [True])]
# grid_search("exp4", exp4, prev_results=True, res_file="results.csv")

# exp13a = [("depth", [4]), ("batch_size", [64]), ("learning_rate", [1e-2]), ("kernel_size", [(3, 3)]), ("start_ch", [64]), ("dropout", [0.2]), ("low_pass", [None, 0.5, 1]), ("prwt", [False, True]), ("elastic_deform", [None, (8, 0.05), (4, 0.1)])]
# exp13b = [("depth", [4]), ("batch_size", [64]), ("learning_rate", [1e-2]), ("kernel_size", [(3, 3)]), ("start_ch", [64]), ("dropout", [0.2]), ("low_pass", [None, 0.5, 1]), ("high_pass", [None, 10, 20]), ("elastic_deform", [None, (8, 0.05), (4, 0.1)])]
# grid_search("exp13", exp13a, prev_results=True, res_file="results.csv")
# grid_search("exp13", exp13b, prev_results=True, res_file="results.csv")

# exp5 = [("model_net", ["Mnet"]), ("batch_size", [16]), ("learning_rate", [1e-5]), ("dropout", [0.2]), ("low_pass", [0.5]), ("prwt", [False]), ("elastic_deform", [None]), ("lr_decay", [True])]
# grid_search("exp5", exp5, prev_results=True, res_file="results.csv")

# exp14a = [("depth", [4]), ("batch_size", [64]), ("learning_rate", [1e-2]), ("kernel_size", [(3, 3)]), ("start_ch", [64]), ("dropout", [0.2]), ("low_pass", [None, 0.5, 1]), ("prwt", [False, True]), ("elastic_deform", [None, (8, 0.05), (4, 0.1)])]
# exp14b = [("depth", [4]), ("batch_size", [64]), ("learning_rate", [1e-2]), ("kernel_size", [(3, 3)]), ("start_ch", [64]), ("dropout", [0.2]), ("low_pass", [None, 0.5, 1]), ("high_pass", [None, 10, 20]), ("elastic_deform", [None, (8, 0.05), (4, 0.1)])]
# grid_search("exp14", exp14a, prev_results=True, res_file="results.csv")
# grid_search("exp14", exp14b, prev_results=True, res_file="results.csv")

# exp15a = [("depth", [4]), ("batch_size", [16]), ("learning_rate", [1e-5]), ("kernel_size", [(5, 5)]), ("start_ch", [32]), ("dropout", [0.4]), ("low_pass", [None, 0.5, 1]), ("prwt", [False, True]), ("elastic_deform", [None, (8, 0.05), (4, 0.1)])]
# exp15b = [("depth", [4]), ("batch_size", [16]), ("learning_rate", [1e-5]), ("kernel_size", [(5, 5)]), ("start_ch", [32]), ("dropout", [0.4]), ("low_pass", [None, 0.5, 1]), ("high_pass", [None, 10, 20]), ("elastic_deform", [None, (8, 0.05), (4, 0.1)])]
# grid_search("exp15", exp15a, prev_results=True, res_file="results.csv")
# grid_search("exp15", exp15b, prev_results=True, res_file="results.csv")

# exp16a = [("model_net", ["Mnet"]), ("batch_size", [16]), ("dropout", [0.2]), ("low_pass", [None, 0.5, 1]), ("prwt", [False, True]), ("elastic_deform", [None, (8, 0.05), (4, 0.1)])]
# exp16b = [("model_net", ["Mnet"]), ("batch_size", [16]), ("dropout", [0.2]), ("low_pass", [None, 0.5, 1]), ("high_pass", [None, 10, 20]), ("elastic_deform", [None, (8, 0.05), (4, 0.1)])]
# grid_search("exp16", exp16a, prev_results=True, res_file="results.csv")
# grid_search("exp16", exp16b, prev_results=True, res_file="results.csv")

# exp17a = [("depth", [4]), ("batch_size", [16]), ("learning_rate", [1e-5]), ("kernel_size", [(5, 5)]), ("start_ch", [32]), ("dropout", [0.4]), ("low_pass", [None, 0.5, 1]), ("prwt", [False, True]), ("elastic_deform", [None, (8, 0.05), (4, 0.1)])]
# exp17b = [("depth", [4]), ("batch_size", [16]), ("learning_rate", [1e-5]), ("kernel_size", [(5, 5)]), ("start_ch", [32]), ("dropout", [0.4]), ("low_pass", [None, 0.5, 1]), ("high_pass", [None, 10, 20]), ("elastic_deform", [None, (8, 0.05), (4, 0.1)])]
# grid_search("exp17", exp17a, prev_results=True, res_file="results.csv")
# grid_search("exp17", exp17b, prev_results=True, res_file="results.csv")

# exp18 = [("model_net", ["Mnet"]), ("batch_size", [16]), ("dropout", [0.2]), ("low_pass", [1]), ("high_pass", [10]), ("elastic_deform", [(8, 0.05)]), ("normalization", ["(GroupNormalization, 8)", "(GroupNormalization, 16)", "LayerNormalization", "InstanceNormalization", "BatchNormalization"])]
# grid_search("exp18", exp18, prev_results=True, res_file="results.csv")

# exp19 = [("model_net", ["Mnet"]), ("batch_size", [16]), ("dropout", [0.2]), ("low_pass", [1]), ("high_pass", [10]), ("elastic_deform", [(8, 0.05)]), ("learning_rate", [1e-4, 1e-5]), ("lr_decay", [True]), ("lr_schedule", ["schedule1", "schedule2", "schedule3", "schedule4"])]
# grid_search("exp19", exp19, prev_results=True, res_file="results.csv")

# final_Mneta = [("model_name", ["exp_name + str('_Mnet')"]), ("model_net", ["Mnet"]), ("batch_size", [16]), ("dropout", [0.2]), ("low_pass", [1]), ("high_pass", [10]), ("elastic_deform", [(8, 0.05)]), ("learning_rate", [1e-4]), ("lr_decay", [True]), ("lr_schedule", ["schedule1"]), ("tstimgs", ["str('test (own) - imgs.npy')"]), ("tstmsks", ["str('test (own) - imgs_mask.npy')"]), ("final_test", [True]), ("imgs", ["str('train - imgs.npy')"]), ("msks", ["str('train - imgs_mask.npy')"])]
# grid_search("test_All_improved", final_Mneta)
# final_Mnetb = [("model_name", ["exp_name + str('_Mnet')"]), ("model_net", ["Mnet"]), ("batch_size", [16]), ("dropout", [0.2]), ("low_pass", [1]), ("high_pass", [10]), ("elastic_deform", [(8, 0.05)]), ("learning_rate", [1e-4]), ("lr_decay", [True]), ("lr_schedule", ["schedule1"]), ("tstimgs", ["str('patient test - imgs.npy')"]), ("tstmsks", ["str('patient test - imgs_mask.npy')"]), ("final_test", [True]), ("imgs", ["str('patient - imgs.npy')"]), ("msks", ["str('patient - imgs_mask.npy')"])]
# grid_search("test_patient_improved", final_Mnetb, prev_results=True, res_file="results.csv")
# final_Mnetc = [("model_name", ["exp_name + str('_Mnet')"]), ("model_net", ["Mnet"]), ("batch_size", [16]), ("dropout", [0.2]), ("low_pass", [1]), ("high_pass", [10]), ("elastic_deform", [(8, 0.05)]), ("learning_rate", [1e-4]), ("lr_decay", [True]), ("lr_schedule", ["schedule1"]), ("tstimgs", ["str('True test - imgs.npy')"]), ("tstmsks", ["str('True test - imgs_mask.npy')"]), ("final_test", [True]), ("imgs", ["str('True - imgs.npy')"]), ("msks", ["str('True - imgs_mask.npy')"])]
# grid_search("test_true_improved", final_Mnetc, prev_results=True, res_file="results.csv")

# exp20 = [("depth", [4]), ("batch_size", [64]), ("learning_rate", [1e-2]), ("kernel_size", [(3, 3)]), ("start_ch", [64]), ("dropout", [0.2]), ("low_pass", [1]), ("elastic_deform", [(8, 0.05)]), ("normalization", ["(GroupNormalization, 8)", "(GroupNormalization, 16)", "LayerNormalization", "InstanceNormalization", "BatchNormalization"])]
# grid_search("exp20", exp20, prev_results=True, res_file="results.csv")

# exp21 = [("depth", [4]), ("batch_size", [64]), ("learning_rate", [1e-1, 1e-2]), ("kernel_size", [(3, 3)]), ("start_ch", [64]), ("dropout", [0.2]), ("low_pass", [1]), ("elastic_deform", [(8, 0.05)]), ("normalization", ["LayerNormalization"]), ("lr_decay", [True]), ("lr_schedule", ["schedule1", "schedule2", "schedule3", "schedule4"]) ]
# grid_search("exp21", exp21, prev_results=True, res_file="results.csv")

# final_Uneta = [("model_name", ["exp_name + str('_Unet_orig')"]), ("nr_epochs", [15]), ("depth", [4]), ("batch_size", [64]), ("learning_rate", [1e-2]), ("kernel_size", [(3, 3)]), ("start_ch", [64]), ("dropout", [0.2]), ("low_pass", [1]), ("elastic_deform", [(8, 0.05)]), ("normalization", ["LayerNormalization"]), ("tstimgs", ["str('test (own) - imgs.npy')"]), ("tstmsks", ["str('test (own) - imgs_mask.npy')"]), ("final_test", [True]), ("imgs", ["str('train - imgs.npy')"]), ("msks", ["str('train - imgs_mask.npy')"])]
# grid_search("test_All_improved", final_Uneta, prev_results=True, res_file="results.csv")
# final_Unetb = [("model_name", ["exp_name + str('_Unet_orig')"]), ("nr_epochs", [15]), ("depth", [4]), ("batch_size", [64]), ("learning_rate", [1e-2]), ("kernel_size", [(3, 3)]), ("start_ch", [64]), ("dropout", [0.2]), ("low_pass", [1]), ("elastic_deform", [(8, 0.05)]), ("normalization", ["LayerNormalization"]), ("tstimgs", ["str('patient test - imgs.npy')"]),  ("tstmsks", ["str('patient test - imgs_mask.npy')"]), ("final_test", [True]), ("imgs", ["str('patient - imgs.npy')"]), ("msks", ["str('patient - imgs_mask.npy')"])]
# grid_search("test_patient_improved", final_Unetb, prev_results=True, res_file="results.csv")
# final_Unetc = [("model_name", ["exp_name + str('_Unet_orig')"]), ("nr_epochs", [15]), ("depth", [4]), ("batch_size", [64]), ("learning_rate", [1e-2]), ("kernel_size", [(3, 3)]), ("start_ch", [64]), ("dropout", [0.2]), ("low_pass", [1]), ("elastic_deform", [(8, 0.05)]), ("normalization", ["LayerNormalization"]), ("tstimgs", ["str('True test - imgs.npy')"]), ("tstmsks", ["str('True test - imgs_mask.npy')"]), ("final_test", [True]), ("imgs", ["str('True - imgs.npy')"]), ("msks", ["str('True - imgs_mask.npy')"])]
# grid_search("test_true_improved", final_Unetc, prev_results=True, res_file="results.csv")

# exp22 = [("depth", [4]), ("batch_size", [16]), ("learning_rate", [1e-5]), ("kernel_size", [(5, 5)]), ("start_ch", [32]), ("dropout", [0.4]), ("elastic_deform", [(4, 0.1)]), ("normalization", ["(GroupNormalization, 8)", "(GroupNormalization, 16)", "LayerNormalization", "InstanceNormalization", "BatchNormalization"])]
# grid_search("exp22", exp22, prev_results=True, res_file="results.csv")

exp23 = [("depth", [4]), ("batch_size", [16]), ("learning_rate", [1e-4, 1e-5]), ("kernel_size", [(5, 5)]), ("start_ch", [32]), ("dropout", [0.4]), ("low_pass", [None]), ("high_pass", [None]), ("elastic_deform", [(4, 0.1)]), ("normalization", ["(GroupNormalization, 8)"]), ("lr_decay", [True]), ("lr_schedule", ["schedule1", "schedule2", "schedule3", "schedule4"])]
grid_search("exp23", exp23, prev_results=True, res_file="results.csv")
      