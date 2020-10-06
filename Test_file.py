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
from Main import train_model
from Data import save_results
from Final_test import test_model
from itertools import product
import random
import pandas as pd
import os

from tensorflow_addons.layers import InstanceNormalization, GroupNormalization, WeightNormalization
from tensorflow.keras.layers import BatchNormalization, LayerNormalization


# data_path = r"/home/jpavboxtel/data"
# save_path = r"/home/jpavboxtel/code/models_exp8"

# if not os.path.exists(save_path): os.mkdir(save_path)
imgs = "train - imgs.npy"
msks = "train - imgs_mask.npy"

#%%
data_path = r"C:\Users\20164798\OneDrive - TU Eindhoven\UNI\BMT 3\BEP\data\prepared"
save_path = "results"
imgs = "test (own) - imgs.npy"
msks = "test (own) - imgs_mask.npy"

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
        arg_dict['model_name'] = exp_name + '_' + model_name
        if arg_dict in options:
            i -= 1
            continue              
        options.append(arg_dict)
        try:
            print(f"start training model {i} out of {nr_options}") 
            print(arg_dict)
            train_model(data_path, imgs, msks, save_path = save_path, **arg_dict)
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
        arg_dict['model_name'] = exp_name + '_' + model_name   
        if not arg_dict in options:             
            options.append(arg_dict)
            train_model(data_path, imgs, msks, save_path = save_path, **arg_dict)
            try:
                print(f"start training model {len(options)} out of {nr_options}") 
                print(arg_dict)

            except:
                print("saving model failed")
                save_results(model_name, 0, 0, elab=False)
        else: 
            print(f"{arg_dict} is already trained")

  

#%%
# exp1 = [("depth", (2, 6)), ("batch_size", (0, 8)), ("learning_rate", (1, 6))]
# exp1_func = [lambda a,b: random.randint(a, b), lambda a,b: 2**random.randint(a, b), lambda a,b: 10**-random.randint(a, b)]
# random_search('exp1', exp1, exp1_func, nr_options = 5)#, prev_results=True, res_file="results.csv")

# exp2a = [("depth", list(range(4, 6))), ("batch_size", [2**elem for elem in list(range(0, 7))]), ("learning_rate", [10**-elem for elem in list(range(5, 8))]) ]
# exp2b = [("depth", list(range(2, 4))), ("batch_size", [2**elem for elem in list(range(2, 7))]), ("learning_rate", [10**-elem for elem in list(range(4, 8))]) ]
# grid_search('exp2', exp2a, prev_results=True, res_file="results.csv")
# grid_search('exp2', exp2b, prev_results=True, res_file="results.csv")

# exp3a = [("depth", [2]), ("batch_size", [64]), ("learning_rate", [0.0001]), ("kernel_size", [(3, 3), (5, 5), (7, 7)]), ("start_ch", [16, 32, 64])]
# exp3b = [("depth", [4]), ("batch_size", [16]), ("learning_rate", [1e-5]), ("kernel_size", [(3, 3), (5, 5), (7, 7)]), ("start_ch", [16, 32, 64])]
# grid_search('exp3', exp3a, prev_results=True, res_file="results.csv")
# grid_search('exp3', exp3b)

# exp6a = [("depth", [2]), ("batch_size", [64]), ("learning_rate", [0.0001]), ("kernel_size", [(3, 3)]), ("start_ch", [64]), ('dropout', [0, 0.2, 0.4])]
# exp6a = [("depth", [4]), ("batch_size", [16]), ("learning_rate", [1e-5]), ("kernel_size", [(5, 5)]), ("start_ch", [32]), ('dropout', [0, 0.2, 0.4])]
# grid_search("exp6", exp6a, prev_results=True, res_file="results.csv")
# grid_search("exp6", exp6b, prev_results=True, res_file="results.csv")

test_model("all data - 4", "train - imgs.npy", "train - imgs_mask.npy", "test (own) - imgs.npy", "test (own) - imgs_mask.npy")
test_model("patient - 4", "patient - imgs.npy", "patient - imgs_mask.npy", "patient test - imgs.npy", "patient test - imgs_mask.npy"   )
test_model("True - 4", "true - imgs.npy", "true - imgs_mask.npy", "true test - imgs.npy", "true test - imgs_mask.npy")

# exp4a = [("depth", [2]), ("batch_size", [64]), ("learning_rate", [0.0001]), ("kernel_size", [3]), ("start_ch", [64]), ("low_pass", [None, 0.5, 1]), ("high_pass", [None, 10, 20]), ("elastic_deform", [None, (8, 0.05), (4, 0.1)])]
# exp4b = [("depth", [4]), ("batch_size", [16]), ("learning_rate", [1e-5]), ("kernel_size", [5]), ("start_ch", [32]), ("low_pass", [None, 0.5, 1]), ("high_pass", [None, 10, 20]), ("elastic_deform", [None, (8, 0.05), (4, 0.1)])]
# grid_search("exp4", exp4a, prev_results=True, res_file="results.csv")
# grid_search("exp4", exp4b, prev_results=True, res_file="results.csv")

# exp5a = [("depth", [2]), ("batch_size", [64]), ("learning_rate", [0.0001]), ("kernel_size", [3]), ("start_ch", [64]), ("low_pass", [None, 0.5, 1]), ("prwt", [False, True]), ("elastic_deform", [None, (8, 0.05), (4, 0.1)])]
# exp5b = [("depth", [4]), ("batch_size", [16]), ("learning_rate", [1e-5), ("kernel_size", [5]), ("start_ch", [32]), ("low_pass", [None, 0.5, 1]), ("prwt", [False, True]), ("elastic_deform", [None, (8, 0.05), (4, 0.1)])]
# grid_search("exp5", exp5a, prev_results=True, res_file="results.csv")
# grid_search("exp5", exp5b, prev_results=True, res_file="results.csv")

# exp7a = [("depth", [2]), ("batch_size", [64]), ("learning_rate", [0.0001]), ("kernel_size", [3]), ("start_ch", [64]), ("low_pass", [None, 0.5, 1]), ("prwt", [False, True]), ("elastic_deform", [None, (8, 0.05), (4, 0.1)])]
# exp7b = [("depth", [2]), ("batch_size", [64]), ("learning_rate", [0.0001]), ("kernel_size", [3]), ("start_ch", [64]), ("low_pass", [None, 0.5, 1]), ("high_pass", [None, 10, 20]), ("elastic_deform", [None, (8, 0.05), (4, 0.1)])]
# grid_search("exp7", exp7a, prev_results=True, res_file="results.csv")
# grid_search("exp7", exp7b, prev_results=True, res_file="results.csv")

exp8a = [("depth", [4]), ("batch_size", [16]), ("learning_rate", [1e-5]), ("kernel_size", [(5, 5)]), ("start_ch", [32]), ("dropout", [0.4]), ("low_pass", [None, 0.5, 1]), ("prwt", [False, True]), ("elastic_deform", [None, (8, 0.05), (4, 0.1)])]
exp8b = [("depth", [4]), ("batch_size", [16]), ("learning_rate", [1e-5]), ("kernel_size", [(5, 5)]), ("start_ch", [32]), ("dropout", [0.4]), ("low_pass", [None, 0.5, 1]), ("high_pass", [None, 10, 20]), ("elastic_deform", [None, (8, 0.05), (4, 0.1)])]
grid_search("exp8", exp8a, prev_results=True, res_file="results.csv")
grid_search("exp8", exp8b, prev_results=True, res_file="results.csv")

#%%
exp9 = [("depth", [4]), ("batch_size", [16]), ("learning_rate", [1e-5]), ("kernel_size", [(5, 5)]), ("start_ch", [32]), ("dropout", [0.4]), ("low_pass", [None]), ("prwt", [False]), ("elastic_deform", [None]), ("normalization", ["(GroupNormalization, 8)", "(GroupNormalization, 16)", "LayerNormalization", "InstanceNormalization", "BatchNormalization"])]
grid_search("exp9", exp9, prev_results=True, res_file="results.csv")


#test_model("all data - elastic deform (4, 0.1)", "train - imgs.npy", "train - imgs_mask.npy", "test (own) - imgs.npy", "test (own) - imgs_mask.npy", elastic_deform = (4, 0.1))


