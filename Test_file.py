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
from itertools import product
import random
import pandas as pd
import os


data_path = r"/home/jpavboxtel/data"
save_path = r"/home/jpavboxtel/code/models_exp2"

if not os.path.exists(save_path): os.mkdir(save_path)
#%%
#data_path = r"C:\Users\20164798\OneDrive - TU Eindhoven\UNI\BMT 3\BEP\data\prepared"
imgs = "train - imgs.npy"
msks = "train - imgs_mask.npy"


exp1 = [("depth", (2, 6)), ("batch_size", (0, 8)), ("learning_rate", (1, 6))]
exp1_func = [lambda a,b: random.randint(a, b), lambda a,b: 2**random.randint(a, b), lambda a,b: 10**-random.randint(a, b)]

exp2a = [("depth", list(range(4, 6))), ("batch_size", [2**elem for elem in list(range(0, 7))]), ("learning_rate", [10**-elem for elem in list(range(5, 8))]) ]
exp2b = [("depth", list(range(2, 4))), ("batch_size", [2**elem for elem in list(range(2, 7))]), ("learning_rate", [10**-elem for elem in list(range(4, 8))]) ]

exp3a = [("depth", 2), ("batch_size", 64), ("learning_rate", 0.0001), ("low_pass", [None, 0.5, 1]), ("high_pass", [None, 10, 20]), ("elastic_deform", [None, (8, 0.05), (4, 0.1)])]
exp3b = [("depth", 5), ("batch_size", 64), ("learning_rate", 1e-5), ("low_pass", [None, 0.5, 1]), ("high_pass", [None, 10, 20]), ("elastic_deform", [None, (8, 0.05), (4, 0.1)])]

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
        train_model(data_path, imgs, msks, save_path = save_path, **arg_dict)
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
            split_result = split_result[1].split(".", len(exp_list)-1)
            for i in range(len(split_result)):
                arg_dict[exp_list[i][0]] = float(split_result[i]) if i < len(split_result)-1 else float(split_result[i][:-1])
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
            arg_dict[exp_list[i][0]] = items[i]
            model_name += str(items[i]) + "."
        arg_dict['model_name'] = exp_name + '_' + model_name   
        if not arg_dict in options:             
            options.append(arg_dict)  
            try:
                print(f"start training model {len(options)} out of {nr_options}") 
                print(arg_dict)
                train_model(data_path, imgs, msks, save_path = save_path, **arg_dict)
            except:
                print("saving model failed")
                save_results(model_name, 0, 0, elab=False)
        else: 
            print(f"{arg_dict} is already trained")


  

#%%
#random_search('exp1', exp1, exp1_func, nr_options = 5)#, prev_results=True, res_file="results.csv")

grid_search('exp2', exp2a, prev_results=True, res_file="results.csv")
#grid_search('exp2', exp2b, prev_results=True, res_file="results.csv")

#grid_search('exp3', exp3a, prev_results=True, res_file="results.csv")


