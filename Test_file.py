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
from Data import save_results, print_func
from Model import Unet, Mnet
from itertools import product
import random
import pandas as pd
import os
from os.path import join as opj

from tensorflow_addons.layers import InstanceNormalization, GroupNormalization, WeightNormalization
from tensorflow.keras.layers import BatchNormalization, LayerNormalization


data_path = r"/home/jpavboxtel/data"
save_path = r"/home/jpavboxtel/code/models_Mnet"
weights_path = r"/home/jpavboxtel/code/final_weights"

if not os.path.exists(save_path): os.mkdir(save_path)

imgs = "train - imgs.npy"
msks = "train - imgs_mask.npy"

#%%
def read_results(exp_name, exp_list, res_file):
    """
    DESCRIPTION: Function to read previous produced results in order to prevent doing experiments double
    ----------
    INPUTS:
    exp_name:   string, the name to identify the experiment by
    exp_list:   list of tuples, first element of each tuple should be a string of the variable name, the second element the parameters to pass to the function given in func_list
    res_file:   string, filepath and name of file containing the previous results
    -------
    OUTPUTS:
    options:    a list of dictionaries containing the sets of parameters that have already been tested
    """
    
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

def random_search(exp_name, exp_list, func_list, nr_options=50, fail_factor = 3, prev_results = False, res_file = ""):
    """
    DESCRIPTION: Function to execute a random search among the given parameters
    ----------
    INPUTS:
    exp_name:       string, the name to identify the experiment by
    exp_list:       list of tuples, first element of each tuple should be a string of the variable name, the second element the parameters to pass to the function given in func_list
    func_list:      list of (lambda) functions, each element should contain a function corresponding the variable in exp_list at equal index
    nr_options:     int, number of options to train the model for
    fail_factor:    int, prevents an infinite while loop with failing models by limiting the maximal attempts to fail_factor*nr_options
    prev_results:   boolean, whether or not previous results are present in a csv file
    res_file:       string, filepath and name of file containing the previous results
    -------
    OUTPUTS:
    A .h5 file per fold per set of parameters containing the model with its trained weights
    A .out file per fold per set of parameters containing the log of the training process in CSV format
    A .csv file where each row contains the DSC and train time per fold per set of parameters of the model, if there is already a file present the results are appended
    A .csv file where each row contains the mean and standard deviation of the DSCs and times of all folds per set of parameters of a model, if there is already a file present the results are appended
    """
    
    options = [] if not prev_results else read_results(exp_name, exp_list, res_file)
    i,j = len(options), len(options)
    while i < nr_options and j < nr_options*fail_factor:
        i += 1
        j += 1
        
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
            print_func(f"start training model {i} out of {nr_options} \n{arg_dict}") 
            train_model(data_path, save_path = save_path, **arg_dict)
        except:
            print_func("saving model failed")
            save_results(model_name, 0, 0, elab=False)
            i -= 1

def grid_search(exp_name, exp_list, prev_results = False, res_file = ""):
    """
    DESCRIPTION: Function execute a grid search among the given parameters
    ----------
    INPUTS:
    exp_name:       string, the name to identify the experiment by
    exp_list:       list of tuples, first element of each tuple should be a string of the variable name, the second element should be a list of the options to pass for the variable
    prev_results:   boolean, whether or not previous results are present in a csv file
    res_file:       string, filepath and name of file containing the previous results
    -------
    OUTPUTS:
    A .h5 file per fold per set of parameters containing the model with its trained weights
    A .out file per fold per set of parameters containing the log of the training process in CSV format
    A .csv file where each row contains the DSC and train time per fold per set of parameters of the model, if there is already a file present the results are appended
    A .csv file where each row contains the mean and standard deviation of the DSCs and times of all folds per set of parameters of a model, if there is already a file present the results are appended
    """
    
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
                print_func(f"start training model {i} out of {nr_options} \n{arg_dict}")
                train_model(data_path, save_path = save_path, **arg_dict)
            except:
                print_func("saving model failed")
                save_results(model_name, 0, 0, elab=False)
        else: 
            print_func(f"{arg_dict} is already trained")

  

#%%
##### example 1: random search for hyperparameters on U-net#####
exp1 = [("depth", (2, 6)), ("batch_size", (0, 8)), ("learning_rate", (1, 6))]
exp1_func = [lambda a,b: random.randint(a, b), lambda a,b: 2**random.randint(a, b), lambda a,b: 10**-random.randint(a, b)]
random_search("exp1", exp1, exp1_func, nr_options = 5)#, prev_results=True, res_file="results.csv")

##### example 2: grid search for image pre-processing techniques on U-net #####
exp2 = [("depth", [4]), ("batch_size", [16]), ("learning_rate", [1e-5]), ("kernel_size", [(5, 5)]), ("start_ch", [32]), ("dropout", [0.4]), ("low_pass", [None, 0.5, 1]), ("prwt", [False, True]), ("elastic_deform", [None, (8, 0.05), (4, 0.1)])]
grid_search("exp2", exp2, prev_results=True, res_file="results.csv")

##### example 3: final test for 2 normalization techniques on M-net #####
exp3 = [("model_name", ["exp_name + str('_Mnet')"]), ("model_net", ["Mnet"]), ("batch_size", [16]), ("dropout", [0.2]), ("normalization", ["(GroupNormalization, 8)", "LayerNormalization"]), ("tstimgs", ["str('test (own) - imgs.npy')"]), ("tstmsks", ["str('test (own) - imgs_mask.npy')"]), ("final_test", [True]), ("imgs", ["str('train - imgs.npy')"]), ("msks", ["str('train - imgs_mask.npy')"])]
grid_search("test_Mnet", exp3)

##### example 4: testing for post-processing techniques #####
post_process(opj(weights_path,"Mnet_optimized/model_Mnet.json"), opj(weights_path, "Mnet_optimized/All"), data_path, imgs, msks, m=True, threshold=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], disk_size=[2, 3, 4, 5, 6, 7], low_pass = 1, high_pass = 10, smooth_sigma = [0.5, 1, 2, 3], smooth_trsh = [0.3, 0.5, 0.7, 0.9], model_name = "model_Mnet_")
