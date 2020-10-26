# Deep learning Segmentation of Brachial plexus nerves in ultrasound images

This repository contains the files which can be used to train deep learning models with the U-net and M-net structure for segmentation of the brachial plexus nerve in ultrasound images.

<b>How to use the code: </b>

first download all the data from: https://www.kaggle.com/c/ultrasound-nerve-segmentation/data 

Then use the function 'create_data' from the file data.py to create npy files from the original data

Next, run either Main.py to fit a specific model to the data sets. 
The other option is to use Test_file.py to train a number of parameter sets. 
Explanation of the functions and their variables are given in the codes. 

<b>Short description of the files and their functions:</b>

<b>Data.py</b> : The file to create the data and including all functions required in pre- and post-processing the data. Also a number of helper functions are included.
Functions included: 
<ul>
  <li>Create_data: function to create .npy files from the original dataset</li>
  <li>Elastic_deformation: function to perform elastic deformation on an image and its corresponding mask</li>
  <li>Image_transformation: function to perform pre-processing on the images and their corresponding masks
  Including (optional): <ul>
    <li> Low-pass filter </li>
    <li> High-pass filter </li>
    <li> Prewitt filter </li>
    <li> Elastic deformation </li>
    </ul>
  <li> reshape_imgs: function to reshape the input images to the desired shape </li>
  <li> load_data: function to load the data and prepare it for fitting or testing the models
  Includes: <ul>
    <li> loading the data from npy file </li>
    <li> pre-processing data with image_transformation function </li>
    <li> reshaping data with reshape_imgs function </li>
    <li> normalizing data by mean centering with a std of 1 </li>
    </ul>
  <li> save_results: help function to save results to a csv file </li>
  <li> downsample_image: help function to downsample masks for M-net use </li>
  <li> print_func: help function to make clear print statements to the console </li>
</ul>

<b>Model.py</b>: the file containing all functions necessary to compile the U-net and M-net models and the metrics used.
Functions included: 
<ul>
  <li> dice_coef_pred: DSC metric used in evaluating the predictions of the model </li>
  <li> precision_pred: Precision metric used in evaluating the predictions of the model </li>
  <li> dice_coef: DSC metric used in fitting and evaluating the model in the keras backend </li>
  <li> dice_coef_loss: DSC loss function used in fitting and evaluating the model in the keras backend </li>
  <li> schedule: schedule function used for learning decay </li>
  <li> conv_block: Function containing the convolutional block used to build the U-net and M-net structures </li>
  <li> level_block_unet: Recursive function used to build the U-net architecture </li>
  <li> Unet: function used to build and compile the U-net architecture </li>
  <li> level_block_mnet: Recursive function used to build the M-net architecture </li>
  <li> Mnet: function used to build and compile the M-net architecture </li>
  <li> eval_Mnet: function used to evaluate the M-net model performance </li>
</ul>

<b>Main.py</b>: the file containing the function to run a full fitting and evaluation run of a specific model, using k-fold cross validation.
Functions included: 
<ul>
  <li> train_model: trains the model according to given parameters using k-fold cross validation </li>
</ul>

<b>Test_file.py</b>: the file used to run the full fitting and evaluation process for multiple variations of a model.
Functions included:
<ul>
  <li> read_results: function to read previous results from a .csv file in order to prevent executing experiments double</li>
  <li> random_search: function to perform a random search among the given parameters </li>
  <li> grid_search: function to perform a grid search among the given parameters </li>
</ul>



