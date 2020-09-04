

from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Activation, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import backend as K
from data import load_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, auc, roc_curve, roc_auc_score, recall_score, precision_score, matthews_corrcoef, confusion_matrix, average_precision_score
import scipy.io as sio

K.set_image_data_format('channels_last')  # TF dimension ordering in this code


########################## variables definition ####################################
img_rows = 96
img_cols = 96

input_shape = (img_rows,img_cols,1)
smooth = 1.
nb_classes = 2 ##### if there is BP in the image,1. else 0
runNum = 5  


fpr = dict()
tpr = dict()
roc_auc = dict()

savePath = 'path to save results'

###########################################
def make_labels(targets):
    """
    DESCRIPTION
    -----------
    This function creates the labels for images based on the existance of BP:
    if the sum of values in the target is zeros returns '0' else returns '1'
    Parameters
    ----------
    targets : TYPE
        DESCRIPTION.

    Returns
    -------
    binary_targets : TYPE
        DESCRIPTION.

    """

###########################################

def split_data(imgs, targets, test_size = 0.2):
    """
    DESCRIPTION:
    -----------
    split the data to train and test
    ***
        this function needs to change to also consider the validation data:
            change it to the k-fold cross validation 
        
    Parameters
    ----------
    imgs : TYPE
        DESCRIPTION.
    targets : TYPE
        DESCRIPTION.
    test_size : TYPE, optional
        DESCRIPTION. The default is 0.2.

    Returns
    -------
    X_train : TYPE
        DESCRIPTION.
    X_test : TYPE
        DESCRIPTION.
    y_train : TYPE
        DESCRIPTION.
    y_test : TYPE
        DESCRIPTION.

    """

#############################################################################################################
def normalization(train_data, train_labels, test_data, test_labels):
    """
    

    Parameters
    ----------
    train_data : TYPE
        DESCRIPTION.
    train_labels : TYPE
        DESCRIPTION.
    test_data : TYPE
        DESCRIPTION.
    test_labels : TYPE
        DESCRIPTION.

    Returns
    -------
    train_data : TYPE
        DESCRIPTION.
    train_labels : TYPE
        DESCRIPTION.
    test_data : TYPE
        DESCRIPTION.
    test_labels : TYPE
        DESCRIPTION.

    """
    
    ###### data normalization ##########
    
    train_data = train_data.astype('float32')
    mean = np.mean(train_data)  # mean for data centering
    std = np.std(train_data)  # std for data normalization

    train_data -= mean
    train_data /= std    
    
    test_data = test_data.astype('float32')
    test_data -= mean
    test_data /= std
    
    return train_data, train_labels, test_data, test_labels
    
########################################################
def preprocess(imgs):
    """
    DESCRIPTION: preprocess data via resizing images 

    Parameters
    ----------
    imgs : TYPE
        DESCRIPTION.

    Returns
    -------
    imgs_p : TYPE
        DESCRIPTION.

    """
    imgs_p = np.ndarray((imgs.shape[0], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i] = resize(imgs[i], (img_cols, img_rows), preserve_range=True)

    imgs_p = imgs_p[..., np.newaxis]
    return imgs_p

################################################################################
def train_and_predict():
    """
    

    Returns
    -------
    None.

    """
    print('-'*30)
    print('Loading and preprocessing data and spliting it to test and train...')
    print('-'*30)
    
    data, labels = load_data()
    labels = make_labels(labels)
    data = preprocess(data)
    
    X_train, X_test, y_train, y_test = split_data(data, labels)

    print('-'*30)
    print('normalize data...')
    print('-'*30)
    trainingFeatures, trainingLabels, testFeatures, testLabels = normalization(X_train, y_train, X_test, y_test)
    
    print('-'*30)
    print('Make labels categorical...')
    print('-'*30)
    
    trainingLabels = keras.utils.to_categorical(trainingLabels, nb_classes)

    # imgs_test, imgs_id_test = load_test_data()
    # imgs_test = preprocess(imgs_test)
    

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    
    
    model = keras.Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape = input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(8, kernel_initializer='glorot_uniform', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    # model.summary()

    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-5), metrics=['accuracy'])
    model.fit(trainingFeatures, trainingLabels, batch_size=32, nb_epoch=20, verbose=2, shuffle=True,
              validation_split=0.2) #class_weight = class_w

    predicted_testLabels = model.predict_classes(testFeatures,verbose = 0)
    soft_targets_test = model.predict(testFeatures,verbose = 0)    
    ############## model prediction and evaluation ##############
    
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    
    precisionNet = precision_score(testLabels, predicted_testLabels)
    recallNet = recall_score(testLabels, predicted_testLabels)
    accNet = accuracy_score(testLabels, predicted_testLabels)
    f1Net = f1_score(testLabels, predicted_testLabels)
    print('precisionNet: %.4f' % (precisionNet))
    print('recallNet : %.4f'%(recallNet))            

    AUCNet = roc_auc_score(testLabels, soft_targets_test[:,1])
    print('f1Net: %.4f' % (f1Net))   
    print('AUCNet : %.4f'%(AUCNet))            
    sio.savemat(savePath + 'CNN_Results' + '.mat', {'precisionNet': precisionNet,'AUCNet':AUCNet,
                                                            'recallNet': recallNet, 'f1Net': f1Net
                                                            ,'accNet': accNet})

if __name__ == '__main__':   
    train_and_predict()
