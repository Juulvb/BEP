from __future__ import print_function

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, UpSampling2D, AveragePooling2D, Average
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K

import numpy as np

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

######################### define variables ##################################
img_rows = 96
img_cols = 96

smooth = 1.

###########################################  

def dice_coef_pred(y_true, y_pred):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def dice_coef(y_true, y_pred):
    """
    
    Parameters
    ----------
    y_true : TYPE
        the real labels.
    y_pred : TYPE
        the predicted labels via network.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
############################################
    
def dice_coef_loss(y_true, y_pred):
    """
    

    Parameters
    ----------
    y_true : TYPE
        DESCRIPTION.
    y_pred : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        returns the dice coef loss.

    """
    return -dice_coef(y_true, y_pred)



def conv_block(m, dim, shape, acti, norm, do=0):
    n = Conv2D(dim, shape, activation=acti, padding='same')(m)
    n = norm()(n) if norm and type(norm) != tuple else norm[0](norm[1])(n) if type(norm) == tuple else n
    n = Dropout(do)(n) if do else n
    n = Conv2D(dim, shape, activation=acti, padding='same')(n)
    n = norm()(n) if norm and type(norm) != tuple else norm[0](norm[1])(n) if type(norm) == tuple else n
    return n

def level_block_unet(m, dim, shape, depth, inc, acti, norm, do, up):
    if depth > 0:
        n = conv_block(m, dim, shape, acti, norm, do)
        m = MaxPooling2D()(n)
        m = level_block_unet(m, int(inc*dim), shape, depth-1, inc, acti, norm, do, up)
        
        if up:
            m = UpSampling2D()(m)
            m = Conv2D(dim, 2, activation=acti, padding='same')(m)
        else:
            m = Conv2DTranspose(dim, shape, strides=(2, 2), padding='same')(m)
        
        n = Concatenate()([n, m])
        m = conv_block(n, dim, shape, acti, norm)   
    else:
        m = conv_block(m, dim, shape, acti, norm, do)
    
    return m

def level_block_mnet(i, m, dim, shape, depth, inc, acti, norm, do, up, out = []):
    if depth > 0: 
        i = AveragePooling2D()(i)
        i_conv = Conv2D(dim, shape, activation=acti, padding='same')(i)

        n = conv_block(m, dim, shape, acti, norm, do)
        m = AveragePooling2D()(n)
        m = Concatenate()([i_conv, m])
        
        m, out = level_block_mnet(i, m, int(inc*dim), shape, depth-1, inc, acti, norm, do, up, out)
        
        if up:
            m = UpSampling2D(interpolation='bilinear')(m)
            #m = Conv2D(dim, 2, activation=acti, padding='same')(m)
        else:
            m = Conv2DTranspose(dim, shape, strides=(2, 2), padding='same')(m)
        
        n = Concatenate()([n, m])
        m = conv_block(n, dim, shape, acti, norm)
        o = UpSampling2D(2**(4-depth))(m)
        o = Conv2D(1, (1, 1), activation = 'sigmoid')(o)

        out.append(o)
    else:
        m = conv_block(m, dim, shape, acti, norm, do)
    
    return m, out

def schedule(epoch, lr):
    if epoch > 10: lr = lr/10
    return lr

def Unet(img_shape = (96, 96, 1), out_ch=1, start_ch=32, depth=4, inc_rate=2, kernel_size = (3, 3), activation='relu', normalization=None, dropout=0, upconv = False, compile_model =True, learning_rate = 1e-5):
    i = Input(shape=img_shape)
    o = level_block_unet(i, start_ch, kernel_size, depth, inc_rate, activation, normalization, dropout, upconv)
    o = Conv2D(out_ch, (1, 1), activation = 'sigmoid')(o)
    model = Model(inputs=i, outputs=o)
    
    if compile_model: model.compile(optimizer=Adam(lr=learning_rate), loss = dice_coef_loss, metrics=[dice_coef])
    return model


def Mnet_rec(img_shape = (96, 96, 1), out_ch=1, start_ch=32, depth=4, inc_rate=2, kernel_size = (3, 3), activation='relu', normalization=None, dropout=0, upconv = True, compile_model =True, learning_rate = 1e-5):
    i = Input(shape=img_shape)
    n, l = level_block_mnet(i, i, start_ch, kernel_size, depth, inc_rate, activation, normalization, dropout, upconv)      
    o = Average()(l)
    model = Model(inputs=i, outputs=l[0])

    if compile_model: model.compile(optimizer=Adam(lr=learning_rate), loss = dice_coef_loss, metrics=[dice_coef])

    return model

def Mnet(img_shape = (96, 96, 1), out_ch=1, start_ch=32, depth=4, inc_rate=2, kernel_size = (3, 3), activation='relu', normalization=None, dropout=0, upconv = True, compile_model =True, learning_rate = 1e-5):
  
    i1 = Input(shape=img_shape)
    i2 = AveragePooling2D()(i1)
    i3 = AveragePooling2D()(i2)
    i4 = AveragePooling2D()(i3)
    
    l1 = conv_block(i1, 32, kernel_size, activation, normalization, dropout)
    n1 = Concatenate()([Conv2D(32, kernel_size, activation=activation, padding='same')(i2),AveragePooling2D()(l1)])
    l2 = conv_block(n1, 64, kernel_size, activation, normalization, dropout)
    n2 = Concatenate()([Conv2D(64, kernel_size, activation=activation, padding='same')(i3),AveragePooling2D()(l2)])
    l3 = conv_block(n2, 128, kernel_size, activation, normalization, dropout)
    n3 = Concatenate()([Conv2D(128, kernel_size, activation=activation, padding='same')(i4),AveragePooling2D()(l3)])
    l4 = conv_block(n3, 256, kernel_size, activation, normalization, dropout)
    n4 = AveragePooling2D()(l4)
    l5 = conv_block(n4, 512, kernel_size, activation, normalization)
    
    m1 = Concatenate()([l4, UpSampling2D(interpolation='bilinear')(l5)])
    u1 = conv_block(m1, 256, kernel_size, activation, normalization)
    m2 = Concatenate()([l3, UpSampling2D(interpolation='bilinear')(u1)])
    u2 = conv_block(m2, 128, kernel_size, activation, normalization)
    m3 = Concatenate()([l2, UpSampling2D(interpolation='bilinear')(u2)])
    u3 = conv_block(m3, 64, kernel_size, activation, normalization)
    m4 = Concatenate()([l1, UpSampling2D(interpolation='bilinear')(u3)])
    u4 = conv_block(m4, 32, kernel_size, activation, normalization)
    
    o4 = Conv2D(out_ch, (1, 1), activation='sigmoid', name='o4')(UpSampling2D(8)(u1))
    o3 = Conv2D(out_ch, (1, 1), activation='sigmoid', name='o3')(UpSampling2D(4)(u2))
    o2 = Conv2D(out_ch, (1, 1), activation='sigmoid', name='o2')(UpSampling2D(2)(u3))
    o1 = Conv2D(out_ch, (1, 1), activation='sigmoid', name='o1')(UpSampling2D(1)(u4))
    
    o = [o1, o2, o3, o4]
    
    model = Model(inputs=i1, outputs=o)

    #losses = [dice_coef_loss]*4
    losses = {"o1": dice_coef_loss, "o2": dice_coef_loss, "o3": dice_coef_loss, "o4": dice_coef_loss}
    lossWeights = [1, 1, 1, 1]
    
    if compile_model: model.compile(optimizer=Adam(lr=learning_rate), loss = losses, loss_weights=lossWeights, metrics=[dice_coef])

    return model

def eval_Mnet(test_images, test_masks, model, verbose=0):
    scores = []
    for i in range(len(test_images)):
        img = np.expand_dims(test_images[i], axis=0)
        results = model.predict(img)
        result = sum(results)/len(results)
        scores.append(dice_coef_pred(test_masks[i], result))
        if verbose == 1 and i % 100 == 0: print(f"tested {i}/{len(test_images)} images. score: {np.mean(scores)}")
    return -np.mean(scores), np.mean(scores)
