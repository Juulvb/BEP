from __future__ import print_function

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, UpSampling2D, AveragePooling2D, average
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K


K.set_image_data_format('channels_last')  # TF dimension ordering in this code

######################### define variables ##################################
img_rows = 96
img_cols = 96

smooth = 1.

###########################################  

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

def level_block_mnet(i, m, dim, shape, depth, inc, acti, norm, do, up, out = [], done=False):
    if depth > 0: 
        i = AveragePooling2D()(i)
        
        n = conv_block(m, dim, shape, acti, norm, do)
        m = AveragePooling2D()(n)
        m = Concatenate()([Conv2D(dim, shape, activation=acti, padding='same')(i), m])
        m, out = level_block_mnet(i, m, int(inc*dim), shape, depth-1, inc, acti, norm, do, up, out, done)
        
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
    nb_epoch = [0 ,10, 30]
    idx = max([i for i in range(len(nb_epoch)) if epoch>=nb_epoch[i]])
    print(lr*(10**(-idx)))
    return lr*(10**(-idx))

def Unet(img_shape = (96, 96, 1), out_ch=1, start_ch=32, depth=4, inc_rate=2, kernel_size = (3, 3), activation='relu', normalization=None, dropout=0, upconv = False, compile_model =True, learning_rate = 1e-5):
    i = Input(shape=img_shape)
    o = level_block_unet(i, start_ch, kernel_size, depth, inc_rate, activation, normalization, dropout, upconv)
    o = Conv2D(out_ch, (1, 1), activation = 'sigmoid')(o)
    model = Model(inputs=i, outputs=o)
    
    if compile_model: model.compile(optimizer=Adam(lr=learning_rate), loss = dice_coef_loss, metrics=[dice_coef])
    return model

def Mnet(img_shape = (96, 96, 1), out_ch=1, start_ch=32, depth=4, inc_rate=2, kernel_size = (3, 3), activation='relu', normalization=None, dropout=0, upconv = True, compile_model =True, learning_rate = 1e-5):
    i = Input(shape=img_shape)
    o, l = level_block_mnet(i, i, start_ch, kernel_size, depth, inc_rate, activation, normalization, dropout, upconv)
    
    o = average(l)
    
    model = Model(inputs=i, outputs=o)
    
    if compile_model: model.compile(optimizer=Adam(lr=learning_rate), loss = dice_coef_loss, metrics=[dice_coef])
    return model


