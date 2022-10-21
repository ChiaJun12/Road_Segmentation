from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, concatenate, Rescaling, Conv2DTranspose 
from keras_tuner import RandomSearch

patch_size = 128

def unet(input_shape=(patch_size, patch_size, 3)): 
    inputs = Input(input_shape)
    rescale = Rescaling(1. / 255, input_shape = (patch_size, patch_size, 3))(inputs)
    
    no_filter = [64, 128, 256]
    contraction = {}
    
    temp_layer = rescale
    
    for filters in no_filter:
        x = Conv2D(filters, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(temp_layer)
        x = Conv2D(filters, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(x)
        contraction[f'conv{filters}'] = x
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Dropout(0.1)(x)
        temp_layer = x

    x = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(temp_layer)
    x = Dropout(0.1)(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(x)
    
    for filters in reversed(no_filter):
        x = Conv2DTranspose(filters, (2, 2), padding='same', strides=(2, 2))(temp_layer)
        x = concatenate([x, contraction[f'conv{filters}']])
        x = Conv2D(filters, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(x)
        x = Dropout(0.1)(x)
        x = Conv2D(filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
        temp_layer = x
    
    outputs = Conv2D(filters=23, kernel_size=(1, 1), activation="softmax")(temp_layer)
    
    return Model(inputs=inputs, outputs=outputs)
