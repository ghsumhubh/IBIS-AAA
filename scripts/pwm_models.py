
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Conv1D, Flatten, Dense
from keras.layers import GlobalMaxPooling1D, Dropout, Concatenate, GlobalAveragePooling1D, MaxPooling1D, AveragePooling1D, BatchNormalization
import scipy.stats
import tensorflow as tf
from keras.regularizers import l1_l2
from keras.layers import Multiply, Conv1D, Activation, Subtract
from keras import backend as K
from scripts.models import get_default_params

import numpy as np

from keras.models import load_model






def get_pwm_model(params):
    input = Input(shape=(params['n_nucleotides'],4))
    # no activation
    conv = Conv1D(1, params['size_to_use'], activation= None, kernel_initializer=params['kernel_initializer_conv'])(input)
    # max pool
    pool = GlobalMaxPooling1D()(conv)

    model = Model(inputs=input, outputs=pool)
    model.compile(optimizer=params['optimizer'], loss=params['loss'])
    return model

def get_filter_from_pwm_model(model):
    filter = model.layers[1].get_weights()[0]
    return filter

def print_filter(filter, protein, info=None):
    if info is None:
        info = 'booboogaga'
    print(f'>{protein} {protein}_{info}')
    for i in range(filter.shape[0]):
        print(' '.join([f'{x:.5f}' for x in filter[i,:,0]]))


# def pwm_to_pfm(pwm):
#     pfm = np.exp(pwm)
#     pfm = pfm / np.sum(pfm, axis=1)[:,None]
#     return pfm

def pwm_to_pfm(pwm):
    # first make sure each nucleotide is positive by looking at the min value for each position
    min_vals = np.min(pwm, axis=2)
    print(min_vals)

def get_pwm_model_and_score(train_data, train_labels, size_to_use):
    params = get_default_params()
    params['n_nucleotides']= train_data.shape[1]
    params.update({'size_to_use':size_to_use})
    model = get_pwm_model(params)
    # TODO: temp
    params['patience'] = 3
    params['batch_size'] = 4096


    # train model
    callbacks = [keras.callbacks.EarlyStopping(patience=params['patience'], restore_best_weights=True, monitor='val_loss')]
    model.fit(train_data, train_labels,
               epochs=params['epochs'], batch_size=params['batch_size'],
                validation_split=params['validation_split'],
                  callbacks=callbacks, workers=8,
                    use_multiprocessing=True, steps_per_epoch=params['steps_per_epoch'])
    
    # get the validation loss
    val_loss = model.history.history['val_loss'][-1]
    
    return model, val_loss
    

def get_best_pwm(train_data,train_labels):
    params = get_default_params()
    best_model = None
    best_val_loss = np.inf
    #sizes = [4,8,16]
    sizes = [8]
    for size in sizes:
        print(f'getting pwm with size {size}')
        model, val_loss = get_pwm_model_and_score(train_data, train_labels, size_to_use=size)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

    # get filter from best model
    filter = get_filter_from_pwm_model(best_model)
    return filter

