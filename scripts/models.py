
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Conv1D, Flatten, Dense
from keras.layers import GlobalMaxPooling1D, Dropout, Concatenate, GlobalAveragePooling1D, MaxPooling1D, AveragePooling1D, BatchNormalization
import scipy.stats
import tensorflow as tf
from keras.regularizers import l1_l2
from keras.layers import Multiply, Conv1D, Activation, Subtract
from keras import backend as K
from sklearn.model_selection import KFold

import numpy as np
import os

from keras.models import load_model

def spearman_correlation(y_true, y_pred):
    return tf.py_function(
        lambda a, b: scipy.stats.spearmanr(a, b, axis=0)[0], 
        [tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)], 
        Tout=tf.float32
    )



def get_last_layer(experiment, kernel_initializer_last_layer):
    if experiment in [None, 'PBM', 'SMS', 'CHS']:
        return Dense(1, activation='sigmoid', kernel_initializer=kernel_initializer_last_layer)
    elif experiment in ['HTS']:
         return Dense(1, activation='relu', kernel_initializer=kernel_initializer_last_layer)
    else:
        raise ValueError('Experiment not implemented')



def get_default_params(params=None):
    if params is None:
        params = {}

    params['model_type'] = params.get('model_type', 'simple')

    params['steps_per_epoch'] = params.get('steps_per_epoch', None)
    params['patience'] = params.get('patience', 50)
    params['experiment'] = params.get('experiment', None)
    params['optimizer'] = params.get('optimizer', keras.optimizers.Adam(learning_rate=0.0001))
    params['validation_split'] = params.get('validation_split', 0.1)
    params['epochs'] = params.get('epochs', 200)
    params['batch_size'] = params.get('batch_size', 512)
    if params['experiment'] in [None, 'PBM','HTS']:
        params['loss'] = params.get('loss', keras.losses.MeanSquaredError())
    elif params['experiment'] in ['SMS', 'CHS']:
        params['loss'] = params.get('loss', keras.losses.BinaryCrossentropy())
    else:
        raise ValueError('Experiment not implemented')
    

    params['optimizer'] = params.get('optimizer', keras.optimizers.Adam(learning_rate=0.0001))
    params['n_filters'] = params.get('n_filters', 64)
    params['sizes_to_use'] = params.get('sizes_to_use', [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    params['conv_activation'] = params.get('conv_activation', 'relu')
    params['dropout_rate'] = params.get('dropout_rate', 0.5)
    params['dense_list'] = params.get('dense_list', [(128, 0.2), (64,0),(32,0), (32, 0)])
    params['dense_activation'] = params.get('dense_activation', 'relu')
    params['kernel_initializer_conv'] = params.get('kernel_initializer_conv', keras.initializers.he_normal())
    params['kernel_initializer_dense']= params.get('kernel_initializer_dense', keras.initializers.he_normal())
    params['kernel_initializer_last_layer'] = params.get('kernel_initializer_ll', keras.initializers.he_normal())
    params['metrics'] = params.get('metrics', [spearman_correlation])

    
    return params



def get_cnn_model(params, compile=True):
    input = Input(shape=(params['n_nucleotides'],4))

    conv_layers = []
    for size in params['sizes_to_use']:
        #conv = gated_conv1d(input, params['n_filters'], size)
        conv = Conv1D(params['n_filters'], size, activation=params['conv_activation'], kernel_initializer=params['kernel_initializer_conv'], kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4))(input)
        pooled = GlobalMaxPooling1D()(conv)
        conv_layers.append(pooled)

    x = Concatenate()(conv_layers)
    x = Dropout(params['dropout_rate'])(x)


    for size, rate in params['dense_list']:
        x = Dense(size, activation=params['dense_activation'], kernel_initializer=params['kernel_initializer_dense'])(x)
        x = Dropout(rate)(x) if rate > 0 else x

    # output
    output = get_last_layer(params['experiment'], params['kernel_initializer_last_layer'])(x)
    
    model = Model(inputs=input, outputs=output)
    if compile:
        model.compile(optimizer=params['optimizer'], loss=params['loss'], metrics=params['metrics'])
    
    return model




def get_simple_model(params):
    input = Input(shape=(params['n_nucleotides'],4))

    conv_layers = []
    for size in params['sizes_to_use']:
        conv = Conv1D(params['n_filters'], size, activation=params['conv_activation'], kernel_initializer=params['kernel_initializer_conv'])(input)
        pooled = GlobalMaxPooling1D()(conv)
        conv_layers.append(pooled)

    x = Concatenate()(conv_layers)
    x = Dropout(params['dropout_rate'])(x)


    for size, rate in params['dense_list']:
        x = Dense(size, activation=params['dense_activation'], kernel_initializer=params['kernel_initializer_dense'])(x)
        x = Dropout(rate)(x) if rate > 0 else x

    # output
    output = get_last_layer(params['experiment'], params['kernel_initializer_last_layer'])(x)
    
    model = Model(inputs=input, outputs=output)

    model.compile(optimizer=params['optimizer'], loss=params['loss'], metrics=params['metrics'])
    
    return model


def get_converted_model(original_model, n_nucleotides, params= None):
    params = get_default_params(params)
    params['n_nucleotides'] = n_nucleotides
    if params['model_type'] == 'simple':
        new_model = get_simple_model(params)
    elif params['model_type'] == 'cnn':
        new_model = get_cnn_model(params)
    else:
        raise ValueError('Model type not implemented')

    # copy all weights
    for i in range(len(original_model.layers)):
        new_model.layers[i].set_weights(original_model.layers[i].get_weights())

    return new_model




def get_trained_model(train_data, train_labels,params):
    params = get_default_params(params)
    params['n_nucleotides']= train_data.shape[1]

    if params['model_type'] == 'simple':
        print('Using simple model')
        model = get_simple_model(params)
    elif params['model_type'] == 'cnn':
        print('Using cnn model')
        model = get_cnn_model(params)
    else:
        raise ValueError('Model type not implemented')
    
    print(params)


    # train model
    callbacks = [keras.callbacks.EarlyStopping(patience=params['patience'], restore_best_weights=True, monitor='val_loss')]
    model.fit(train_data, train_labels,
               epochs=params['epochs'], batch_size=params['batch_size'],
                validation_split=params['validation_split'],
                  callbacks=callbacks, workers=8,
                    use_multiprocessing=True, steps_per_epoch=params['steps_per_epoch'])
    


    
    return model




def get_ensemble(data, labels, params, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold = 1
    models = []
    histories = []
    original_params = params

    for train_index, val_index in kf.split(data):
        print(f"Training fold {fold}")
        train_data, val_data = data[train_index], data[val_index]
        train_labels, val_labels = labels[train_index], labels[val_index]

        params = original_params.copy()
        params = get_default_params(params)
        params['n_nucleotides'] = train_data.shape[1]

        if params['model_type'] == 'simple':
            print('Using simple model')
            model = get_simple_model(params)
        elif params['model_type'] == 'cnn':
            print('Using cnn model')
            model = get_cnn_model(params)
        else:
            raise ValueError('Model type not implemented')
        
        print(params)

        callbacks = [keras.callbacks.EarlyStopping(patience=params['patience'], restore_best_weights=True, monitor='val_loss')]
        history = model.fit(train_data, train_labels,
                            epochs=params['epochs'], batch_size=params['batch_size'],
                            validation_data=(val_data, val_labels),
                            callbacks=callbacks, workers=8,
                            use_multiprocessing=True, steps_per_epoch=params['steps_per_epoch'])

        models.append(model)
        histories.append(history)
        fold += 1

    return models


def ensemble_predict(models, data):
    predictions = []

    for model in models:
        pred = model.predict(data, batch_size=256, workers=8, use_multiprocessing=True, verbose=1)
        predictions.append(pred)

    # Averaging the predictions from each model
    ensemble_predictions = np.mean(predictions, axis=0)

    return ensemble_predictions



    


def check_if_protein_model_exists(experiment, protein, custom_name=None):
    path = f'models/{experiment}/{protein}_model'
    if custom_name is not None:
        path += f'_{custom_name}'
    path += '.h5'
    return os.path.exists(path)


def save_protein_model(model, experiment, protein, custom_name=None):
    path = f'models/{experiment}/{protein}_model'
    if custom_name is not None:
        path += f'_{custom_name}'
    path += '.h5'
    model.save(path)

def load_protein_model(experiment, protein, custom_name=None):
    path = f'models/{experiment}/{protein}_model'
    if custom_name is not None:
        path += f'_{custom_name}'
    path += '.h5'
    model = load_model(path, compile=False)
    return model



def check_if_ensemble_model_exists(experiment, protein, custom_name=None, num_models=5):
    for i in range(num_models):
        path = f'models/{experiment}/{protein}_model_{i}'
        if custom_name is not None:
            path += f'_{custom_name}'
        path += '.h5'
        if not os.path.exists(path):
            return False
    return True

def save_ensemble_model(models, experiment, protein, custom_name=None):
    for i, model in enumerate(models):
        path = f'models/{experiment}/{protein}_model_{i}'
        if custom_name is not None:
            path += f'_{custom_name}'
        path += '.h5'
        model.save(path)


def load_ensemble_model(experiment, protein, custom_name=None, num_models=5):
    models = []
    for i in range(num_models):
        path = f'models/{experiment}/{protein}_model_{i}'
        if custom_name is not None:
            path += f'_{custom_name}'
        path += '.h5'
        model = load_model(path, compile=False)
        models.append(model)
    return models