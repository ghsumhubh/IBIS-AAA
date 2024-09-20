
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Conv1D, Flatten, Dense
from keras.layers import GlobalMaxPooling1D, Dropout, Concatenate
import scipy.stats
from sklearn.model_selection import KFold
import numpy as np


import tensorflow as tf
def spearman_correlation(y_true, y_pred):
    return tf.py_function(
        lambda a, b: scipy.stats.spearmanr(a, b, axis=0)[0], 
        [tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)], 
        Tout=tf.float32
    )


def get_model_with_kmm_input(optimizer, loss):
       # input
    input = Input(shape=(35,4)) # was 35,4
    input_knn = Input(shape=(1))

    # N_FILTERS = 64
    # SIZES_TO_USE = [3, 5, 7, 9, 11, 13]
    # DROPOUT_RATE = 0.5
    # dense_list = [(128, 0.2), (64,0), (32, 0)]
    N_FILTERS = 64
    SIZES_TO_USE = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    DROPOUT_RATE = 0.5
    dense_list = [(128, 0.2), (64,0),(32,0), (32, 0)]

    conv_layers = []

    # Create convolutional layers in a loop
    for size in SIZES_TO_USE:
        conv = Conv1D(N_FILTERS, size, activation='relu')(input)
        pooled = GlobalMaxPooling1D()(conv)
        conv_layers.append(pooled)
        #conv_layers.append(Flatten()(conv))

    # Concatenate all the convolutional layer outputs
    x = Concatenate()(conv_layers)

    # dropout
    x = Dropout(DROPOUT_RATE)(x)

    # flatten
    x = Flatten()(x)

    # concatenate with knn input
    x = Concatenate()([x, input_knn])

    for size, rate in dense_list:
        x = Dense(size, activation='relu')(x)
        if rate > 0:
            x = Dropout(rate)(x)

    # output

    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[input, input_knn], outputs=output)

    #auroc_metric = AUC(curve='ROC', name='auroc', from_logits=False)
    #aupr_metric = AUC(curve='PR', name='aupr', from_logits=False)

    #print(auroc_metric)

    model.compile(optimizer=optimizer, loss=loss,
                   metrics=[spearman_correlation])
    
    # add initialization
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel_initializer = keras.initializers.he_normal()
    
    return model




def get_trained_knn_model(train_data, train_labels, knn_data):
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    loss = keras.losses.MeanSquaredError()
    model = get_model_with_kmm_input(optimizer, loss)

    # get validation data as 10% of training data
    validation_split = 0.1

    # train the model with early stopping
    callbacks = [keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True, monitor='val_loss')]
    model.fit([train_data, knn_data], train_labels, epochs=200, batch_size=256,
                validation_split=validation_split, callbacks=callbacks)
    
    return model
    

def get_trained_knn_models(train_data, train_labels, knn_data, n_splits=5):
    # Define the K-fold cross-validator
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Store the models
    models = []

    # Loop through each split
    for train_index, val_index in kfold.split(train_data):
        # Split the data
        train_data_split, val_data_split = train_data[train_index], train_data[val_index]
        knn_data_split, val_knn_data_split = knn_data[train_index], knn_data[val_index]
        train_labels_split, val_labels_split = train_labels[train_index], train_labels[val_index]

        # Set up the optimizer and loss function
        optimizer = keras.optimizers.Adam(learning_rate=0.0001)
        loss = keras.losses.MeanSquaredError()

        # Create the model
        model = get_model_with_kmm_input(optimizer, loss)

        # Train the model with early stopping
        callbacks = [keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True, monitor='val_loss')]
        model.fit([train_data_split, knn_data_split], train_labels_split, epochs=200, batch_size=256,
                  validation_data=([val_data_split, val_knn_data_split], val_labels_split),
                  callbacks=callbacks)

        # Store the model
        models.append(model)

    return models


def custom_weights(distances):
    # make each 0 distance have 0 weight
    indexes_of_same = np.where(distances == 0)
    # Add 1 to each distance to ensure no distance is zero
    distances = distances + 1
    # Compute the reciprocal of the modified distances
    scaled = 1 / distances
    # Set the 0 distances to 0
    scaled[indexes_of_same] = 0
    return scaled