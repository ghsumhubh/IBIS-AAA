from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Conv1D, Flatten, Dense
from keras.layers import GlobalMaxPooling1D, Dropout, Concatenate
import scipy.stats
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.losses import binary_crossentropy, sparse_categorical_crossentropy
import numpy as np
from scipy.stats import spearmanr



class SpearmanCorrelationCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data  # Store validation data

    def on_epoch_end(self, epoch, logs=None):
        # Predict using the validation data
        x_val, y_val = self.validation_data
        y_pred = self.model.predict(x_val)
        # Assuming 'pbm_head' predictions are the first element of the output
        spearman_corr = spearmanr(y_val.flatten(), y_pred[0].flatten()).correlation
        spearman_corr_2 = spearmanr(y_val.flatten(), y_pred[1].flatten()).correlation
        # for third we need to do argmax to get the class
        y_pred_2 = np.argmax(y_pred[2], axis=1)
        spearman_corr_3 = spearmanr(y_val.flatten(), y_pred_2.flatten()).correlation
        logs['val_spearman_corr'] = spearman_corr
        logs['val_spearman_corr_2'] = spearman_corr_2
        logs['val_spearman_corr_3'] = spearman_corr_3
        print(f"Spearman Correlation - Epoch {epoch + 1}: {spearman_corr}, {spearman_corr_2}, {spearman_corr_3}")

def get_model(optimizer, loss, n_hts_classes=4,):
    # input
    input = Input(shape=(40,4))

    # Parameters
    N_FILTERS = 128
    SIZES_TO_USE = list(range(5, 21))
    DROPOUT_RATE = 0.5
    dense_shared_list = [(256, 0.2), (128, 0.1)]
    dense_individual_list = [(64,0), (32,0), (32, 0)]  
    # TODO: temp
    dense_shared_list = [(256, 0.2), (128, 0.1), (64,0), (32,0)]
    dense_individual_list = [(32, 0)]  

    # Create convolutional layers in a loop
    conv_layers = []
    for size in SIZES_TO_USE:
        conv = Conv1D(N_FILTERS, size, activation='relu')(input)
        pooled = GlobalMaxPooling1D()(conv)
        conv_layers.append(pooled)

    # Concatenate all the convolutional layer outputs
    x = Concatenate()(conv_layers)
    x = Dropout(DROPOUT_RATE)(x)
    x = Flatten()(x)

    # Add shared dense layers
    for i, (size, rate) in enumerate(dense_shared_list):
        x = Dense(size, activation='relu')(x)
        if rate > 0:
            x = Dropout(rate)(x)

    # After the kth shared layer, branching out for each head
    pbm_x, sms_x, hts_x = x, x, x  # start individual branches with the output of last shared dense layer

    # Individual branches
    for size, rate in dense_individual_list:
        pbm_x = Dense(size, activation='relu')(pbm_x)
        sms_x = Dense(size, activation='relu')(sms_x)
        hts_x = Dense(size, activation='relu')(hts_x)
        if rate > 0:
            pbm_x = Dropout(rate)(pbm_x)
            sms_x = Dropout(rate)(sms_x)
            hts_x = Dropout(rate)(hts_x)

    # Each output layer
    pbm_head = Dense(1, activation='sigmoid', name='pbm_head')(pbm_x)
    sms_head = Dense(1, activation='sigmoid', name='sms_head')(sms_x)
    hts_head = Dense(n_hts_classes, activation='softmax', name='hts_head')(hts_x)

    model = Model(inputs=input, outputs=[pbm_head, sms_head, hts_head])
    model.compile(optimizer=optimizer, loss=loss)

    # Initialize weights with He normal initializer
    for layer in model.layers:
        if hasattr(layer, 'kernel_initializer'):
            layer.kernel_initializer = keras.initializers.he_normal()
    
    return model




def custom_loss_wrapper(output_index):

    def custom_loss(y_true, y_pred):
        # Find indices where y_true is not -1
        valid_indices = tf.where(tf.not_equal(y_true, -1))

        # Check if there are any valid indices
        if tf.size(valid_indices) == 0:
            return tf.constant(0.0)  # Return zero loss if no valid indices
        
        
        # Extract only the valid entries for both predictions and labels
        y_true_valid = tf.gather_nd(y_true, valid_indices)
        y_pred_valid = tf.gather_nd(y_pred, valid_indices)

        #tf.print("Shapes - y_true_valid:", tf.shape(y_true_valid), "y_pred_valid:", tf.shape(y_pred_valid))

        
        if output_index == 0:  # pbm_head, regression
            # Calculate MSE only for valid entries
            mse = tf.keras.losses.mean_squared_error(y_true_valid, y_pred_valid)
            mse = tf.multiply(1000.0, mse)
            return tf.reduce_mean(mse)
        elif output_index == 1:  # sms_head, binary classification
            # Calculate Binary Crossentropy only for valid entries
            bce = tf.keras.losses.binary_crossentropy(y_true_valid, y_pred_valid)
            bce = tf.multiply(10.0, bce)
            return tf.reduce_mean(bce)
        elif output_index == 2:  # hts_head, multi-class classification
            # Calculate Sparse Categorical Crossentropy only for valid entries
            # make into 1d array
            sce = tf.keras.losses.categorical_crossentropy(y_true_valid, y_pred_valid)
            sce = tf.multiply(0.001, sce)
            return tf.reduce_mean(sce)

    return custom_loss


def predict_using_specific_head(model, X, head_index):
    if head_index in [0,1]:
        y_pred = model.predict(X)[head_index]
    else:
        y_pred = model.predict(X)[2]
        y_pred = np.argmax(y_pred, axis=1)

    # normalize between 0 and 1
    y_pred = (y_pred - np.min(y_pred)) / (np.max(y_pred) - np.min(y_pred))

    return y_pred




def train_model(data):
    # Assuming data is a DataFrame with the necessary columns.
    # Prepare your inputs and outputs here.
    X = data['Sequence'].values
    # make into 3d array from X
    X = np.stack(X, axis=0)

    y_pbm = data['pbm score'].values
    y_sms = data['sms score'].values
    y_hts = data['hts class'].values


    X_train, X_val, y_pbm_train, y_pbm_val, y_sms_train, y_sms_val, y_hts_train, y_hts_val = train_test_split(
        X, y_pbm, y_sms, y_hts, test_size=0.2, random_state=42)


    valid_pbm_indices = np.where(y_pbm_train != -1)[0]
    valid_sms_indices = np.where(y_sms_train != -1)[0]
    valid_hts_indices = np.where(y_hts_train != -1)[0]


    max_size = max(len(valid_pbm_indices), len(valid_sms_indices), len(valid_hts_indices))

    # Randomly sample indices to balance the dataset (with replacement to reach max_size)
    np.random.seed(42)  # For reproducibility
    sampled_pbm_indices = np.random.choice(valid_pbm_indices, max_size, replace=True)
    sampled_sms_indices = np.random.choice(valid_sms_indices, max_size, replace=True)
    sampled_hts_indices = np.random.choice(valid_hts_indices, max_size, replace=True)



    # Extract the sampled entries for X and y
    X_pbm_train = X_train[sampled_pbm_indices]
    y_pbm_train = y_pbm_train[sampled_pbm_indices]

    X_sms_train = X_train[sampled_sms_indices]
    y_sms_train = y_sms_train[sampled_sms_indices]

    X_hts_train = X_train[sampled_hts_indices]
    y_hts_train = y_hts_train[sampled_hts_indices]


    # Combine the balanced datasets
    X_train = np.concatenate((X_pbm_train, X_sms_train, X_hts_train), axis=0)

    # extend the y_pbm to the same size as X by padding after
    y_pbm_train = np.concatenate((y_pbm_train, -1*np.ones(2*max_size)))

    # extend the y_sms to the same size as X by padding before and after
    y_sms_train = np.concatenate((-1*np.ones(max_size), y_sms_train, -1*np.ones(max_size)))

    # extend the y_hts to the same size as X by padding before
    y_hts_train = np.concatenate((-1*np.ones(2*max_size), y_hts_train))


    



    # get the number of classes, we remove 1 since we have -1 as the unknown class
    n_classes = len(np.unique(y_hts))-1

    y_hts_train = tf.keras.utils.to_categorical(y_hts_train, num_classes=n_classes)
    y_hts_val = tf.keras.utils.to_categorical(y_hts_val, num_classes=n_classes)


    
    # Compile losses
    losses = {
    'pbm_head': custom_loss_wrapper(0),
    'sms_head': custom_loss_wrapper(1),
    'hts_head': custom_loss_wrapper(2)
    }

    # Build model
    model = get_model(Adam(learning_rate=0.0001), losses, n_classes)

    spearman_valid_indices = np.where(y_pbm_val != -1)
    x_spearman_val = X_val[spearman_valid_indices]
    y_spearman_val = y_pbm_val[spearman_valid_indices]
    validation_data = (x_spearman_val, y_spearman_val)

    spearman_callback = SpearmanCorrelationCallback(validation_data)

    callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor='val_loss'), spearman_callback]


    # Train model
    # history = model.fit(
    #     X_train,
    #     {'pbm_head': y_pbm_train, 'sms_head': y_sms_train, 'hts_head': y_hts_train},
    #     validation_data=(X_val, {'pbm_head': y_pbm_val, 'sms_head': y_sms_val, 'hts_head': y_hts_val}),
    #     batch_size=256,
    #     epochs=200,
    #     callbacks=callbacks
    # )


    history = model.fit(
        X_train,
        {'pbm_head': y_pbm_train, 'sms_head': y_sms_train, 'hts_head': y_hts_train},
        validation_data=(X_val, {'pbm_head': y_pbm_val, 'sms_head': y_sms_val, 'hts_head': y_hts_val}),
        batch_size=256,
        epochs=200,
        callbacks=callbacks,
        steps_per_epoch= 1000
    )

    return model, history