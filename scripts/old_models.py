

def get_mean_pooling_model(params):
    input = Input(shape=(params['n_nucleotides'],4))

    conv_layers = []
    for size in params['sizes_to_use']:
        conv = Conv1D(params['n_filters'], size, activation=params['conv_activation'], kernel_initializer=params['kernel_initializer_conv'])(input)
        max_pooled = GlobalMaxPooling1D()(conv)
        mean_pooled = GlobalAveragePooling1D()(conv)
        conv_layers.append(max_pooled)
        conv_layers.append(mean_pooled)

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

def gated_conv1d(x, filters, kernel_size, padding='same'):
    value = Conv1D(filters, kernel_size, padding=padding)(x)
    #gate = Conv1D(filters, kernel_size, padding=padding)(x)

    gate = Conv1D(filters, kernel_size, padding=padding, kernel_regularizer=l1_l2(l1=None, l2=0.01/(kernel_size**2)))(x)
    gate = Activation('sigmoid')(gate)
    return Multiply()([value, gate])



# def ranknet_loss(y_true, y_pred):
#     # Assuming y_true {0, 1} where 0 means the first item is ranked higher
#     P_ij = 1 / (1 + K.exp(-(y_pred[:, 0] - y_pred[:, 1])))
#     return -K.mean(y_true * K.log(P_ij) + (1 - y_true) * K.log(1 - P_ij))


def ranknet_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)  # Ensure y_true is of type float32
    P_ij = 1 / (1 + K.exp(-y_pred))  # Assuming y_pred is the difference score
    return -K.mean(y_true * K.log(P_ij) + (1 - y_true) * K.log(1 - P_ij))



def margin_ranking_loss(y_true, y_pred, margin=1.0):
    return K.mean(K.maximum(0.0, margin - (y_pred[:, 0] - y_pred[:, 1]) * y_true))

def get_pairwise_cnn_model(params):
    input1 = Input(shape=(params['n_nucleotides'], 4))
    input2 = Input(shape=(params['n_nucleotides'], 4))

    base_model = get_cnn_model(params, compile=False)

    processed1 = base_model(input1)
    processed2 = base_model(input2)

    # Compute the difference or any other pairwise transformation
    diff = Subtract()([processed1, processed2])
    

    model = Model(inputs=[input1, input2], outputs=diff)
    model.compile(optimizer=params['optimizer'], loss=ranknet_loss, metrics=params['metrics'])  # Use RankNet loss or switch to margin_ranking_loss
    return model




def second_cnn_model(params):
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
    


def generate_pairs(data, labels):
    # Assuming data is [samples, features] and labels are the corresponding rankings
    n = len(labels)
    pairs = []
    pair_labels = []

    for _ in range(2*n):
        # pick 2 random indices
        j = np.random.randint(n)
        k = np.random.randint(n)

        if labels[j] > labels[k]:
            pairs.append((data[j], data[k]))
            pair_labels.append(1)
        elif labels[j] < labels[k]:
            pairs.append((data[j], data[k]))
            pair_labels.append(-1)

    # for i in range(n):
    #     for j in range(i + 1, n):
    #         if labels[i] > labels[j]:
    #             pairs.append((data[i], data[j]))
    #             pair_labels.append(1)
    #         elif labels[i] < labels[j]:
    #             pairs.append((data[i], data[j]))
    #             pair_labels.append(-1)  # Use 0 for RankNet

    pair_data = np.array(pairs)
    pair_labels = np.array(pair_labels)
    return pair_data, pair_labels



def get_trained_model_rank(pair_data, pair_labels, params):
    params = get_default_params(params)  # Ensure this function correctly initializes any needed parameters
    

    # Model setup
    print('Using cnn model for pairwise input')
    model = get_pairwise_cnn_model(params)

    # Splitting data into pairs
    # Assuming pair_data is already split into pairs as needed by the model
    input_1 = pair_data[:, 0]  # First item of each pair
    input_2 = pair_data[:, 1]  # Second item of each pair

    # Train model
    callbacks = [tf.keras.callbacks.EarlyStopping(patience=params['patience'], restore_best_weights=True, monitor='val_loss')]
    history = model.fit([input_1, input_2], pair_labels,
                         epochs=params['epochs'], batch_size=params['batch_size'],
                         validation_split=params['validation_split'],
                         callbacks=callbacks, workers=8,
                         use_multiprocessing=True)

    return model, history