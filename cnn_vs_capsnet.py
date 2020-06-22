
import numpy as np
import keras
from keras.datasets import mnist


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)


def get_train_data(labels_per_class):
    x_test_chunk = []
    y_test_chunk = []
    current_label_nums = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in np.random.permutation(len(y_train)):
        current_label = np.argmax(y_train[i])
        if current_label_nums[current_label] < labels_per_class:
            current_label_nums[current_label] += 1
            x_test_chunk.append(x_train[i])
            y_test_chunk.append(y_train[i])
        if sum(current_label_nums) == labels_per_class * 10:
            break
    return np.array(x_test_chunk), np.array(y_test_chunk)


def get_test_data():
    return x_test, y_test



def get_train_data_generator(labels_per_class, batch_size):
    current_index = 0
    x, y = get_train_data(labels_per_class)
    while current_index < len(x):
        yield x[current_index:current_index+batch_size], y[current_index:current_index+batch_size]
        current_index += batch_size


def get_test_data_generator(batch_size):
    current_index = 0
    x, y = get_test_data()
    while current_index < len(x):
        yield x[current_index:current_index+batch_size], y[current_index:current_index+batch_size]
        current_index += batch_size

import tensorflow as tf
import numpy as np

from time import time

caps1_n_maps = 16
caps1_n_caps = caps1_n_maps * 6 * 6  
caps1_n_dims = 6
caps2_n_caps = 10  
caps2_n_dims = 8
routing_rounds = 3

def squash(s, axis=-1, epsilon=1e-7, name=None):
    with tf.name_scope(name, default_name="squash"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keep_dims=True)
        safe_norm = tf.sqrt(squared_norm + epsilon)
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = s / safe_norm
        return squash_factor * unit_vector


def safe_norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    with tf.name_scope(name, default_name="safe_norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keep_dims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)


def get_capsnet_performance(labels_per_class):
    tf.reset_default_graph()
    X = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32, name="X")
    y = tf.placeholder(shape=[None, 10], dtype=tf.int64, name="y")
    batch_size = tf.shape(X)[0]

    conv1_params = {
        "filters": caps1_n_maps * caps1_n_dims,
        "kernel_size": 9,
        "strides": 1,
        "padding": "valid",
        "activation": tf.nn.relu,
    }

    conv2_params = {
        "filters": caps1_n_maps * caps1_n_dims,
        "kernel_size": 9,
        "strides": 2,
        "padding": "valid",
        "activation": tf.nn.relu
    }

    conv1 = tf.layers.conv2d(X, name="conv1", **conv1_params)
    conv2 = tf.layers.conv2d(conv1, name="conv2", **conv2_params)
    caps1_raw = tf.reshape(conv2, [-1, caps1_n_caps, caps1_n_dims], name="caps1_raw")
    caps1_output = squash(caps1_raw, name="caps1_output")

 
    W_init = tf.random_normal(
        shape=(1, caps1_n_caps, caps2_n_caps, caps2_n_dims, caps1_n_dims),
        stddev=0.1, dtype=tf.float32, name="W_init"
    )
    W = tf.Variable(W_init, name="W")
    W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled")

    caps1_output_expanded = tf.expand_dims(caps1_output, -1, name="caps1_output_expanded")
    caps1_output_tile = tf.expand_dims(caps1_output_expanded, 2, name="caps1_output_tile")
    caps1_output_tiled = tf.tile(caps1_output_tile, [1, 1, caps2_n_caps, 1, 1], name="caps1_output_tiled")
    caps2_predicted = tf.matmul(W_tiled, caps1_output_tiled, name="caps2_predicted")

  
    raw_weights = tf.zeros([batch_size, caps1_n_caps, caps2_n_caps, 1, 1], dtype=np.float32, name="raw_weights")
    routing_weights = tf.nn.softmax(raw_weights, dim=2, name="routing_weights")
    weighted_predictions = tf.multiply(routing_weights, caps2_predicted, name="weighted_predictions")
    weighted_sum = tf.reduce_sum(weighted_predictions, axis=1, keep_dims=True, name="weighted_sum")
    caps2_output_round_1 = squash(weighted_sum, axis=-2, name="caps2_output_round_1")

    def has_more_rounds(previous_round_output, rounds_passed):
        return tf.less(rounds_passed, routing_rounds)

    def do_routing_round(previous_round_output, rounds_passed):
        previous_round_output_tiled = tf.tile(previous_round_output, [1, caps1_n_caps, 1, 1, 1])
        agreement = tf.matmul(caps2_predicted, previous_round_output_tiled, transpose_a=True)
        raw_weights_current_round = tf.add(raw_weights, agreement)
        routing_weights_current_round = tf.nn.softmax(raw_weights_current_round, dim=2)
        weighted_predictions_current_round = tf.multiply(routing_weights_current_round, caps2_predicted)
        weighted_sum_current_round = tf.reduce_sum(weighted_predictions_current_round, axis=1, keep_dims=True)
        return squash(weighted_sum_current_round, axis=-2), tf.add(rounds_passed, 1)

    rounds_passed = tf.constant(1)
    caps2_output = tf.while_loop(has_more_rounds, do_routing_round, [caps2_output_round_1, rounds_passed], swap_memory=True)[0]

    y_proba = safe_norm(caps2_output, axis=-2, name="y_proba")
    y_proba_softmax = tf.nn.softmax(tf.squeeze(y_proba, axis=[1, 3]), name="y_proba_softmax")

    loss = tf.reduce_mean(-tf.reduce_sum(tf.cast(y, tf.float32) * tf.log(y_proba_softmax), reduction_indices=[1]))

    correct = tf.equal(
        tf.argmax(y, axis=-1),
        tf.argmax(y_proba_softmax, axis=-1),
        name="correct"
    )
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

   

    learning_rate = tf.placeholder(tf.float32, shape=[])
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss, name="training_op")

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        epochs_wo_improvement = 0
        epochs_wo_improvement_lr = 0
        current_lr = 0.001
        best_loss = None
        
        train_data = list(get_train_data_generator(labels_per_class, 16))

        start = time()
        while True:
            accumulated_loss = 0
            train_accs = []
            for batch in train_data:
                X_batch, y_batch = batch
                _, loss_train, accuracy_train = sess.run(
                    [training_op, loss, accuracy],
                    feed_dict={
                        X: X_batch,
                        y: y_batch,
                        learning_rate: current_lr
                    }
                )
                accumulated_loss += loss_train
                train_accs.append(accuracy_train)

            if best_loss is None or best_loss > accumulated_loss * 1.001:
                best_loss = float(accumulated_loss)
                epochs_wo_improvement = 0
                epochs_wo_improvement_lr = 0
            else:
                epochs_wo_improvement += 1
                epochs_wo_improvement_lr += 1

          
            if epochs_wo_improvement_lr >= 5:
                epochs_wo_improvement_lr = 0
                current_lr = current_lr * 0.3

            
            if epochs_wo_improvement >= 10:
                break

        train_time = time() - start
        train_acc = np.mean(train_accs)

        start = time()
        accuracies = []
        num_test = 0
        for batch in get_test_data_generator(16):
            X_batch, y_batch = batch
            num_test += len(y_batch)
            batch_accuracy = sess.run(
                [accuracy],
                feed_dict={
                    X: X_batch,
                    y: y_batch
                }
            )
            accuracies.append(batch_accuracy)
        test_time = time() - start

        
        test_acc = np.mean(accuracies)

        return train_acc, test_acc, train_time, 1000 * test_time / num_test

from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.backend import clear_session
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization

from time import time


image_size = 28
num_channels = 1
num_labels = 10

batch_size = 16
patch_size = 3




num_hidden_1 = 400
num_hidden_2 = 300
learning_rate = 0.001
dropout_rate_1 = 0.1
dropout_rate_2 = 0.2
dropout_rate_3 = 0.2
num_filters_1 = 16
num_filters_2 = 32
num_filters_3 = 64


def get_cnn_performance(labels_per_class):
    clear_session()
    model = Sequential()
    model.add(
        Conv2D(
            num_filters_1,
            (patch_size, patch_size),
            activation='relu',
            padding="valid",
            input_shape=(image_size, image_size, num_channels),
            kernel_initializer='he_uniform'
        )
    )
    model.add(Conv2D(num_filters_2, (patch_size, patch_size), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D(pool_size=(patch_size, patch_size)))
    model.add(Conv2D(num_filters_3, (patch_size, patch_size), activation='relu', kernel_initializer='he_uniform'))
    model.add(Flatten())
    model.add(Dropout(dropout_rate_1))
    model.add(Dense(num_hidden_1, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate_2))
    model.add(Dense(num_hidden_2, activation='relu', kernel_initializer='he_uniform'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate_3))
    model.add(Dense(num_labels, kernel_initializer='he_uniform', activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(lr=learning_rate),
        metrics=['accuracy']
    )

    x_train, y_train = get_train_data(labels_per_class)
    x_test, y_test = get_test_data()

    start = time()
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=1000,  
        callbacks=[
            
            EarlyStopping(monitor='loss', patience=10, verbose=False, mode='min'),
            ReduceLROnPlateau(monitor='loss', factor=0.3, patience=5, verbose=False, mode='min'),
        ],
        verbose=False
    )
    train_acc = history.history['acc'][-1]
    train_time = time() - start

    start = time()
    test_acc = model.evaluate(x_test, y_test, verbose=False)[1]
    test_time = time() - start

    return train_acc, test_acc, train_time, 1000 * test_time / len(y_test)

import matplotlib.pyplot as plt
import numpy as np



def get_cnn_score(train_size):
    return average([
        get_cnn_performance(train_size),
        get_cnn_performance(train_size),
        get_cnn_performance(train_size)
    ])


def get_capsnet_score(train_size):
    return average([
        get_capsnet_performance(train_size),
        get_capsnet_performance(train_size),
        get_capsnet_performance(train_size)
    ])


def average(ps):
   
    q = []
    w = []
    e = []
    r = []
    for p in ps:
        nq, nw, ne, nr = p
        q.append(nq)
        w.append(nw)
        e.append(ne)
        r.append(nr)
    return np.mean(q), np.mean(w), np.mean(e), np.mean(r)


def plot_performace(train_size_powers):
    
    train_size_powers.pop()
    train_size_powers.pop()
    train_size_powers.pop()
    print (train_size_powers)
    cnn_times_test = []
    cnn_times_train = []
    caps_times_test = []
    caps_times_train = []
    cnn_accs_train = []
    cnn_accs_test = []
    caps_accs_train = []
    caps_accs_test = []
    print(
        '{} settings to evaluate. Each next is going to take exponentially longer than previous'
        .format(len(train_size_powers))
    )
    for index, train_size_power in enumerate(train_size_powers):
        train_size = 8 ** train_size_power
        new_cnn_acc_train, new_cnn_acc_test, new_cnn_time_train, new_cnn_time_test = get_cnn_score(train_size)
        print('   ... evaluated case #{} for CNN...'.format(index + 1))
        new_capsnet_acc_train, new_capsnet_acc_test, new_capsnet_time_train, new_capsnet_time_test = get_capsnet_score(train_size)
        print('   ... evaluated case #{} for CapsNet...'.format(index + 1))
        cnn_accs_train.append(new_cnn_acc_train)
        cnn_accs_test.append(new_cnn_acc_test)
        cnn_times_train.append(new_cnn_time_train)
        cnn_times_test.append(new_cnn_time_test)

        caps_accs_train.append(new_capsnet_acc_train)
        caps_accs_test.append(new_capsnet_acc_test)
        caps_times_train.append(new_capsnet_time_train)
        caps_times_test.append(new_capsnet_time_test)
        print('Completed case #{} out of {}'.format(index + 1, len(train_size_powers)))

    print('Raw data:')
    print(caps_accs_train, caps_accs_test, cnn_accs_train, cnn_accs_test)
    print('')
    print(caps_times_test, cnn_times_test)
    print('')
    print(caps_times_train, cnn_times_train)
    print('-----------------')

    plt.plot()
    plt.title('Accuracy')
    plt.plot(train_size_powers, caps_accs_train, 'b', label='CapsNet train', color='green')
    plt.plot(train_size_powers, caps_accs_test, 'c', label='CapsNet test', color='blue')
    plt.plot(train_size_powers, cnn_accs_train, 'r', label='CNN train', color='brown')
    plt.plot(train_size_powers, cnn_accs_test, 'm', label='CNN test', color='red')
    plt.legend()
    plt.xlabel('Training dataset size\n(10 to the power x samples per class)')
    plt.show()

    plt.plot()
    plt.title('Inference time, ms per case')
    plt.plot(train_size_powers, caps_times_test, 'b', label='CapsNet', color='green')
    plt.plot(train_size_powers, cnn_times_test, 'r', label='CNN', color='blue')
    plt.legend()
    plt.xlabel('Training dataset size\n(10 to the power x samples per class)')
    plt.show()

    plt.plot()
    plt.title('Training time, seconds')
    plt.plot(train_size_powers, caps_times_train, 'b', label='CapsNet', color='green')
    plt.plot(train_size_powers, cnn_times_train, 'r', label='CNN', color='blue')
    plt.legend()
    plt.xlabel('Training dataset size\n(10 to the power x samples per class)')

    plt.show()
    plt.close()


plot_performace([v / 2 for v in list(range(0, 7))])

