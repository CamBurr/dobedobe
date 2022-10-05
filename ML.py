from os.path import exists
import os
import pickle
import tensorflow as tf
from keras.layers import TextVectorization
from tensorflow import keras
import keras.layers as layers
import keras.losses as losses
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time
import sklearn
from scikeras.wrappers import KerasRegressor
import sklearn.model_selection

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

cached_vectorized_file = "Data/cached_tensor.pickle"
glob_file = "Data/glob_3.pickle"
vocab_file = "Data/vocab.pickle"



def dict_load():
    with open(glob_file, 'rb') as fp:
        to_return = pickle.load(fp)
    return to_return


def unsparse_labels(sparse_labels):
    split = 0.15
    centrist = 0
    unsparsed = []

    for i in range(len(sparse_labels)):
        label = sparse_labels[i]
        base = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        if label > 4:
            if label < 7:
                base[1] = split
            if label >= 7:
                base[0] = split
            if label == 5 or label == 8:
                base[3] = split
            else:
                base[2] = split
            base[4] = centrist
            base[label] = 1 - (split * 2) - centrist
        elif label != 4:
            if label == 0 or label == 2:
                base[7] = split
            if label == 1 or label == 3:
                base[5] = split
            if label == 0 or label == 3:
                base[8] = split
            if label == 1 or label == 2:
                base[6] = split
            base[4] = centrist
            base[label] = 1 - (split * 2) - centrist
        else:
            base[label] = 1
        unsparsed.append(base)
    return unsparsed


def write_file(file, obj):
    with open(file, 'wb') as fp:
        pickle.dump(obj, fp)


def create_seq(dropout=0.25, lr_sched=0.0001, dense_size=20, num_layers=1):

    dense_layers = []

    for i in range(num_layers):
        if dropout > 0:
            dense_layers.append(layers.Dropout(dropout))
        dense_layers.append(layers.Dense(dense_size, activation='elu',))

    model_to_return = keras.models.Sequential()

    model_to_return.add(tf.keras.Input(shape=(5000,)))

    for layer in dense_layers:
        model_to_return.add(layer)

    model_to_return.add(layers.Dense(9, activation='softmax'))

    accuracy = 'categorical_accuracy'

    optimizer = keras.optimizers.Adam(learning_rate=lr_sched)
    loss = losses.CategoricalCrossentropy(from_logits=False)

    model_to_return.compile(optimizer=optimizer,
                            loss=loss,
                            metrics=[accuracy, keras.metrics.CategoricalCrossentropy()])

    return model_to_return


def preprocess_tensors(vect_tensors, score_tensors, labels):
    labels = np.asarray(labels)
    unsparsed = np.asarray(unsparse_labels(labels))

    sum_tensor = np.asarray([i if i != 0 else 1 for i in vect_tensors.sum(axis=1)])
    vect_tensors = vect_tensors / sum_tensor[:, None] * 100
    score_tensors = np.asarray([[a] for a in score_tensors])
    prepped_tensors = np.append(vect_tensors, score_tensors, axis=1)

    dict_to_return = {'prepped_tensors': prepped_tensors, 'unsparsed_labels': unsparsed, 'labels': labels}

    return dict_to_return


def create_vectorize_layer(text, max_tokens, vocab_s):
    if exists(vocab_file):
        with open(vocab_file, 'rb') as fp:
            vocab = pickle.load(fp)
        vect_layer = layers.TextVectorization(output_mode='count')
        vect_layer.set_vocabulary(vocab[2:vocab_s])
    else:
        vect_layer = layers.TextVectorization(max_tokens=max_tokens, output_mode='count')
        vect_layer.adapt(text[:30000])
        with open(vocab_file, 'wb') as fp:
            pickle.dump(vect_layer.get_vocabulary(), fp)
        vect_layer.set_vocabulary(vect_layer.get_vocabulary()[:vocab_s])

    return vect_layer


def create_validation_data(split, tensor_dict):
    labels = tensor_dict['unsparsed_labels']
    prepped = tensor_dict['prepped_tensors']

    split_index = round(len(labels) * (1 - split))

    # train_tensors, val_tensors, train_labels, val_labels

    return prepped[:split_index], prepped[split_index:], labels[:split_index], labels[split_index:]


def delta_time(t):
    return round(time.time() - t, 2)


if __name__ == '__main__':
    init_time = time.time()

    raw_dict = dict_load()

    vocab_size = 5000

    vectorize_layer = create_vectorize_layer(raw_dict['text'], 20000, vocab_size)

    if exists(cached_vectorized_file):
        with open(cached_vectorized_file, 'rb') as fp:
            vectorized_tensors = pickle.load(fp)
        print("Loaded cached tensors in {} seconds".format(delta_time(init_time)))
    else:
        vectorized_tensors = np.asarray(tf.cast(vectorize_layer(raw_dict['text']), tf.float64))
        with open(cached_vectorized_file, 'wb') as fp:
            pickle.dump(vectorized_tensors, fp)
        print("Vectorized tensors in {} seconds".format(delta_time(init_time)))
    init_time = time.time()

    # if you change anything before this point, the cached_tensor.pickle file must be deleted for changes to take effect

    prepped_dict = preprocess_tensors(vectorized_tensors, raw_dict['scores'], raw_dict['labels'])

    print(vectorized_tensors.shape)

    train_tensors, val_tensors, train_labels, val_labels = create_validation_data(0.2, prepped_dict)

    print("Tensors prepped and split in {} seconds".format(delta_time(init_time)))
    init_time = time.time()

    old_model = None
    if exists("current_model"):
        print("Loading saved model.")
        old_model = keras.models.load_model('current_model')

    batch_size = 256
    steps_per_epoch = len(train_tensors) // batch_size
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        0.00001,
        decay_steps=steps_per_epoch * 1000,
        decay_rate=.75,
        staircase=False)

    model = create_seq(lr_sched=lr_schedule)

    scikit_model = KerasRegressor(model=create_seq, dense_size=9, num_layers=1, dropout=0)

    print("Model created in {} seconds".format(delta_time(init_time)))

    history = model.fit(x=train_tensors,
                        y=train_labels,
                        epochs=10000,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=(val_tensors, val_labels),
                        callbacks=tf.keras.callbacks.EarlyStopping(monitor='val_categorical_crossentropy',
                                                                   patience=200))

    new_metrics = model.evaluate(val_tensors, val_labels)
    if old_model:
        old_metrics = old_model.evaluate(val_tensors, val_labels)

        print("Loaded model accuracy: " + str(old_metrics[1]))
        print("New model accuracy: " + str(new_metrics[1]))

        if old_metrics[1] < new_metrics[1]:
            model.save('current_model')
    else:
        model.save('current_model')

    model.summary()

    param_grid = {"dense_size": [25], "num_layers": [1, 2], "dropout": [.25]}
    print(train_labels[:10])
    grid = sklearn.model_selection.GridSearchCV(estimator=scikit_model, param_grid=param_grid,
                                                n_jobs=-1, cv=2, verbose=2)
    grid_result = grid.fit(train_tensors, train_labels, verbose=0,
                           epochs=50,
                           steps_per_epoch=steps_per_epoch,
                           validation_data=(val_tensors, val_labels),
                           callbacks=tf.keras.callbacks.EarlyStopping(monitor='val_categorical_crossentropy',
                                                                      patience=200))

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
