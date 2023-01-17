from os.path import exists
import os
import pickle
import tensorflow as tf
from keras.layers import TextVectorization
from tensorflow import keras
import keras.layers as layers
import keras.losses as losses
import numpy as np
import time
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
cached_dict_file = "Data/cached_dict.pickle"
cached_balanced_file = "Data/cached_balanced_dict.pickle"
glob_file = "Data/glob_3.pickle"
vocab_file = "Data/vocab.pickle"
use_balance = True
output_format = "fouraxis"
random.seed()


def dict_load():
    with open(glob_file, 'rb') as load_fp:
        to_return = pickle.load(load_fp)
    return to_return


def unsparse_raw(sparse_labels):
    split = 0
    base_float = 0
    unsparsed = []

    for i in range(len(sparse_labels)):
        label = sparse_labels[i]
        base = [base_float] * 9
        if label > 4:
            if label < 7:
                base[1] = split
            if label >= 7:
                base[0] = split
            if label == 5 or label == 8:
                base[3] = split
            else:
                base[2] = split
            base[label] = 1 - (split * 2) - (base_float * 6)
        elif label != 4:
            if label == 0 or label == 2:
                base[7] = split
            if label == 1 or label == 3:
                base[5] = split
            if label == 0 or label == 3:
                base[8] = split
            if label == 1 or label == 2:
                base[6] = split
            base[label] = 1 - (split * 2) - (base_float * 6)
        else:
            base[label] = 1 - base_float * 8
        unsparsed.append(base)
    return unsparsed


def unsparse_four_axis(sparse_labels):
    unsparsed_labels = []
    for i, label in enumerate(sparse_labels):
        new_label = []
        if label < 2:
            new_label.append(1)
        else:
            new_label.append(0)
        if label % 2 == 0:
            new_label.append(1)
        else:
            new_label.append(0)
        unsparsed_labels.append(new_label)
    return unsparsed_labels


def write_file(file, obj):
    with open(file, 'wb') as write_fp:
        pickle.dump(obj, write_fp)


def create_seq(dropout=0.0, lr_sched=0.0001, dense_size=20, num_layers=1, output_size=1):
    dense_layers = []

    for i in range(num_layers):
        if dropout > 0:
            dense_layers.append(layers.Dropout(dropout))
        dense_layers.append(layers.Dense(dense_size, activation='elu',))

    model_to_return = keras.models.Sequential()
    model_to_return.add(tf.keras.Input(shape=(5001,)))

    for layer in dense_layers:
        model_to_return.add(layer)

    accuracy = 'binary_accuracy'
    model_loss = losses.BinaryCrossentropy()
    metric = keras.metrics.BinaryCrossentropy()
    if output_size > 1:
        accuracy = 'categorical_accuracy'
        model_loss = losses.CategoricalCrossentropy()
        metric = keras.metrics.CategoricalCrossentropy()

    model_to_return.add(layers.Dense(output_size, activation='sigmoid'))
    optimizer = keras.optimizers.Adam(learning_rate=lr_sched)
    model_to_return.compile(optimizer=optimizer,
                            loss=model_loss,
                            metrics=[accuracy, metric])

    return model_to_return


def preprocess_tensors(vect_tensors, score_tensors, labels):
    labels = np.asarray(labels)
    if max(labels) == 3:
        unsparsed = np.asarray(unsparse_four_axis(labels))
    elif max(labels) > 3:
        unsparsed = np.asarray(unsparse_raw(labels))
    else:
        unsparsed = None

    sum_tensor = np.asarray([i if i != 0 else 1 for i in vect_tensors.sum(axis=1)])
    vect_tensors = vect_tensors / sum_tensor[:, None] * 100
    score_tensors = np.asarray([[a] for a in score_tensors])
    prepped_tensors = np.append(vect_tensors, score_tensors, axis=1)

    dict_to_return = {'prepped_tensors': prepped_tensors, 'prepped_labels': unsparsed, 'labels': labels}

    return dict_to_return


def create_vectorize_layer(text, max_tokens, vocab_s):
    init_t = time.time()
    if exists(vocab_file):
        with open(vocab_file, 'rb') as load_fp:
            vocab = pickle.load(load_fp)
        vect_layer = TextVectorization(output_mode='count')
        vect_layer.set_vocabulary(vocab[:vocab_s])
        print("Loaded cached vocabulary in {} seconds".format(delta_time(init_t)))
    else:
        vect_layer = TextVectorization(max_tokens=max_tokens, output_mode='count')
        vect_layer.adapt(text[:15000])
        with open(vocab_file, 'wb') as write_fp:
            pickle.dump(vect_layer.get_vocabulary(), write_fp)
        vect_layer.set_vocabulary(vect_layer.get_vocabulary()[:vocab_s])
        print("Generated new vocabulary in {} seconds".format(delta_time(init_t)))
    return vect_layer


def create_validation_data(split, tensor_dict):
    if tensor_dict['prepped_labels'] is None:
        labels = tensor_dict['labels']
    else:
        labels = tensor_dict['prepped_labels']
    prepped = tensor_dict['prepped_tensors']
    total = len(prepped)
    train_size = round(total * (1-split))
    val_size = total - train_size
    train_t = list(prepped)
    train_l = list(labels)
    val_t = []
    val_l = []
    indices = sorted(random.sample(range(total), val_size), reverse=True)
    for i in indices:
        val_t.append(prepped[i])
        val_l.append(labels[i])
        train_t.pop(i)
        train_l.pop(i)
    train_t = np.asarray(train_t)
    train_l = np.asarray(train_l)
    val_t = np.asarray(val_t)
    val_l = np.asfarray(val_l)
    # train_tensors, val_tensors, train_labels, val_labels
    return train_t, val_t, train_l, val_l


def delta_time(t):
    return round(time.time() - t, 2)


def raw_to_left_right(data_dict):
    text = data_dict['text']
    labels = data_dict['labels']
    scores = data_dict['scores']
    new_text = []
    new_labels = []
    new_scores = []
    for index, label in enumerate(labels):
        new_label = None
        if label in [0, 7, 8]:
            new_label = 0
        if label in [1, 5, 6]:
            new_label = 1
        if new_label is not None:
            new_labels.append(new_label)
            new_text.append(text[index])
            new_scores.append(scores[index])
    return {'text': new_text, 'labels': new_labels, 'scores': new_scores}


def raw_to_four_axis(data_dict):
    text = data_dict['text']
    labels = data_dict['labels']
    scores = data_dict['scores']
    new_text = []
    new_labels = []
    new_scores = []
    for index, label in enumerate(labels):
        if label in [5, 6, 7, 8]:
            new_labels.append(label - 5)
            new_text.append(text[index])
            new_scores.append(scores[index])
    return {'text': new_text, 'labels': new_labels, 'scores': new_scores}


def balance_dict(data_dict):
    labels = data_dict['labels']
    text = data_dict['text']
    scores = data_dict['scores']
    counts = [0] * (max(labels) + 1)
    new_labels = []
    new_text = []
    new_scores = []
    for label in labels:
        counts[label] += 1
    print(counts)
    min_c = min(counts)
    for i in range(len(counts)):
        counts[i] = 0
    for i, label in enumerate(labels):
        if counts[label] < min_c:
            new_labels.append(label)
            new_text.append(text[i])
            new_scores.append(scores[i])
            counts[label] += 1
    print(counts)
    return {'text': new_text, 'labels': new_labels, 'scores': new_scores}


if __name__ == '__main__':
    init_time = time.time()
    raw_dict = dict_load()
    print("Loaded data in {} seconds.".format(delta_time(init_time)))
    init_time = time.time()
    left_right_dict = raw_to_left_right(raw_dict)
    four_axis_dict = raw_to_four_axis(raw_dict)

    loss = "categorical"
    monitor = "val_categorical_crossentropy"

    if output_format == "leftright":
        loss = "binary"
        monitor = "val_binary_crossentropy"
        active_dict = left_right_dict
        final_layer_size = 1
    elif output_format == "fouraxis":
        active_dict = four_axis_dict
        final_layer_size = 2
    else:
        active_dict = raw_dict
        final_layer_size = 9

    if use_balance:
        active_dict = balance_dict(active_dict)

    vocab_size = 5000
    vectorize_layer = create_vectorize_layer(active_dict['text'], 20000, vocab_size)

    init_time = time.time()
    if exists(cached_dict_file):
        with open(cached_dict_file, 'rb') as dict_fp:
            prepped_dict = pickle.load(dict_fp)
        print("Loaded cached tensors in {} seconds.".format(delta_time(init_time)))
    else:
        vectorized_tensors = np.asarray(tf.cast(vectorize_layer(active_dict['text']), tf.float64))
        prepped_dict = preprocess_tensors(vectorized_tensors, active_dict['scores'], active_dict['labels'])
        with open(cached_dict_file, 'wb') as dict_fp:
            pickle.dump(prepped_dict, dict_fp)
        print("Vectorized and prepped tensors in {} seconds.".format(delta_time(init_time)))
    # if you change anything before this point, the cached_tensor.pickle file must be deleted for changes to take effect
    # this incurs a large delay to vectorize the tensors again (1-5min)
    print(type(prepped_dict))

    init_time = time.time()
    train_tensors, val_tensors, train_labels, val_labels = create_validation_data(0.2, prepped_dict)
    print("Tensors split in {} seconds.".format(delta_time(init_time)))
    init_time = time.time()
    if len(train_tensors) != len(train_labels) or len(val_labels) != len(val_tensors):
        print("Tensor size does not match label size. Exiting.")
        exit()
    print("{} training points.".format(len(train_labels)))
    print("{} validation points.".format(len(val_labels)))

    old_model = None
    compare_models = False
    if exists("current_model") and compare_models:
        print("Loading saved model.")
        old_model = keras.models.load_model('current_model')

    batch_size = 256
    steps_per_epoch = len(train_tensors) // batch_size
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        0.00001,
        decay_steps=steps_per_epoch * 1000,
        decay_rate=.75,
        staircase=False)

    model = create_seq(lr_sched=lr_schedule, dense_size=1000, dropout=.35, output_size=final_layer_size)

    print("Model created in {} seconds".format(delta_time(init_time)))

    history = model.fit(x=train_tensors,
                        y=train_labels,
                        epochs=500,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=(val_tensors, val_labels),
                        callbacks=tf.keras.callbacks.EarlyStopping(monitor=monitor,
                                                                   patience=100))

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
    exit()
