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
glob_file = "Data/glob_4.pickle"
vocab_file = "Data/vocab.pickle"
use_balance = True
output_format = "leftright"
random.seed()


def dict_load():
    with open(glob_file, 'rb') as load_fp:
        to_return = pickle.load(load_fp)
    return to_return


def unsparse_raw(sparse_labels):
    inclusion_smoothing = 0.5
    base_float = 0.0
    unsparsed = []

    for i in range(len(sparse_labels)):
        label = sparse_labels[i]
        base = [base_float] * 9
        if label > 4:
            if label < 7:
                base[1] = inclusion_smoothing
            if label >= 7:
                base[0] = inclusion_smoothing
            if label == 5 or label == 8:
                base[3] = inclusion_smoothing
            else:
                base[2] = inclusion_smoothing
            base[label] = 1
        elif label != 4:
            if label == 0 or label == 2:
                base[7] = inclusion_smoothing
            if label == 1 or label == 3:
                base[5] = inclusion_smoothing
            if label == 0 or label == 3:
                base[8] = inclusion_smoothing
            if label == 1 or label == 2:
                base[6] = inclusion_smoothing
            base[label] = 1
        else:
            base[label] = 1
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


def create_seq(dropout=0.0, lr_sched=0.0001, dense_size=20, num_layers=1, output_size=1, vocab_s=5000):
    dense_layers = []

    for i in range(num_layers):
        if dropout > 0:
            dense_layers.append(layers.Dropout(dropout))
        dense_layers.append(layers.Dense(dense_size, activation='relu',))

    model_to_return = keras.models.Sequential()
    model_to_return.add(tf.keras.Input(shape=(vocab_s+1,)))

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


def preprocess_tensors(vect_tensors, score_tensors):

    sum_tensor = np.asarray([i if i != 0 else 1 for i in vect_tensors.sum(axis=1)])
    vect_tensors = vect_tensors / sum_tensor[:, None] * 100
    score_tensors = np.asarray([[a] for a in score_tensors])
    finished_tensors = np.append(vect_tensors, score_tensors, axis=1)

    return finished_tensors


def unsparse(labels):
    labels = np.asarray(labels)
    if max(labels) == 3:
        unsparsed = np.asarray(unsparse_four_axis(labels))
    elif max(labels) > 3:
        unsparsed = np.asarray(unsparse_raw(labels))
    else:
        unsparsed = labels

    return unsparsed


def init_vectorize_layer(text, max_tokens, vocab_s):
    init_t = time.time()
    if exists(vocab_file):
        with open(vocab_file, 'rb') as load_fp:
            vocab = pickle.load(load_fp)
        vect_layer = TextVectorization(output_mode='count', ngrams=3)
        vect_layer.set_vocabulary(vocab[:vocab_s])
        print("Loaded cached vocabulary in {} seconds".format(delta_time(init_t)))
    else:
        vect_layer = TextVectorization(max_tokens=max_tokens, output_mode='count', ngrams=3)
        vect_layer.adapt(text[:15000])
        with open(vocab_file, 'wb') as write_fp:
            pickle.dump(vect_layer.get_vocabulary(), write_fp)
        vect_layer.set_vocabulary(vect_layer.get_vocabulary()[:vocab_s])
        print("Generated new vocabulary in {} seconds".format(delta_time(init_t)))
    return vect_layer


def create_validation_data(split, tensor_dict):
    labels = tensor_dict['labels']
    prepped = tensor_dict['tensors']
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
    tensors = data_dict['tensors']
    labels = data_dict['labels']
    new_tensors = []
    new_labels = []
    for index, label in enumerate(labels):
        new_label = None
        if label in [0, 7, 8]:
            new_label = 0
        if label in [1, 5, 6]:
            new_label = 1
        if new_label is not None:
            new_labels.append(new_label)
            new_tensors.append(tensors[index])
    return {'labels': new_labels, 'tensors': new_tensors}


def raw_to_four_axis(data_dict):
    tensors = data_dict['tensors']
    labels = data_dict['labels']
    new_tensors = []
    new_labels = []
    for index, label in enumerate(labels):
        if label in [5, 6, 7, 8]:
            new_labels.append(label - 5)
            new_tensors.append(tensors[index])
    return {'labels': new_labels, 'tensors': new_tensors}


def print_dist(labels):
    counts = dict()
    for label in labels:
        if str(label) in counts.keys():
            counts[str(label)] += 1
        else:
            counts[str(label)] = 1
    print(list(counts.values()))


def balance_dict(data_dict):
    labels = data_dict['labels']
    tensors = data_dict['tensors']
    counts = dict()
    new_labels = []
    new_tensors = []
    for label in labels:
        label_key = str(label)
        if label_key in counts.keys():
            counts[label_key] += 1
        else:
            counts[label_key] = 1
    min_c = min(counts.values())
    for i in counts.keys():
        counts[i] = 0
    for i, label in enumerate(labels):
        label_key = str(label)
        if counts[label_key] < min_c:
            new_labels.append(label)
            new_tensors.append(tensors[i])
            counts[label_key] += 1
    return {'tensors': new_tensors, 'labels': new_labels}


if __name__ == '__main__':
    init_time = time.time()
    raw_dict = dict_load()
    print("Loaded data in {} seconds.".format(delta_time(init_time)))

    loss = "categorical"
    monitor = "val_categorical_crossentropy"

    vocab_size = 7500
    vectorize_layer = init_vectorize_layer(raw_dict['text'], 20000, vocab_size)

    init_time = time.time()
    if exists(cached_dict_file):
        with open(cached_dict_file, 'rb') as dict_fp:
            cache_dict = pickle.load(dict_fp)
    else:
        vectorized_tensors = np.asarray(tf.cast(vectorize_layer(raw_dict['text']), tf.float64))
        prepped_tensors = preprocess_tensors(vectorized_tensors, raw_dict['scores'])
        cache_dict = {'tensors': prepped_tensors, 'labels': raw_dict['labels']}
        with open(cached_dict_file, 'wb') as dict_fp:
            pickle.dump(cache_dict, dict_fp)
    print("Loaded cached tensors in {} seconds.".format(delta_time(init_time)))
    # if you change anything before this point, the cached_tensor.pickle file must be deleted for changes to take effect
    # this incurs a large delay to vectorize the tensors again (1-5min)

    init_time = time.time()
    dicts = {'leftright': raw_to_left_right(cache_dict),
             'fouraxis': raw_to_four_axis(cache_dict),
             'raw': cache_dict}
    for key in dicts.keys():
        dicts[key]['labels'] = unsparse(dicts[key]['labels'])

    if output_format == "leftright":
        loss = "binary"
        monitor = "val_binary_crossentropy"
        final_layer_size = 1
    elif output_format == "fouraxis":
        final_layer_size = 2
    else:
        final_layer_size = 9

    active_dict = dicts[output_format]

    if use_balance:
        active_dict = balance_dict(active_dict)

    print_dist(active_dict['labels'])
    train_tensors, val_tensors, train_labels, val_labels = create_validation_data(0.2, active_dict)
    print(f"Tensors prepared in {delta_time(init_time)} seconds.")
    init_time = time.time()
    if len(train_tensors) != len(train_labels) or len(val_labels) != len(val_tensors):
        print("Tensor size does not match label size. Exiting.")
        exit()
    print(f"{len(train_labels)} training points.")
    print(f"{len(val_labels)} validation points.")

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

    model = create_seq(lr_sched=lr_schedule, dense_size=1000, dropout=.35,
                       output_size=final_layer_size, vocab_s=vocab_size)

    print("Model created in {} seconds".format(delta_time(init_time)))

    history = model.fit(x=train_tensors,
                        y=train_labels,
                        epochs=1000,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=(val_tensors, val_labels),
                        callbacks=tf.keras.callbacks.EarlyStopping(monitor=monitor,
                                                                   patience=150))

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
