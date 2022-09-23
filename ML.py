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

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

glob_file = "Data/glob.pickle"
vocab_file = "Data/vocab.pickle"


def dict_load():
    with open(glob_file, 'rb') as fp:
        to_return = pickle.load(fp)
    return to_return


def preprocessing(tensors):
    text_tensors = tensors['text']
    score_tensors = tensors['scores']
    labels = np.asarray(tensors['labels'])
    vocab_size = 3000

    vectorize_layer: TextVectorization = layers.TextVectorization(max_tokens=10000, output_mode='count')
    flat_text = []
    for i in text_tensors:
        flat_text.extend(i)

    if exists(vocab_file):
        with open(vocab_file, 'rb') as handle:
            vocab = pickle.load(handle)
            vectorize_layer.set_vocabulary(vocab[0:vocab_size])
    else:
        vectorize_layer.adapt(flat_text)
        print(vectorize_layer.get_vocabulary(include_special_tokens=True))
        with open(vocab_file, 'wb') as handle:
            pickle.dump(vectorize_layer.get_vocabulary(), handle)

    sigendian = -vocab_size + vocab_size

    text_tensors = np.asarray(tf.cast(vectorize_layer(text_tensors), tf.float64))[19:, :]
    sum_tensor = np.asarray([i if i != 0 else 1 for i in text_tensors.sum(axis=1)])
    text_tensors = text_tensors / sum_tensor[:, None] * 100
    score_tensors = np.asarray([[a] for a in score_tensors])[19:]
    text_tensors = np.append(text_tensors, score_tensors, axis=1)[:, sigendian:]

    batch = 512

    class_model = keras.Sequential([
        layers.Dropout(.3),
        layers.Dense(2000, activation='relu', use_bias=False),
        layers.Dropout(.3),
        layers.Dense(1000),
        layers.Dropout(.3),
        layers.Dense(500),
        layers.Dropout(.3),
        layers.Dense(100),
        layers.Dropout(.3),
        layers.Dense(9)
    ])

    np.set_printoptions(suppress=True)

    nested_labels = []

    for label in labels:
        nested_labels.append([label])

    class_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.00001),
                        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                        metrics='sparse_categorical_accuracy')
    class_model.fit(x=text_tensors, y=labels[19:], epochs=40, batch_size=batch, validation_split=.05)
    print(np.append(class_model.predict(text_tensors[:19, sigendian:]), nested_labels[:19], axis=1))


if __name__ == '__main__':
    glob = dict_load()
    preprocessing(glob)
