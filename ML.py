from os.path import exists
import os
from typing import List, Any

import praw
import pickle
import tensorflow as tf
from keras.layers import TextVectorization
from tensorflow import keras
import keras.layers as layers
import keras.losses as losses
import tensorflow_hub as hub
import numpy as np
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

sample_tensor = [[1, [2, 3]],
                 [4, [5, 6]],
                 [7, [8, 9]]]

mid_tensor = [[1, 2],
              [3, 4],
              [5, 6]]

real_tensor = tf.convert_to_tensor([[1, 2, 3, 4, 5],
                                            [6, 7, 8, 9, 0],
                                            [1, 2, 3, 4, 5]])

to_add = [1, 2, 3]

np_tensor = real_tensor.numpy()

np_tensor = np.insert(np_tensor, 0, to_add, axis=1)

print(np_tensor)


final_tensor = [.1, .2, .3]

sample_model = ["Preprocess",
                "Dense",
                "Output"]


def cache_dict(files):
    for key in files:
        with open(key + '.pickle', 'wb') as handle:
            pickle.dump(files[key], handle)


def dict_load():
    variables = ['users', 'text', 'scores', 'labels']
    to_return = dict()
    for i in variables:
        with open(i + ".pickle", 'rb') as handle:
            to_return[i] = pickle.load(handle)
    return to_return


def user_loop(session):
    if exists('users.pickle'):
        tensors = dict_load()
        text_tensors = tensors['text']
        score_tensors = tensors['scores']
        users = tensors['users']
        labels = tensors['labels']
    else:
        text_tensors = []
        score_tensors = []
        labels = []
        users = set()

    subreddit = session.subreddit('politicalcompassmemes')

    new_users = set()
    new_labels = []
    new_text_tensors = []
    new_score_tensors = []

    for comment in subreddit.stream.comments():
        if comment.author not in users and comment.author not in new_users:
            if comment.author_flair_text is None:
                continue
            else:
                if 'Right' in comment.author_flair_text:
                    new_labels.append(1)
                elif 'Left' in comment.author_flair_text:
                    new_labels.append(-1)
                elif any(flair in comment.author_flair_text for flair in ['Center', 'Centrist']):
                    new_labels.append(0)

            user = comment.author
            text_tensor = []
            score_tensor = []
            for i in user.comments.new(limit=25):
                text = ''.join(e for e in i.body if (e.isalnum() or e == ' '))
                text_tensor.append(text)
                score_tensor.append(i.score)
            text_tensor += ' ' * (25 - len(text_tensor))
            score_tensor.extend([0] * (25 - len(score_tensor)))
            new_score_tensors.append(score_tensor)
            new_text_tensors.append(text_tensor)
            new_users.add(user)

            if len(new_users) % 25 == 0:
                print(len(users))
                text_tensors.extend(new_text_tensors)
                score_tensors.extend(new_score_tensors)
                users.update(new_users)
                labels.extend(new_labels)

                new_tensors = []
                new_users = set()
                new_score_tensors = []
                new_text_tensors = []
                new_labels = []

            cache_dict({'users': users, 'text': text_tensors, 'scores': score_tensors, 'labels': labels})

        if len(users) >= 5000:
            return {'users': users, 'text': text_tensors, 'scores': score_tensors, 'labels': labels}


def preprocessing(tensors):
    time_a = time.time_ns()
    text_tensors = tensors['text']
    score_tensors = tensors['scores']
    labels = np.asarray([x+1 for x in tensors['labels']])
    vocab_size = 2000

    vectorize_layer: TextVectorization = layers.TextVectorization(max_tokens=10000, output_mode='count')
    flat_text = []
    for i in text_tensors:
        flat_text.extend(i)

    if exists('vocab.pickle'):
        with open('vocab.pickle', 'rb') as handle:
            vocab = pickle.load(handle)
            vectorize_layer.set_vocabulary(vocab[0:vocab_size])
    else:
        vectorize_layer.adapt(flat_text)
        print(vectorize_layer.get_vocabulary(include_special_tokens=True))
        with open('vocab.pickle', 'wb') as handle:
            pickle.dump(vectorize_layer.get_vocabulary(), handle)

    nested_text = text_tensors

    for i in range(len(text_tensors)):
        for j in range(len(text_tensors[0])):
            nested_text[i][j] = [text_tensors[i][j]]

    score_tensors = np.asarray(score_tensors)
    #full_tensor = tf.convert_to_tensor(np.asarray(nested_text))
    #print(full_tensor)
    print(time.time_ns() - time_a)

    vec_split = np.vectorize(str.split)

    splitter = lambda t: t.split()

    flat_array: list[list[Any]] = []

    for sublist_1 in nested_text:
        flat_sublist = []
        for sublist_2 in sublist_1:
            flat_sublist.append(sublist_2[0])
        flat_array.append(' '.join(flat_sublist))

    print(type(flat_array))
    sigendian = -vocab_size + vocab_size

    text_tensors = tf.convert_to_tensor(flat_array)
    text_tensors = np.asarray(tf.cast(vectorize_layer(flat_array), tf.float64))[19:, 1:]
    text_tensors = text_tensors / text_tensors.sum(axis=1)[:, None] * 100
    score_tensors = np.asarray(score_tensors)
    print(score_tensors.mean(axis=1, keepdims=True))
    text_tensors = np.append(text_tensors, score_tensors.mean(axis=1, keepdims=True)[19:, :], axis=1)[:, sigendian:]

    print(text_tensors.shape)
    print(score_tensors.mean(axis=1).shape)

    batch = 256

    class_model = keras.Sequential([
        layers.Dropout(.1),
        layers.Dense(2000, activation='relu'),
        layers.Dropout(.3),
        layers.Dense(1000),
        layers.Dropout(.3),
        layers.Dense(500),
        layers.Dropout(.3),
        layers.Dense(100),
        layers.Dropout(.3),
        layers.Dense(3, activation='softmax'),
    ])

    reg_model = keras.Sequential([
        layers.GaussianNoise(.1),
        layers.Dropout(.1),
        layers.Dense(1000, activation='relu', use_bias=False),
        layers.Dropout(.1),
        layers.Dense(100, use_bias=False),
        layers.Dropout(.1),
        layers.Dense(10, use_bias=False),
        layers.Dense(1),
    ])

    np.set_printoptions(suppress=True)

    nested_labels = []

    for label in labels:
        nested_labels.append([label])

    class_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.000005), loss=losses.SparseCategoricalCrossentropy(from_logits=False), metrics='accuracy')
    reg_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss=losses.MeanSquaredError(), metrics='accuracy')
    class_model.fit(x=text_tensors, y=labels[19:], epochs=40, batch_size=batch, validation_split=.05)
    reg_model.fit(x=text_tensors, y=labels[19:], epochs=40, batch_size=batch, validation_split=.05)
    print(np.append(class_model.predict(text_tensors[:19, sigendian:]), nested_labels[:19], axis=1))
    print(np.append(reg_model.predict(text_tensors[:19, sigendian:]), nested_labels[:19], axis=1))

    '''class_pred = class_model.predict(text_tensors[19:, sigendian:])
    reg_pred = reg_model.predict(text_tensors[19:, sigendian:])
    combined_pred = np.append(class_pred, reg_pred, axis=1)
    combined_val = np.append(class_model.predict(text_tensors[:19, sigendian:]), reg_model.predict(text_tensors[:19, sigendian:]), axis=1)
    final_model = keras.Sequential([
        layers.Dense(50),
        layers.Dense(1)
    ])
    final_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
                        loss=losses.MeanSquaredError(), metrics='accuracy')
    final_model.fit(x=combined_pred, y=labels[19:], epochs=100, batch_size=batch, validation_split=.05)
    print(np.append(final_model.predict(combined_val[:, sigendian:]), nested_labels[:19], axis=1))'''

if __name__ == '__main__':
    reddit = praw.Reddit("DEFAULT")
    preprocessing(user_loop(reddit))