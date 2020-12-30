"""Shakespere dataset"""

import os

import numpy as np
import tensorflow as tf


def _download_corpus(cache_dir=None):
    download_path = ('https://storage.googleapis.com/download.tensorflow.org'
                     '/data/shakespeare.txt')
    corpus_path = tf.keras.utils.get_file(os.path.basename(download_path),
                                          download_path,
                                          cache_dir=cache_dir)
    return corpus_path


def _make_dictionary(text):
    dictionary = sorted(set(text))
    return {char: index
            for index, char in enumerate(dictionary)}, np.array(dictionary)


def _encode(char_array, dictionary: dict):
    return [dictionary[char] for char in char_array]


def get_dataset(cache_dir=None, seq_length=100):
    corpus_path = _download_corpus(cache_dir)

    text = ''
    with open(corpus_path, 'rb') as file:
        text = file.read().decode(encoding='utf-8')

    # TODO(jseo): use tfds text encoder
    dictionaries = _make_dictionary(text)
    encoded = _encode(text, dictionaries[0])

    # (1)
    dataset = tf.data.Dataset.from_tensor_slices(encoded)
    for i in dataset.take(5):
        print(i.numpy())

    #
