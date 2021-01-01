"""Shakespere dataset"""

import os
import typing

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


class DatasetConfig(typing.NamedTuple):
    """Dataset Configurations"""
    seq_length: int = 100
    batch_size: int = 64
    suffle_buffer: int = 1000


def get_dataset(cache_dir=None, conf=DatasetConfig()):
    """Create dataset

    Args:
        cache_dir: a cache directory path to store
            (or pass if it is already exist) dataset file
        conf: a DatasetConfig instance

    Returns:
        a dataset which contains two tensors
        The first is 'input' tensor, and the second is 'target' tensor.
        Both of them has (batch, seq_len) size
    """
    corpus_path = _download_corpus(cache_dir)

    text = ''
    with open(corpus_path, 'rb') as file:
        text = file.read().decode(encoding='utf-8')

    # TODO(jseo): use tfds text encoder
    dictionaries = _make_dictionary(text)
    encoded = _encode(text, dictionaries[0])

    # make a dataset: (1)
    dataset = tf.data.Dataset.from_tensor_slices(encoded)
    # make sequence: (seq_len+1)
    dataset = dataset.batch(conf.seq_length + 1, drop_remainder=True)

    def split(data):
        return data[:-1], data[1:]

    # split input and target: (seq_len), (seq_len)
    dataset = dataset.map(split)

    # batching: (batch, seq_len), (batch, seq_len)
    dataset = dataset.shuffle(conf.suffle_buffer).batch(conf.batch_size,
                                                        drop_remainder=True)
    return dataset
