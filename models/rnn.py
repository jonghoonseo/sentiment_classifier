"""Simple RNN Model"""

import typing

import tensorflow as tf


class ModelConfig(typing.NamedTuple):
    vocab_size: int = 1000
    embedding_dim: int = 256
    rnn_units: int = 1024
    batch_size: int = 64


def build_model(conf: ModelConfig) -> tf.keras.Model:
    """Build a model

    Args: a ModelConfig instance to configure model structure

    Returns: a model created
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(conf.vocab_size,
                                  conf.embedding_dim,
                                  batch_input_shape=[conf.batch_size, None]),
        tf.keras.layers.GRU(conf.rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(conf.vocab_size),
    ])
    return model
