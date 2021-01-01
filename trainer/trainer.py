"""Train a model"""

import typing

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

from dataset import shakespeare
from models import rnn

FLAGS = flags.FLAGS
flags.DEFINE_enum('mode', 'train', ['train', 'predict'], 'Mode')
flags.DEFINE_string('model_dir', None, 'Where to save model checkpoints')


class TrainerConfig(typing.NamedTuple):
    """Trainer configurations"""
    model_dir: str
    epochs: int = 100


def _loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels,
                                                           logits,
                                                           from_logits=True)


def _train(dataset_conf: shakespeare.DatasetConfig,
           model_conf: rnn.ModelConfig, train_conf: TrainerConfig):
    dataset = shakespeare.get_dataset(dataset_conf)
    model = rnn.build_model(model_conf)

    checkpoint_prefix = train_conf.model_dir
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix, save_weights_only=True)

    model.compile(optimizer='adam', loss=_loss)

    history = model.fit(dataset,
                        epochs=train_conf.epochs,
                        callbacks=[checkpoint_callback])

    logging.info('Train done')
    logging.info('history: \n%s', history)


def _predict(dataset_conf: shakespeare.DatasetConfig,
             model_conf: rnn.ModelConfig):
    pass


def main(args):
    """Main function"""
    del args  # unused

    dataset_conf = shakespeare.DatasetConfig()
    model_conf = rnn.ModelConfig()
    train_conf = TrainerConfig(model_dir=FLAGS.model_dir)
    if FLAGS.mode == 'train':
        _train(dataset_conf, model_conf, train_conf)
    else:
        _predict(dataset_conf, model_conf)


if __name__ == '__main__':
    flags.mark_flag_as_required('model_dir')
    app.run(main)
