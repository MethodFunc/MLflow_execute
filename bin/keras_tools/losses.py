from enum import Enum

import tensorflow as tf


class LossType(str, Enum):
    MSE = 'mse'
    MAE = 'mae'
    HUBER = 'huber'
    CATEGORICAL = 'classification'


def get_loss_fn(config, log):
    loss_name = config['train_setting']['loss']

    losses = {
        LossType.MSE: tf.losses.mean_squared_error,
        LossType.MAE: tf.losses.mean_absolute_error,
        LossType.HUBER: tf.losses.Huber,
        LossType.CATEGORICAL: tf.losses.CategoricalCrossentropy
    }

    loss_fn = losses.get(loss_name, None)

    if loss_fn is None:
        log.error(f'{loss_name} 손실 함수는 지원하지 않습니다.\n'
                  f'지원하는 손실 함수는 다음과 같습니다.\n'
                  f'mse, mae, huber')

    return loss_fn
