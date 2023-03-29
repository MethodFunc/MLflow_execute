from enum import Enum

import tensorflow as tf


class OptimzerType(str, Enum):
    ADAM = 'adam'
    SGD = 'sgd'
    RMSPROP = 'rmsprop'
    ADAMAX = 'adamax'


def get_optimizer_fn(config, log):
    optim_name = config['train_setting']['optimizer']
    learning_rate = float(config['train_setting']['learning_rate'])
    optimizers = {
        OptimzerType.ADAM: tf.optimizers.Adam(learning_rate=learning_rate),
        OptimzerType.SGD: tf.optimizers.SGD(learning_rate=learning_rate),
        OptimzerType.RMSPROP: tf.optimizers.RMSprop(learning_rate=learning_rate),
        OptimzerType.ADAMAX: tf.optimizers.Adamax(learning_rate=learning_rate),
    }

    optim_fn = optimizers.get(optim_name, None)

    if optim_fn is None:
        log.error(f'{optim_name} 옵티마이저는 지원하지 않습니다.\n'
                  f'지원하는 옵티마이저는 다음과 같습니다.\n'
                  f'adam, sgd, rmsprop, adamax')

    return optim_fn
