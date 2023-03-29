from enum import Enum

from bin.datatools import TimeWindowFunc
from bin.preprocessing import MinMaxScale, StandardScale


class ScaleType(str, Enum):
    MINMAX = 'minmax'
    STANDARD = 'standardize'


class InsType(str, Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


def scale_func(data, estimator, ins_type='train'):
    if ins_type == InsType.TRAIN:
        scale_data = estimator.fit(data)

    elif ins_type == InsType.VAL or ins_type == InsType.TEST:
        scale_data = estimator.transform(data)
    else:
        raise TypeError(f'{ins_type} is not support')

    return scale_data


def keras_processing(loader, config):
    dataloader, scale_dict = {}, {}
    if ScaleType.MINMAX == config['scale_type']:
        fscale = MinMaxScale(feature_range=(config['minmax_range']['min'], config['minmax_range']['max']))
        tscale = MinMaxScale(feature_range=(config['minmax_range']['min'], config['minmax_range']['max']))

    if ScaleType.STANDARD == config['scale_type']:
        fscale = StandardScale()
        tscale = StandardScale()

    x_train, y_train = loader['train'][0], loader['train'][1]
    x_test, y_test = loader['test'][0], loader['test'][1]

    x_train, y_train = scale_func(x_train, fscale), scale_func(y_train, tscale)
    x_test, y_test = scale_func(x_test, fscale, 'test'), scale_func(y_test, tscale, 'test')

    x_train, y_train = TimeWindowFunc(x_train, y_train, **config['window_setting']).get_data()
    x_test, y_test = TimeWindowFunc(x_test, y_test, **config['window_setting'],
                                    step=config['window_setting']['target_len']).get_data()

    if 'val' in loader:
        x_val, y_val = loader['val'][0], loader['val'][1]
        x_val, y_val = scale_func(x_val, fscale, 'val'), scale_func(y_val, tscale, 'val')
        x_val, y_val = TimeWindowFunc(x_val, y_val, **config['window_setting']).get_data()

        dataloader['val'] = x_val, y_val

    dataloader['train'] = x_train, y_train
    dataloader['test'] = x_test, y_test
    scale_dict['feature'], scale_dict['target'] = fscale, tscale

    return dataloader, scale_dict
