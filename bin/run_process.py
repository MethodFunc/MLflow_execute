from enum import Enum

import pandas as pd

from bin.database_tools import database_adapt
from bin.datatools import split_dataframe
from bin.mlflows.mlflow_train import keras_run, neural_run
from user_modify import preprocess_data, neural_process

ARTIFACT_URI = "sftp://a2malgo:a2minfo123@211.240.98.153:14022/mlflow_greenmark"

# postgresql numpy 문제 해결
database_adapt()


# todo : Pytorch도 가능한가

class MlfowType(str, Enum):
    KERAS = 'keras'
    TENSORFLOW = 'tensorflow'
    NEURALPROPHET = 'neural'


def split_process(file_path, config):
    dataloader = {}
    raw_data = pd.read_csv(file_path)

    # 뉴럴 프로펫 전용
    if MlfowType.NEURALPROPHET == config['mlflow_type']:
        pre_df, col_list, signature = neural_process(raw_data)
        train_data, validation_data = pre_df
        dataloader['train'] = train_data
        dataloader['val'] = validation_data
        dataloader['col_list'] = col_list
        dataloader['signature'] = signature

        return dataloader

    pre_df = preprocess_data(raw_data, output_type=config['output_type'])

    if config['split_setting']['val_ratio']:
        train_data, val_data, test_data = split_dataframe(pre_df, **config['split_setting'])
    else:
        train_data, test_data = split_dataframe(pre_df, **config['split_setting'])

    dataloader['train'] = train_data
    dataloader['test'] = test_data

    if config['split_setting']['val_ratio']:
        dataloader['val'] = val_data

        return dataloader

    return dataloader


def mlflow_run(name, loader, logger, config):
    active_func = {
        MlfowType.KERAS: keras_run,
        MlfowType.TENSORFLOW: keras_run,
        MlfowType.NEURALPROPHET: neural_run
    }

    active_func.get(config['mlflow_type'])(name, loader, logger, config)
