from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

"""
    전처리 데이터 수정은 이 파일에서 해주시길 바랍니다.
"""


def preprocess_data(raw_data, output_type: str = 'original'):
    """
    :param raw_data:
    preprocessing custom your dataframe
     :param output_type: str [original, hour] default: original
    :return: Pandas DataFrame
    """
    dataframe = raw_data.copy()
    dataframe.rename(columns={
        '//Datetime': 'DATETIME',
        'datetime': 'DATETIME',
        'index': 'DATETIME'
    }, inplace=True)

    dataframe['DATETIME'] = pd.to_datetime(dataframe['DATETIME'])

    if output_type == 'hour':
        dataframe['min'] = dataframe['DATETIME'].dt.minute
        dataframe['sec'] = dataframe['DATETIME'].dt.second
        dataframe = dataframe.loc[(dataframe['min'] == 0) & (dataframe['sec'] == 0)]
        dataframe.drop(['min', 'sec'], axis=1, inplace=True)

    dataframe.dropna(inplace=True)
    dataframe.drop_duplicates(subset=['DATETIME'], keep='first', inplace=True, ignore_index=True
                              )
    dataframe['year'] = dataframe['DATETIME'].dt.year
    dataframe['month'] = dataframe['DATETIME'].dt.month
    dataframe['day'] = dataframe['DATETIME'].dt.day
    dataframe['hour'] = dataframe['DATETIME'].dt.hour
    dataframe['quarter'] = dataframe['DATETIME'].dt.quarter
    hour = 60 * 60
    day = 24 * hour
    year = 365.2425 * day

    timestamp_s = dataframe['DATETIME'].map(lambda y: datetime.timestamp(y))

    dataframe['year_sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    dataframe['year_cos'] = np.cos(timestamp_s * (2 * np.pi / year))

    dataframe['day_sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    dataframe['day_cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    dataframe['diff'] = dataframe['INFO_POWER_ACTIVE'].diff()
    dataframe['diff_2'] = dataframe['INFO_SPEED_WIND'].diff()

    dataframe['rolling'] = dataframe['INFO_POWER_ACTIVE'].rolling(6).mean()
    dataframe.fillna(method='ffill', inplace=True)
    dataframe.fillna(method='bfill', inplace=True)

    dataframe.drop(['DATETIME'], axis=1, inplace=True)

    return dataframe


def neural_process(dataframe, output_type='hour'):
    modify_data = dataframe.copy()
    modify_data.rename(columns={'DATETIME': 'ds', 'index': 'ds', '//Datetime': 'ds', 'datetime': 'ds',
                                'INFO_SPEED_WIND': 'WS', 'INFO.SPEED.WIND': 'WS',
                                'INFO.DIRECTION.WIND': 'WD', 'INFO_DIRECTION_WIND': 'WD', 'AVG.DIRECTION.EMS': 'WD',
                                'INFO.POWER.ACTIVE': 'y', 'INFO_POWER_ACTIVE': 'y'}, inplace=True)

    modify_data = modify_data.reindex(columns=['ds', 'WD', 'WS', 'y'])
    modify_data['ds'] = pd.to_datetime(modify_data['ds'])

    if output_type == 'hour':
        modify_data['min'] = modify_data['ds'].dt.minute
        modify_data['sec'] = modify_data['ds'].dt.second
        modify_data = modify_data.loc[(modify_data['min'] == 0) & (modify_data['sec'] == 0)]
        modify_data.drop(['min', 'sec'], axis=1, inplace=True)

    modify_data.set_index('ds', inplace=True)

    modify_data = modify_data.loc[~modify_data.index.duplicated(keep='first')]
    modify_data = modify_data.loc[:'2022-07-31']

    modify_data = modify_data.resample('1H').mean()
    modify_data.fillna(method='ffill', inplace=True)
    modify_data.fillna(method='bfill', inplace=True)

    modify_data.loc[(modify_data['y'] < 0, 'y')] = 0

    col_list = ['WD', 'WS']
    train_data = modify_data.loc[:'2022-05-31']
    test_data = modify_data.loc['2022-05-31':'2022-07-28']

    # 아래는 수정 x 다른 함수로 빠지면 signature가 생성이 안됨
    train_data.reset_index(inplace=True)
    test_data.reset_index(inplace=True)

    input_schema = Schema([ColSpec(set_signature(train_data[col]), col) for col in train_data.columns])
    output_schema = Schema([ColSpec(set_signature(train_data['y']), 'y')])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)

    return (train_data, test_data), col_list, signature


def create_model(x_train, config):
    inps = tf.keras.layers.Input(shape=x_train.shape[1:])
    x = tf.keras.layers.GRU(64, return_sequences=True)(inps)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.GRU(32, return_sequences=False)(x)
    outs = tf.keras.layers.Dense(config['window_setting']['target_len'])(x)

    model = tf.keras.models.Model(inps, outs, name=config['model']['name'])

    return model


def set_signature(data):
    if data.dtype == 'datetime64[ns]':
        schema_type = 'datetime'
    if (data.dtype == 'float64') or (data.dtype == 'float32'):
        schema_type = 'float'

    return schema_type
