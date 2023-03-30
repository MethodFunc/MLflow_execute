import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def get_file_list(path, generator_plant: Optional[str] = None):
    if Path(path).is_dir():
        file_ = [os.path.join(path, f) for f in os.listdir(path)
                 if Path(os.path.join(path, f)).is_file()]
        file_.sort()

        if generator_plant:
            file_ = [f for f in file_ if generator_plant in f]

    elif Path(path).is_file():
        file_ = path

    return file_


def split_dataframe(data, target_col: Optional[str] = None, train_ratio: float = 0.8,
                    val_ratio: Optional[float] = None) -> Tuple[np.ndarray]:
    """
    :param data: input train data
    :param target_col: Target columns
    :param train_ratio: float, default 0.8
    :param val_ratio:  float or None default None
    :return: val_ratio is None -> train_data, test_data, in numpy array
            val_ratio is noe None -> train_data, val_data, test_data, in numpy array
    """
    if isinstance(data, pd.core.frame.DataFrame) or isinstance(data, pd.core.series.Series):
        if target_col and target_col not in data.columns:
            raise KeyError('Target col is not exists. check your target name')

        if target_col:
            feature_data = data.drop([target_col], axis=1).values
            target_data = data.pop(target_col).values

        else:
            feature_data = data.values
    else:
        raise ValueError('Data is not Dataframe')

    data_size = len(data)
    train_size = int(data_size * train_ratio)

    if val_ratio:
        val_size = int(data_size * (train_ratio + val_ratio))

    if target_col:
        train_data = feature_data[:train_size], target_data[:train_size]

        if val_ratio:
            val_data = feature_data[train_size:val_size], target_data[train_size:val_size]
            test_data = feature_data[val_size:], target_data[val_size:]

            return train_data, val_data, test_data

        test_data = feature_data[train_size:], target_data[train_size:]

        return train_data, test_data

    train_data = feature_data[:train_size]

    if val_ratio:
        val_data = feature_data[train_size:val_size]
        test_data = feature_data[val_size:]

        return train_data, val_data, test_data

    test_data = feature_data[train_size:]

    return train_data, test_data


class TimeWindowFunc:
    def __init__(self, features, target, seq_len: int, target_len: int = 1, step: int = 1):
        """
        :param features: 데이터 변수가 존재하는 데이터프레임 혹은 넘파이 배열을 입력 (2차원)
        :param target: 목표 변수가 존재한는 데이터프레임 혹은 넘파이 배열을 입력 (2차원) None 지원
        :param seq_len: 윈도우 크기를 지정합니다.
        :param target_len: 단일이면 1, 여러개면 1이상을 입력해주세요 (기본 1)
        :param step: 데이터를 몇개를 건너뛰며 생성할지 정합니다 (기본 1)
        :return: target is None -> data
                target is not None -> data, label
        """
        self.features = features
        self.target = target
        self.start_index = seq_len
        self.target_len = target_len
        self.end_index = len(features) - target_len + 1
        self.step = step
        self.check = True if isinstance(features, pd.core.frame.DataFrame) or isinstance(features,
                                                                                         pd.core.series.Series) else False

    def get_data(self):
        return self.create_window_function()

    def create_window_function(self):
        data, label = list(), list()
        for i in range(self.start_index, self.end_index, self.step):
            indices = range(i - self.start_index, i)
            if self.check:
                data_ = self.features.iloc[indices]
            else:
                data_ = self.features[indices]

            if self.target is not None:
                if self.check:
                    label_ = self.target.iloc[i:i + self.target_len]
                else:
                    label_ = self.target[i:i + self.target_len]

                label.append(label_)

            data.append(data_)

        data = np.array(data)
        if self.target is not None:
            label = np.array(label)
            return data, label
        else:
            return data
