import numpy as np
import pandas as pd


# todo : 다른 스케일 함수 넣기 ( Standard, Minmax 추가완료)

class MinMaxScale:
    """
        사이킷런의 scale 함수를 기반하여 작성되었습니다.
        사이킷런에서 불편한 inverse부분을 고쳐서 카테고리를 번호(넘파이 변환시) 혹은 데이터프레임 행이름(판다스 변환시)적으면 해당 카테고리만 되돌려집니다.
        (사이킷런은 shape가 모두 같아야 가능함)
        카테고리를 아무 것도 적지 않을 시 사이킷런처럼 모두 역변환됩니다.
        사용법:
            scale = MinMaxScale()
            # 스케일 훈련 변환(필요한 변수가 저장됨)
            data = scale.fit(data)
            # 스케일 변환
            data2 = scale.transform(data2)
            # 역변환
            data = scale.inverse(data)
            # numpy 역변환 (shape = (3, 2))
            data = scale.inverse(data, categories=2)
            # 판다스 역변환
            data = scale.inverse(data, categories='width')
    """

    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.range_scale = None
        self._max = None
        self._min = None
        self._columns = None

    def get_columns(self):
        return self._columns

    def fit(self, data):
        feature_range = self.feature_range
        if feature_range[0] >= feature_range[1]:
            raise ValueError(
                "Minimum of desired feature range must be smaller than maximum. Got %s."
                % str(feature_range)
            )

        self._max = np.max(data, axis=0)
        self._min = np.min(data, axis=0)

        data_range = self._max - self._min

        self.range_scale = (feature_range[1] - feature_range[0]) / data_range

        self._min = feature_range[0] - self._min * self.range_scale

        if isinstance(data, pd.core.frame.DataFrame):
            self._columns = {name: i for i, name in enumerate(data.columns)}
        else:
            self._columns = f'This data is not dataframe. {data.shape}'

        scale = data * self.range_scale
        scale += self._min

        return scale

    def transform(self, data):
        if self._max is None:
            raise ValueError("Must should be fit data")
        scale = data * self.range_scale
        scale += self._min

        return scale

    def inverse(self, data, categories=None):
        if categories is not None:
            inv = data - self._min[categories]
            inv /= self.range_scale[categories]
        else:
            if not isinstance(data, pd.core.frame.DataFrame):
                inv = data - self._min
                inv /= self.range_scale
            else:
                inv = data - self._min.values
                inv /= self.range_scale

        return inv


class StandardScale:
    """
        사이킷런의 scale 함수를 기반하여 작성되었습니다.
        사이킷런에서 불편한 inverse부분을 고쳐서 카테고리를 번호(넘파이 변환시) 혹은 데이터프레임 행이름(판다스 변환시)적으면 해당 카테고리만 되돌려집니다.
        (사이킷런은 shape가 모두 같아야 가능함)
        카테고리를 아무 것도 적지 않을 시 사이킷런처럼 모두 역변환됩니다.
        사용법:
            scale = StandardScale()
            # 스케일 훈련 변환(필요한 변수가 저장됨)
            data = scale.fit(data)
            # 스케일 변환
            data2 = scale.transform(data2)
            # 역변환
            data = scale.inverse(data)
            # numpy 역변환 (shape = (3, 2))
            data = scale.inverse(data, categories=2)
            # 판다스 역변환
            data = scale.inverse(data, categories='width')
    """

    def __init__(self):
        self._std = None
        self._mean = None
        self._columns = None

    def fit(self, data):
        self._mean = np.mean(data, axis=0)
        self._std = np.std(data, axis=0)

        if isinstance(data, pd.core.frame.DataFrame):
            self._columns = {name: i for i, name in enumerate(data.columns)}
        else:
            self._columns = f'This data is not a dataframe. {data.shape}'

        scale = (data - self._mean) / self._std

        return scale

    def transform(self, data):
        scale = (data - self._mean) / self._std

        return scale

    def inverse(self, data, categories=None):
        if categories is not None:
            inv = data * self._std[categories] + self._mean[categories]
        else:
            inv = data * self._std + self._mean

        return inv

    def get_columns(self):
        return self._columns
