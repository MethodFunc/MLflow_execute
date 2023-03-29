import numpy as np
from psycopg2.extensions import AsIs, register_adapter

"""
    postgresql numpy 오류 해결
"""


def adapt_numpy_float64(numpy_float64):
    return AsIs(numpy_float64)


def adapt_numpy_int64(numpy_int64):
    return AsIs(numpy_int64)


def adapt_numpy_float32(numpy_float32):
    return AsIs(numpy_float32)


def adapt_numpy_int32(numpy_int32):
    return AsIs(numpy_int32)


def adapt_numpy_array(numpy_array):
    return AsIs(tuple(numpy_array))


def database_adapt():
    register_adapter(np.float64, adapt_numpy_float64)
    register_adapter(np.int64, adapt_numpy_int64)
    register_adapter(np.float32, adapt_numpy_float32)
    register_adapter(np.int32, adapt_numpy_int32)
    register_adapter(np.ndarray, adapt_numpy_array)
