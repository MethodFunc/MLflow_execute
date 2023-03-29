import tensorflow as tf
from mlflow import log_metrics


class MLflowCallbacks(tf.keras.callbacks.Callback):
    """
        케라스 mlflow용  에포치 당 손실함수 값 및 평가 값 적재 콜백 함수
        autolog 미 사용 시 사용하면 됨
    """

    def __init__(self):
        super(MLflowCallbacks, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        log_metrics(logs, epoch)
