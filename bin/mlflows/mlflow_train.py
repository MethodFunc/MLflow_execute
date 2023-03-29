import shutil
import pickle
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.keras

from bin.keras_tools.losses import get_loss_fn
from bin.keras_tools.optimizers import get_optimizer_fn
from bin.metrics import evaluation_metric
from bin.mlflows.experiments import set_experiments
from bin.mlflows.preprocessing import keras_processing
from bin.mlflows.registers import model_register
from bin.neural.automl import hyperopt_fit
from bin.neural.models import NeuralModel
from bin.neural.predict_tools import eval_predict
from bin.neural.wrapper import NeuralProphetWrapper
from user_modify import create_model
from setting import load_logger


def insert_logs(name, y_true, y_pred):
    for idx in range(len(y_pred)):
        mlflow.log_metrics({
            'actual': y_true[idx],
            'prediction': y_pred[idx],
            'nmae': evaluation_metric(name, y_true[idx], y_pred[idx])
        }, idx)


def set_run_name(name, config, logger):
    run_name = datetime.now().strftime('%Y%m%d%H%M%S')

    return run_name


def save_scale(run_id, scale_dict):
    artifact_scale_logs = f'./mlruns/{run_id}/scale'
    Path(artifact_scale_logs).mkdir(parents=True, exist_ok=True)

    with open(f'{artifact_scale_logs}/feature_scale.pkl', 'wb') as f:
        pickle.dump(scale_dict['feature'], f)
    with open(f'{artifact_scale_logs}/target_scale.pkl', 'wb') as f:
        pickle.dump(scale_dict['target'], f)

    mlflow.log_artifacts(artifact_scale_logs, artifact_path='model/scale')

    shutil.rmtree('./mlruns')


def keras_run(name, loader, logger, config):
    loader, scale_dict = keras_processing(loader, config)
    # 오토로그 (파라미터 및 로스 값 로깅)s
    mlflow.keras.autolog()
    model = create_model(loader['train'][0], config)
    model.compile(loss=get_loss_fn(config, logger), optimizer=get_optimizer_fn(config, logger))

    # 런 이름 (변경이 가능하지만 시간으로 지정)
    run_name = set_run_name(name, config, logger)
    set_experiments(name, config['ARTIFACT_URI'], logger)
    logger = load_logger()
    with mlflow.start_run(run_name=run_name) as run:
        # 모델 인증에 필요함
        run_id = run.info.run_id
        # 스케일 저장
        logger.info('Scale Saved')
        save_scale(run_id, scale_dict)

        model.fit(loader['train'][0], loader['train'][1],
                  epochs=config['train_setting']['epochs'],
                  batch_size=config['train_setting']['batch_size'],
                  validation_data=(loader['val'][0], loader['val'][1]),
                  validation_batch_size=config['train_setting']['batch_size'],
                  use_multiprocessing=True)

        prediction = model.predict(loader['test'][0], batch_size=64)

        logger.info('mlflow log prediction')
        # todo : 더 빠른 방법 찾아보기 (데이터 셋이 많아 질 수록 느림)
        y_true = scale_dict['target'].inverse(loader['test'][1].ravel())
        y_pred = scale_dict['target'].inverse(prediction.ravel())

        insert_logs(name, y_true, y_pred)

    logger.info('register models')
    # 모델 인증 저장
    model_register(model_name=name, artifact_uri=config['ARTIFACT_URI'], run_id=run_id)


def neural_run(name, loader, logger, config):
    train_data = loader['train']
    val_data = loader['val']
    col_list = loader['col_list']
    signature = loader['signature']

    best_params = hyperopt_fit(train_data, val_data, col_list, config)

    run_name = set_run_name(name, config, logger)
    set_experiments(name, config['ARTIFACT_URI'], logger)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            'n_forecasts': config['neural_setting']['n_forecast'],
            'n_lags': config['neural_setting']['n_lags']})
        mlflow.log_params(best_params)

        run_id = run.info.run_id

        model = NeuralModel(**best_params, n_forecasts=config['neural_setting']['n_forecast'],
                            n_lags=config['neural_setting']['n_lags'])

        model = model.add_lagged_regressor(names=col_list, normalize=True)

        result = model.fit(train_data, freq='h', validation_df=val_data)

        input_example = train_data.sample(n=1)

        # 실시간으로 손실함수 저장 안됨
        logger.info('loss insert')
        for idx, dict_value in enumerate(result.to_dict('records')):
            for key, value in dict_value.items():
                mlflow.log_metric(key.lower(), value, idx)

        y_true, y_pred = eval_predict(model, val_data)

        logger.info('mlflow log prediction')
        insert_logs(name, y_true, y_pred)

        mlflow.pyfunc.log_model("model",
                                python_model=NeuralProphetWrapper(model),
                                signature=signature)

    logger.info('register models')
    model_register(model_name=name, artifact_uri=config['ARTIFACT_URI'], run_id=run_id)
