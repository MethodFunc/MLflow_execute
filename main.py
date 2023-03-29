import os
from datetime import datetime
from functools import partial
from pathlib import Path

import mlflow

from bin.datatools import get_file_list
from bin.metrics import GenPower
from bin.run_process import split_process, mlflow_run
from setting import load_setting, load_logger


def rename_old(name):
    for value in GenPower:
        if value.rename() == name:
            return value.name

    return name


def rename_java_team(name):
    return GenPower[name].rename()


# 설정 파일 불러오기 (전역)

config = load_setting()

mlflow.set_tracking_uri(config['MLFLOW_TRACKING_URI'])
os.environ["CUDA_VISIBLE_DEVICES"] = str(config['gpu_num'])

if __name__ == '__main__':
    logger = load_logger()
    logger.info('Start MLflow Training')
    logger.info(f'Training {config["mlflow_type"]} start')

    st = datetime.now()

    files = get_file_list(config['data_path'], config['generator_plant'])
    process_partial = partial(split_process, config=config)
    if isinstance(files, list):
        for file in files:
            sub_st = datetime.now()

            file_name = rename_old(Path(file).stem)
            dataloader = process_partial(file)
            mlflow_run(file_name, dataloader, logger, config)

            sub_ed = datetime.now()
            logger = load_logger()
            logger.info(f'{file_name} elapsed time: {sub_ed - sub_st}')

    if isinstance(files, str):
        file_name = rename_old(Path(config['data_path']).stem)
        dataloader = process_partial(files)
        mlflow_run(file_name, dataloader, logger, config)

    ed = datetime.now()
    logger = load_logger()
    logger.info(f'Total elapsed Time: {ed - st}')
    logger.info('Training End')
