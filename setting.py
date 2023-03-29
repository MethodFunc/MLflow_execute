import logging.config
import yaml
import os


def load_logger():
    logging_config_path = 'logging.yaml'
    if os.path.exists(logging_config_path):
        with open(logging_config_path, 'rt') as f:
            logging_config = yaml.load(f, Loader=yaml.FullLoader)
            logging.config.dictConfig(logging_config)
    else:
        logging.basicConfig(level=logging.INFO)

    return logging.getLogger()


def load_setting():
    with open('./setting.yaml', encoding='utf8') as config_file:
        config = yaml.load(config_file, yaml.FullLoader)

        return config
