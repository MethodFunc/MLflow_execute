from functools import partial

import neuralprophet
import numpy as np
from hyperopt import hp, STATUS_OK, tpe, fmin, Trials
from neuralprophet import NeuralProphet

neuralprophet.utils.set_random_seed(2022)


def hyperopt_fn(params, train_data, validation_data, col_list, scale_type, n_forecasts=24, n_lags=24):
    model = NeuralProphet(**params, n_forecasts=n_forecasts, n_lags=n_lags)
    model.add_lagged_regressor(names=col_list, normalize=scale_type)
    result = model.fit(train_data, freq='h', validation_df=validation_data)
    loss_col = [col for col in result.columns if 'Loss_val' in col]
    if len(loss_col) == 0:
        loss_col = [col for col in result.columns if 'Loss' in col]

    result.rename(columns={
        loss_col[0]: 'loss'
    }, inplace=True)

    best_loss = result['loss'].to_list()[-1]

    return {"loss": best_loss, "status": STATUS_OK}


def hyperopt_fit(train_data, validation_data, col_list, config):
    trials = Trials()
    random_state = np.random.default_rng(0)

    growth = ["off", "linear", "discontinuous"]
    d_hidden = [16, 32, 64, 128]
    epochs = [30, 60, 100, 150, 200, 300]
    batch_size = [16, 32, 64, 72, 128]
    weekly_seasonality = [0, 1]
    seasonality_mode = ["additive", "multiplicative"]
    normalize = ['minmax', 'standardize', 'soft', 'soft1']
    params = {
        "growth": hp.choice("growth", growth),
        "num_hidden_layers": hp.randint("num_hidden_layers", 0, 6),
        "d_hidden": hp.choice("d_hidden", d_hidden),
        "learning_rate": hp.loguniform("learning_rate", -9, -3),
        "epochs": hp.choice("epochs", epochs),
        "batch_size": hp.choice("batch_size", batch_size),
        "weekly_seasonality": hp.choice("weekly_seasonality", weekly_seasonality),
        "seasonality_mode": hp.choice("seasonality_mode", seasonality_mode),
        "normalize": hp.choice("normalize", normalize)
    }

    _eval = config['neural_setting']['eval']
    n_forecast = config['neural_setting']['n_forecast']
    n_lags = config['neural_setting']['n_lags']
    scale_type = config['scale_type']
    train_fn = partial(hyperopt_fn, train_data=train_data, validation_data=validation_data, col_list=col_list,
                       n_forecasts=n_forecast, n_lags=n_lags, scale_type=scale_type)
    best_result = fmin(train_fn, space=params, algo=tpe.suggest, max_evals=_eval, timeout=6369,
                       rstate=random_state, trials=trials)
    best_params = {
        'growth': growth[best_result['growth']],
        'd_hidden': d_hidden[best_result['d_hidden']],
        'epochs': epochs[best_result['epochs']],
        'learning_rate': best_result['learning_rate'],
        'num_hidden_layers': best_result['num_hidden_layers'],
        'weekly_seasonality': best_result['weekly_seasonality'],
        'seasonality_mode': best_result['seasonality_mode'],
        'normalize': normalize[best_result['normalize']]

    }

    return best_params
