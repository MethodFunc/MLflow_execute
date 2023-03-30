from datetime import timedelta

import numpy as np
import pandas as pd


def eval_predict(model, ori_data, config):
    dataframe = ori_data.copy()
    y_pred = []

    for i in range(0, len(ori_data)-config['neural_setting']['n_lags'], config['neural_setting']['n_forecast']):
        tmp_date = dataframe.loc[i]['ds']
        date_str = tmp_date.strftime('%Y-%m-%d %H:%M:%S')
        daterange = pd.date_range(date_str, periods=config['neural_setting']['n_forecast'] +
                                                    config['neural_setting']['n_lags'],
                                  freq=config['neural_setting']['freq'])

        range_df = pd.DataFrame(daterange, columns=['ds'])
        tmp_df = pd.merge(range_df, dataframe.iloc[i:i+config['neural_setting']['n_lags']], on='ds', how='outer')
        # decompose False안하면 Pandas Warning 뜸
        prediction = model.predict(tmp_df, decompose=False)
        cols = [col for col in prediction.columns if 'yhat' in col]
        predict = prediction[cols].sum()
        y_pred.append(predict)

    y_pred = np.array(y_pred)

    y_true = dataframe.iloc[config['neural_setting']['n_lags']:]['y'].values
    y_pred = y_pred.ravel()

    return y_true, y_pred
