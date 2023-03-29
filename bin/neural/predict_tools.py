from datetime import timedelta

import numpy as np
import pandas as pd


def eval_predict(model, ori_data):
    dataframe = ori_data.copy()
    start_date = dataframe.loc[0]['ds']
    end_date = dataframe.loc[len(dataframe) - 1]['ds']
    if type(start_date) != 'Timestamp':
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

    date_range = (end_date - start_date).days
    dataframe.set_index('ds', inplace=True)
    y_pred = []

    for i in range(date_range):
        tmp_date = start_date + timedelta(days=i)
        date_str = tmp_date.strftime('%Y-%m-%d')
        daterange = pd.date_range(date_str, periods=48, freq='1H')
        range_df = pd.DataFrame(daterange, columns=['ds'])
        tmp_df = pd.merge(range_df, dataframe.loc[date_str], on='ds', how='outer')
        # decompose False안하면 Pandas Warning 뜸
        prediction = model.predict(tmp_df, decompose=False)
        cols = [col for col in prediction.columns if 'yhat' in col]
        predict = prediction[cols].sum()
        y_pred.append(predict)

    y_pred = np.array(y_pred)

    tmp_date = start_date + timedelta(days=1)
    date_str = tmp_date.strftime('%Y-%m-%d')

    y_true = dataframe.loc[date_str:]['y'].values
    y_pred = y_pred.ravel()

    return y_true, y_pred
