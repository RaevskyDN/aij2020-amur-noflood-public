# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import date
from typing import Union

from ..config_features import CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from ..config import DAYS_FORECAST, ALL_STATIONS
from ..utils.normalizer import get_normalizer_stats


def prepare_data(amur_df: pd.DataFrame,
                     start_date: Union[date, str],
                     end_date: Union[date, str]) -> np.array:
    '''
    Преоразование из датафрейма в 3d-array для формата модели

    Итоговый шейп [n,DAYS_FORECAST,n_features] - n - объем выборки (для инференса 1)
                                            DAYS_FORECAST - количество дней предсказания (10),
                                           n_features - количество признаков

    :param amur_df: pd.DataFrame
    :param start_date: date,str - начало по времени тестовой выборки
    :param end_date: date,str - конец по времени тестовой выборки
    :return: np.array, выборка по формату для модели
    '''

    x_df = amur_df[(amur_df['date'] >= start_date) &
                   (amur_df['date'] < end_date)].copy()

    stats = get_normalizer_stats()
    for col in NUMERICAL_FEATURES:
        _mean = stats[col]['mean']
        _std = stats[col]['std']
        x_df[col] = (x_df[col] - _mean) / _std

    x_df.sort_values('date', inplace=True)

    x_array = []
    step = 0
    while True:
        if step >= len(x_df):
            break
        x_array.append(x_df.iloc[step:step + DAYS_FORECAST][NUMERICAL_FEATURES + CATEGORICAL_FEATURES].values)
        if step + DAYS_FORECAST + 1 >= len(x_df):
            break
        step += DAYS_FORECAST

    X = np.transpose(np.dstack(x_array), (2, 0, 1))

    return X