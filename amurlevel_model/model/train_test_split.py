# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from datetime import date
from typing import Union,Tuple,Optional,List

from ..config_features import CATEGORICAL_FEATURES,NUMERICAL_FEATURES
from ..config import DAYS_FORECAST,ALL_STATIONS
from ..utils.normalizer import get_normalizer_stats

def train_test_split(amur_df: pd.DataFrame,
                     start_test_date: Union[date,str],
                     end_test_date: Union[date,str],
                     fname: Optional[str]=None,
                     numerical_features: Optional[List[str]]=None,
                     categorical_features: Optional[List[str]]=None) -> Tuple[np.array,np.array,np.array,np.array]:
    '''
    Деление на трейн, тест для обучения.
    Шаг с которым идем по трейну - 1 день, шак с которым идем по тесту - 10 дней

    Итоговый шейп [n,DAYS_FORECAST,n_features] - n - объем выборки,
                                           DAYS_FORECAST - количество дней предсказания (10),
                                           n_features - количество признаков

    :param amur_df: pd.DataFrame
    :param start_test_date: date,str - начало по времени тестовой выборки
    :param end_test_date: date,str - конец по времени тестовой выборки
    :param fname: str, путь до файла json cо статистикой mean,std для каждого поля
    :param numerical_features: List[str] - список численных признаков
    :param categorical_features: List[str] - список категориальных признаков
    :return: tuple:
                    X_train - обучающая выборка
                    y_train - метки для обучающей выборки
                    X_test - тестовая выборка
                    y_test - метки для обучающей выборки
    '''
    if numerical_features is None:
        numerical_features = NUMERICAL_FEATURES

    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES

    targets = ['sealevel_max_' + identifier for identifier in ALL_STATIONS]

    train = amur_df[amur_df['date'] < start_test_date].copy()
    test = amur_df[(amur_df['date'] >= start_test_date) &
                       (amur_df['date'] < end_test_date)].copy()

    stats = get_normalizer_stats(fname)
    for col in numerical_features:
        _mean = stats[col]['mean']
        _std = stats[col]['std']
        train[col] = (train[col] - _mean) / _std
        test[col] = (test[col] - _mean) / _std

    train.sort_values('date', inplace=True)

    train_x_array = []
    train_y_array = []
    step = 0
    while True:
        if step + DAYS_FORECAST + 1 >= len(train):
            break
        if train.iloc[step:step + DAYS_FORECAST][targets].count().min() < DAYS_FORECAST:
            step += 1
            continue
        train_x_array.append(train.iloc[step:step + DAYS_FORECAST][numerical_features + categorical_features].values)
        train_y_array.append(train.iloc[step:step + DAYS_FORECAST][targets].values)
        step += 1
    X_train = np.transpose(np.dstack(train_x_array), (2, 0, 1))
    y_train = np.transpose(np.dstack(train_y_array), (2, 0, 1))

    step = 0
    test.sort_values('date', inplace=True)
    test_x_array = []
    test_y_array = []
    while True:
        if step >= len(test):
            break
        if test.iloc[step:step + DAYS_FORECAST][targets].count().min() < DAYS_FORECAST:
            step += DAYS_FORECAST
            continue
        test_x_array.append(test.iloc[step:step + DAYS_FORECAST][numerical_features + categorical_features].values)
        test_y_array.append(test.iloc[step:step + DAYS_FORECAST][targets].values)
        if step + DAYS_FORECAST*2+1 >= len(test):
            break
        step += DAYS_FORECAST
    X_test = np.transpose(np.dstack(test_x_array), (2, 0, 1))
    y_test = np.transpose(np.dstack(test_y_array), (2, 0, 1))

    return X_train, y_train, X_test, y_test