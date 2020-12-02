# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
from dateutil import relativedelta
import gc
import logging

from .demodulation import level_demodul_121,level_demodul_365
from ..config_features import NUMERICAL_FEATURES,CATEGORICAL_FEATURES,CAT_MAP
from ..config import ALL_STATIONS

def make_dataset(raw_hydro_df: pd.DataFrame, train=False) -> pd.DataFrame:
    '''
    Метод для создания датасета со всеми таргетами и фичами
    Важно! Все гидропосты не из ALL_STATIONS будут проигнорированы
    Используются следующие фичи:
          1) cos,sin преобразование от текущего дня (для учета цикличности)
          2) cos,sin преобразование 10 дней назад (для учета цикличности)
          3) воссталовленные периодические составляющие для уровня 10 дней назад с периодом 365 и 121 дней
          4) уровень воды 10,20,30 дней назад
          5) уровень воды год назад (ровно в тот жен день и усредненный за 3 дня)
          6) уровень воды, код состояния водного объекта, расход воды, толщина льда,
                 снежный покров 10 дней назад
          7) количество дней без осадков
          8) направление ветра, скорость ветра, количество осадков, мин. температура воздуха,
                   макс. температура воздуха, относительная влажность, давление за текущий день
          9) направление ветра, скорость ветра, количество осадков, мин. температура воздуха,
                   макс. температура воздуха, относительная влажность, давление за вчерашний день
         10) направление ветра, скорость ветра, максимальная скорость ветра,
                   количество осадков, мин. температура воздуха, макс. температура воздуха,
                   температура почвы, относительная влажность, давление,погода между сроками,
                   погода в в срок наблюдения, количество облачности 10 дней назад
        11) среднее количество осадков и относительной влажности за 3,7 и 45 дней
        12) среднее количество осадков и относительной влажности за 3,7 дней 10 дней назад

    Каждые признаки вычисляются для каждого гидропоста из ALL_STATIONS и разворачиваются по дате
                 в единый датафрейм

    :param raw_hydro_df: pd.DataFrame, датафрейм со гидро-метео признаками для всех станций
    :param train: bool, если True то считаем что это датасет для обучения модели
    :return: amur_df - pd.DataFrame - получившийся датасет
    '''
    hydro_df = raw_hydro_df.copy()
    features = []
    logger = logging.getLogger()
    logger.info('~~~START MAKING FEATURES~~~')

    # категориальные фичи
    if len(CATEGORICAL_FEATURES) > 0:
        logger.info('\tcategorical features')
        for col in ['water_code', 'presentWeather', 'pastWeather', 'cloudCoverTotal']:
            hydro_df[col] = col + hydro_df[col].astype(np.int).astype(str)
            hydro_df[col] = hydro_df[col].map(CAT_MAP)

    # sin,cos преобразования от дат
    logger.info('\tsin-cos dates')
    hydro_df['sin_day_of_year'] = hydro_df['date'].apply(lambda x: np.sin(2 * math.pi * x.timetuple().tm_yday / 365))
    hydro_df['cos_day_of_year'] = hydro_df['date'].apply(lambda x: np.cos(2 * math.pi * x.timetuple().tm_yday / 365))
    features.extend(['sin_day_of_year', 'cos_day_of_year'])
    
    # количество дней без осадков
    logger.info('\tdays without precipitation')
    hydro_df['perc_days'] = hydro_df.groupby('identifier')['totalAccumulatedPrecipitation'].apply(
        lambda x: (x > 0.01).astype(np.int).cumsum())
    hydro_df['perc_days'] = (hydro_df.groupby(['identifier', 'perc_days']).cumcount() - 1).clip(lower=0)
    features.append('perc_days')
    
    # --- 10 дней назад---
    logger.info('\t10 days before')
    hydro_df.sort_values('date', inplace=True)
    for col in ['sealevel_max', 'ice_thickness', 'snow_height', 'water_flow', 'water_code',
                'water_temp', 'pastWeather', 'presentWeather', 'cloudCoverTotal',
                'windDirection', 'windSpeed', 'maximumWindGustSpeed',
                'totalAccumulatedPrecipitation', 'soilTemperature',
                'airTemperature_min', 'airTemperature_max',
                'relativeHumidity', 'pressure',
                'sin_day_of_year', 'cos_day_of_year']:
        features.append('shift_10_days_' + col)
        hydro_df['shift_10_days_' + col] = hydro_df.groupby('identifier')[col].shift(10)

    # -----20-30 дней назад
    logger.info('\t20,30 days before')
    hydro_df.sort_values('date', inplace=True)
    for shift in [20, 30]:
        for col in ['sealevel_max', 'ice_thickness', 'snow_height', 'water_flow',
                    'water_temp']:
            features.append(f'shift_{shift}_days_' + col)
            hydro_df[f'shift_{shift}_days_' + col] = hydro_df.groupby('identifier')[col].shift(shift)
    
    # ----метод демодуляции

    logger.info('\tdemodulation')
    hydro_df['demodule_365_sealevel_max'] = hydro_df.groupby('identifier')['shift_10_days_sealevel_max'].apply(
        lambda x: level_demodul_365(x))
    hydro_df['x'] = hydro_df['shift_10_days_sealevel_max'] - hydro_df['demodule_365_sealevel_max']
    hydro_df['demodule_121_sealevel_max'] = hydro_df.groupby('identifier')['x'].apply(lambda x: level_demodul_121(x))
    hydro_df.drop('x', axis=1, inplace=True)
    hydro_df['demodule_diff'] = (hydro_df['shift_10_days_sealevel_max'] - hydro_df['demodule_365_sealevel_max'] -
                                hydro_df['demodule_121_sealevel_max'] + hydro_df['shift_10_days_sealevel_max'].mean())
    features.extend(['demodule_365_sealevel_max', 'demodule_121_sealevel_max', 'demodule_diff'])

    # --- для метео - день назад
    logger.info('\t1 day before')
    for col in ['windDirection', 'windSpeed',
                'totalAccumulatedPrecipitation', 'airTemperature_min', 'airTemperature_max',
                'relativeHumidity', 'pressure']:
        hydro_df['shift_1_days_' + col] = hydro_df.groupby('identifier')[col].shift(1)
        features.append('shift_1_days_' + col)
    
    # --- для метео - текущий день
    logger.info('\tmeteo')
    for col in ['windDirection', 'windSpeed',
                'totalAccumulatedPrecipitation', 'airTemperature_min', 'airTemperature_max',
                'relativeHumidity', 'pressure']:
        features.append(col)
    
    # --- накопленная влажность, осадки за 7,3,45 дней:
    logger.info('\t7,3,45 days mean')
    hydro_df.sort_values('date', inplace=True)
    for col in ['relativeHumidity', 'totalAccumulatedPrecipitation']:
        for acc in [7, 3, 45]:
            new_col = f'accumulate_{acc}_days_{col}_sum'
            test_df = hydro_df.groupby('identifier').rolling(f'{acc}D', on='date', center=False,
                                                            min_periods=1)[col].mean().reset_index().rename(
                columns={col: new_col})
            hydro_df = hydro_df.merge(test_df, on=['identifier', 'date'], how='left')
            features.append(new_col)
            if acc != 45:
                new_col = f'accumulate_{acc}_days_shift_10_days_{col}_sum'
                test_df = hydro_df.groupby('identifier').rolling(f'{acc}D', on='date', center=False,
                                                                min_periods=1)[
                    'shift_10_days_' + col].mean().reset_index().rename(columns={'shift_10_days_' + col: new_col})
                hydro_df = hydro_df.merge(test_df, on=['identifier', 'date'], how='left')
                features.append(new_col)
    
    # --- прошлогодние усредненные за три дня
    logger.info('\tlast year')
    test_df = hydro_df[['identifier', 'date', 'sealevel_max']].copy()
    test_df['date'] = test_df['date'].apply(lambda x: x + relativedelta.relativedelta(years=1))
    test_df.sort_values(['identifier', 'date'], inplace=True)
    test_df = test_df.groupby('identifier').rolling(3, on='date', center=True,
                                                    min_periods=1)['sealevel_max'].mean().reset_index() \
        .rename(columns={'sealevel_max': 'past_year_sealevel_max_3D'})
    hydro_df = hydro_df.merge(test_df[['identifier', 'date', 'past_year_sealevel_max_3D']], on=['identifier', 'date'],
                            how='left')
    features.append('past_year_sealevel_max_3D')
    
    test_df = hydro_df[['identifier', 'date', 'sealevel_max', 'water_code']].copy()
    test_df['date'] = test_df['date'].apply(lambda x: x + relativedelta.relativedelta(years=1))
    test_df.rename(columns={'sealevel_max': 'past_year_sealevel_max',
                            'water_code': 'past_year_water_code'}, inplace=True)
    hydro_df = hydro_df.merge(test_df[['identifier', 'date', 'past_year_sealevel_max']], on=['identifier', 'date'],
                            how='left')
    features.append('past_year_sealevel_max')
    #------------------------
    
    hydro_df.sort_values(['identifier', 'date'], inplace=True)
    hydro_df.drop_duplicates(['identifier', 'date'], keep='last', inplace=True)
    features = list(set(features))
    
    amur_df = pd.DataFrame()
    new_features = [] # список с финальными фичами после разворачивания датафрейма
    new_targets = [] # список с финальными таргетами после разворачивания датафрейма
    for grp_name, grp_df in hydro_df.groupby('identifier'):
        if grp_name not in ALL_STATIONS:
            continue
        new_features.extend([col + '_' + str(grp_name) for col in features])
        new_targets.append('sealevel_max_' + str(grp_name))
        if amur_df.empty:
            amur_df = grp_df[features + ['date', 'sealevel_max']].copy()
            amur_df.rename(columns={col: col + '_' + str(grp_name) for col in features + ['sealevel_max']}, inplace=True)
        else:
            part_df = grp_df[features + ['date', 'sealevel_max']].copy()
            part_df.rename(columns={col: col + '_' + str(grp_name) for col in features + ['sealevel_max']}, inplace=True)
            amur_df = amur_df.merge(part_df[[col + '_' + str(grp_name) for col in features + ['sealevel_max']] + ['date']],
                                  on='date', how='left')
        amur_df.sort_values('date', inplace=True)

        gc.collect()
        bad_cols = list(set(new_features).difference(set(NUMERICAL_FEATURES+CATEGORICAL_FEATURES)))
        amur_df.drop(bad_cols, axis=1, inplace=True)
        for col in bad_cols:
            if col in new_features:
                new_features.remove(col)
    
        count_df = amur_df.count() / len(amur_df)
        bad_cols = set(count_df[count_df < 1].index)
        amur_df.sort_values('date', inplace=True)
        for col in bad_cols:
            if (not train) or (col in new_features):
                amur_df[col] = amur_df[col].interpolate(method='linear').fillna(method='ffill').fillna(method='bfill')
    logger.info('~~~END MAKING FEATURES~~~')
    return amur_df