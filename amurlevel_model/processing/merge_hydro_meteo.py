# -*- coding: utf-8 -*-

import pandas as pd
from geopandas.geodataframe import GeoDataFrame
import numpy as np
from pykrige.ok import OrdinaryKriging
from sklearn.neighbors import KDTree
import gc
from typing import Optional
import logging
from tqdm import tqdm

from ..dataloaders.asunp import get_asunp_hydro_stations

def merge_hydro_meteo(hydro_df: pd.DataFrame, meteo_df: pd.DataFrame,
                                   asunp_hydro: Optional[GeoDataFrame]=None) -> pd.DataFrame:
    '''
    Мердж данных с гидростанций с историческими метеоданными
    Для категориальных значений ищется ближайшая метеостанция
    Для численных значений применяется кригинг по трем ближайшим меоестанциям
    :param hydro_df: pd.DataFrame, датафрейм с данными с гидростанций
    :param meteo_df: pd.DataFrame, датафрейм с метео данными
    :param asunp_hydro: GeoDataFrame, датафрейм с гидростанциями. Если None, то выгружается по API
    :return: pd.DataFrame, смердженный датафрейм с гидро-метео данными
    '''
    logger = logging.getLogger()
    full_df = hydro_df.copy()

    if asunp_hydro is None:
        asunp_hydro = get_asunp_hydro_stations()
    # ---- добавляем категориальные признаки - их заменяаем по ближайшему соседу
    meteo_df['ts'] = (meteo_df['datetime'].astype(np.int64) / 10 ** 10)*2
    full_df['ts'] = (full_df['date'].values.astype(np.int64) / 10 ** 10)*2
    kd = KDTree(meteo_df[['lon', 'lat', 'ts']])
    union_ind = full_df[full_df['ts'].astype(np.int).astype(str).isin(meteo_df['ts'].astype(np.int).astype(str))].index
    indices = kd.query(full_df.loc[union_ind][['lon', 'lat', 'ts']], 1, return_distance=False)[:, 0]
    for col in ['pastWeather', 'presentWeather', 'cloudCoverTotal']:
        full_df[col] = np.nan
        full_df.loc[union_ind,col] = meteo_df.iloc[indices][col].values

    del kd
    gc.collect()

    # ---- численные признаки с помощью киргинга по 3 ближайшим соседям (как в примере решения)
    new_df = pd.DataFrame(columns=['identifier', 'date', 'windDirection', 'windSpeed', 'maximumWindGustSpeed',
                                   'totalAccumulatedPrecipitation', 'soilTemperature', 'airTemperature_min',
                                   'airTemperature_max',
                                   'relativeHumidity', 'pressureReducedToMeanSeaLevel', 'pressure'])
    logger.info('Kriging for meteo')
    for day, day_meteo in tqdm(meteo_df.groupby('datetime'),total=len(meteo_df['datetime'].unique())):
        part_df = pd.DataFrame(columns=['identifier', 'date', 'windDirection', 'windSpeed', 'maximumWindGustSpeed',
                                        'totalAccumulatedPrecipitation', 'soilTemperature', 'airTemperature_min',
                                        'airTemperature_max',
                                        'relativeHumidity', 'pressureReducedToMeanSeaLevel', 'pressure'])
        part_df['identifier'] = asunp_hydro.index.values
        part_df['date'] = [day] * len(part_df)
        for prop in ['windDirection', 'windSpeed', 'maximumWindGustSpeed',
                     'totalAccumulatedPrecipitation', 'soilTemperature', 'airTemperature_min', 'airTemperature_max',
                     'relativeHumidity', 'pressureReducedToMeanSeaLevel', 'pressure']:
            longitudes = day_meteo[day_meteo[prop].notnull()].lon
            latitudes = day_meteo[day_meteo[prop].notnull()].lat
            values = day_meteo[day_meteo[prop].notnull()][prop].values
            if values.max() == values.min():
                interpolated_values = np.full(len(asunp_hydro), values.mean())
            else:
                OK = OrdinaryKriging(
                    longitudes,
                    latitudes,
                    values,
                    variogram_model='spherical',
                    coordinates_type="geographic")
                interpolated_values, ss1 = OK.execute("points", asunp_hydro.lon, asunp_hydro.lat, backend="C", n_closest_points=3)
            part_df[prop] = interpolated_values
        new_df = new_df.append(part_df)
        gc.collect()
    full_df = full_df.merge(new_df, on=['identifier', 'date'], how='left')
    new_df = new_df[new_df['date'] > full_df['date'].max()]
    if not new_df.empty:
        full_df = pd.concat([full_df,new_df],ignore_index=True)
    return full_df