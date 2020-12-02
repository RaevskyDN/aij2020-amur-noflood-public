# -*- coding: utf-8 -*-

import pandas as pd
from geopandas.geodataframe import GeoDataFrame
from typing import Optional,Union

from ..dataloaders.asunp import get_asunp_hydro_stations
from ..weatherforecast.yandex import forecast_weather_yandex
from ..weatherforecast.openweather import forecast_weather_openweather
from ..config import ALL_STATIONS,DATASETS_PATH
import os

MAX_WEATHER_COLS = ['airTemperature_max','totalAccumulatedPrecipitation',
                    'relativeHumidity','pressure'] # метеорологические параметры, заменяющиеся на max среди всех прогнозов

MIN_WEATHER_COLS = ['airTemperature_min'] # метеорологические параметры, заменяющиеся на min среди всех прогнозов

WIND_COLS = ['windSpeed','windDirection'] # характеристики ветра

def merge_weather_forecasts(yandex_df: pd.DataFrame, open_weather_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Мердж прогнозных данных от яндекс.погода и openweather

    Макс. температура воздуха, кол-во осадков, влажность, давление заменяются на макс. значения
    Мин. температура воздуха заменяется на мин. значение
    Скорость ветра и направление ветра выбираются от яндекса (если от него данные есть), иначе от openweather

    :param yandex_df: pd.DataFrame, прогноз от яндекс.погоды, полученный методом weatherforecast.yandex.forecast_weather_yandex
    :param open_weather_df: pd.DataFrame, прогноз от openweather, полученный методом weatherforecast.openweather.forecast_weather_openweather
    :return: pd.DataFrame, итоговый датафрейм с прогнозом погоды
    '''
    weather_df = open_weather_df.merge(yandex_df,on='date',suffixes=['_ow','_ya'],how='left') #именно так мержим, т.к. в openweather 10 дней, в яндексе 7 дней

    for col in MAX_WEATHER_COLS:
        weather_df[col] = weather_df[[col+'_ya',col+'_ow']].max(axis=1)
        weather_df.drop([col+'_ya',col+'_ow'],axis=1,inplace=True)

    for col in MIN_WEATHER_COLS:
        weather_df[col] = weather_df[[col+'_ya',col+'_ow']].min(axis=1)
        weather_df.drop([col+'_ya',col+'_ow'],axis=1,inplace=True)

    for col in WIND_COLS:
        weather_df[col] = weather_df[col+'_ya'].copy()
        ya_null_ind = weather_df[weather_df[col].isnull()].index
        weather_df.loc[ya_null_ind,col] = weather_df.loc[ya_null_ind,col+'_ow']
        weather_df.drop([col + '_ya', col + '_ow'], axis=1, inplace=True)

    return weather_df

def merge_hydro_weatherforecast(hydro_df: pd.DataFrame,hydro_asunp: Optional[GeoDataFrame]):
    '''
    Мердж полученного датасета из features.amur_features.make_dataset с прогнозными данными метео:

    :param hydro_df: pd.DataFrame, датафрейм с данными с гидростанций
    :param hydro_asunp: GeoDataFrame, датафрейм с гидростанциями. Если None, то выгружается по API
    :return: pd.DataFrame, смердженный датафрейм с гидро-метео данными
    '''
    if hydro_asunp is None:
        hydro_asunp = get_asunp_hydro_stations()

    full_weather_df = pd.DataFrame()
    for identifier in ALL_STATIONS:
        lat,lon = hydro_asunp.loc[identifier,['lat','lon']]
        yandex_df = forecast_weather_yandex(lat,lon)
        openweather_df = forecast_weather_openweather(lat,lon)
        weather_df = merge_weather_forecasts(yandex_df=yandex_df,
                                             open_weather_df=openweather_df)
        weather_df['identifier'] = identifier
        full_weather_df = pd.concat([full_weather_df,weather_df],ignore_index=True)

    full_weather_df.to_csv(os.path.join(DATASETS_PATH,'full_weather.csv'))
    hydro_df = pd.concat([hydro_df,full_weather_df],ignore_index=True)
    hydro_df.sort_values(['identifier','date'],inplace=True)

    return hydro_df