# -*- coding: utf-8 -*-

import requests
import pandas as pd
import datetime
import json
from typing import Union
import logging

from ..config import OPEN_WEATHER_HOST,OPEN_WEATHER_KEY,OPEN_WEATHER_URL

def forecast_weather_openweather(lat: Union[float,str],lon: Union[float,str]) -> pd.DataFrame:
    '''
    Прогноз погоды с помощью сервиса Openweather
    Преобразуем в датафрейм по формате из meteo_new
    :param lat: float,str широта для запроса
    :param lon: float,str, долгота для запроса
    :return: pd.DataFrame
    '''

    params = {"lat":str(lat),"lon":str(lon),"cnt":"10","units":"metric","mode":"JSON","lang":"ru"}

    headers = {'x-rapidapi-key': OPEN_WEATHER_KEY,
               'x-rapidapi-host': OPEN_WEATHER_HOST}

    logger = logging.getLogger()
    logger.info(f'GET {OPEN_WEATHER_URL}\trequest params {json.dumps(params)}..')
    weather_json = requests.get(OPEN_WEATHER_URL, headers=headers, params=params).json()

    forecast_list = []
    for day_f in weather_json['list']:
        temp_min = day_f['temp']['min']
        temp_max = day_f['temp']['max']
        prec_sum = day_f.get('rain', 0) + day_f.get('snow', 0)
        rel_hum = day_f['humidity']
        wind_speed = day_f['speed']
        pressure = day_f['pressure']
        wind_dir = day_f['deg']
        forecast_list.append({'airTemperature_max': temp_max, 'airTemperature_min': temp_min,
                      'totalAccumulatedPrecipitation': prec_sum,
                      'date': pd.to_datetime(datetime.datetime.fromtimestamp(day_f['dt']).date()),
                      'relativeHumidity': rel_hum, 'windSpeed': wind_speed, 'windDirection': wind_dir,
                      'pressure': pressure})

    return pd.DataFrame(forecast_list)
