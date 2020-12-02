# -*- coding: utf-8 -*-

import requests
import pandas as pd
import numpy as np
from typing import Union
import json
import logging

from ..config import YANDEX_KEY,YANDEX_URL

WIND_DIRECTIONS = {'sw':225,'n':0,'nw':315,'ne':45,'e':90,'s':180,'w':270,'se':135}

def forecast_weather_yandex(lat: Union[float,str],lon: Union[float,str]) -> pd.DataFrame:
    '''
    Прогноз погоды с помощью сервиса Яндекс.погода
    Преобразуем в датафрейм по формате из meteo_new
    :param lat: float,str широта для запроса
    :param lon: float,str, долгота для запроса
    :return: pd.DataFrame
    '''

    params = dict(lat=str(lat), lon=str(lon), extra='true')
    logger = logging.getLogger()
    logger.info(f'GET {YANDEX_URL}\trequest params {json.dumps(params)}..')

    weather_json = requests.get(YANDEX_URL,params=params,
                                headers={'X-Yandex-API-Key':YANDEX_KEY}).json()

    forecast_list = []
    for day_f in weather_json['forecasts']:
        temp_min = 100
        temp_max = -100
        prec_sum = 0
        rel_hum = None
        wind_speed = 0
        pressure = 0
        wind_dir = []
        for i in ['evening','morning','night','day']:
            temp_min = min(temp_min,day_f ['parts'][i]['temp_min'])
            temp_max = max(temp_max,day_f ['parts'][i]['temp_max'])
            prec_sum += day_f ['parts'][i]['prec_mm']
            if rel_hum is None:
                rel_hum = day_f ['parts'][i].get('humidity',0.0)
            else:
                rel_hum = max(rel_hum,day_f ['parts'][i].get('humidity',0.0))
            wind_speed += day_f ['parts'][i].get('wind_speed',0)
            pressure += day_f ['parts'][i].get('pressure_pa',1002)
            wind_dir.append(WIND_DIRECTIONS[day_f ['parts'][i]['wind_dir']])
        forecast_list.append({'airTemperature_max':temp_max,'airTemperature_min':temp_min,
                      'totalAccumulatedPrecipitation':prec_sum,'date':pd.to_datetime(day_f ['date']),
                     'relativeHumidity':rel_hum,'windSpeed':wind_speed/4,'windDirection':np.median(wind_dir),
                     'pressure':pressure/4})

    return pd.DataFrame(forecast_list)