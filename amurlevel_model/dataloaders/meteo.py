# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
from datetime import datetime,date
from typing import Union

from ..config import DATASETS_PATH,MAX_DATE


# идентификаторы метеостанций и список полей, где плохая заполненность
BAD_IDENTIFIER_COLS = {'4923811':['soilTemperature']}


def read_history_meteo(start_date: Union[str,date,datetime] = '1980-01-01',
                       end_date: Union[str,date,datetime]=MAX_DATE) -> pd.DataFrame:
    '''
    Чтение всех исторических метео данных из meteo_new
    Все данные с недостоверным качеством заменяются на np.nan
    Пропуски для численных значений заменяются линейной интерполяцией
    Пропуски для категориальных значений заменяются по предыдущему и следующему значению
    Ресемплирование данных до суточных
    :param start_date: str,datetime,date - начальная дата, меньше которой данные выкидываем
    :param end_date: str,datetime,date - конечная дата, больше которой данные выкидываем
    :return: pd.DataFrame с историческими данными метео
    '''

    meteo_df = pd.DataFrame()
    meteo_dir = os.path.join(DATASETS_PATH,'meteo_new')
    for _f in os.listdir(meteo_dir):
        identifier = _f[0:-4]
        if _f.endswith('.csv'):
            time_cols = ['localYear' ,'localMonth' ,'localDay' ,'localTimePeriod']
            meteo_cols = ['cloudCoverTotal' ,'pastWeather' ,'presentWeather',
                          'windDirection' ,'windSpeed' ,'maximumWindGustSpeed',
                          'totalAccumulatedPrecipitation' ,'soilTemperature' ,'airTemperature',
                          'relativeHumidity' ,'pressureReducedToMeanSeaLevel',
                          'pressure']
            quality_cols = [col + 'Quality' for col in meteo_cols]
            try:
                part_meteo_df = pd.read_csv(os.path.join(meteo_dir,_f), sep=',',
                            usecols=time_cols +meteo_cols +quality_cols)
            except UnicodeDecodeError:
                part_meteo_df = pd.read_csv(os.path.join(meteo_dir,_f), encoding='cp1251',sep=',',
                            usecols=time_cols +meteo_cols +quality_cols)

            for col in BAD_IDENTIFIER_COLS.get(identifier,[]):
                meteo_cols.remove(col)
                quality_cols.remove(col+'Quality')
                part_meteo_df.drop([col,col+'Quality'],axis=1,inplace=True)

            # по температуре - берем и мин и макс
            # for col in ['airTemperature']:
            col = 'airTemperature'
            part_meteo_df[col + '_min'] = part_meteo_df[col].copy()
            part_meteo_df[col + '_minQuality'] = part_meteo_df[col + 'Quality'].copy()
            part_meteo_df.rename(columns={col :col + '_max' ,col + 'Quality' :col + '_maxQuality'} ,inplace=True)
            meteo_cols.remove(col)
            meteo_cols.extend([col + '_min' ,col + '_max'])

            # для всех значений с сомнительным качеством - зануляем
            part_meteo_df['datetime'] = pd.to_datetime(part_meteo_df.apply(lambda x: datetime(int(x['localYear']) ,int(x['localMonth']),
                                                                      int(x['localDay']) ,int(x['localTimePeriod']))
                                                   ,axis=1))
            part_meteo_df = part_meteo_df[(part_meteo_df['datetime'] >= start_date) &
                                          (part_meteo_df['datetime'] <= end_date)]
            
            # делаем линейную интерполяцию для численных, для категориальных - ffill,bfill
            part_meteo_df.sort_values('datetime' ,inplace=True)
            for col in meteo_cols:
                bad_ind = part_meteo_df[part_meteo_df[col +'Quality'].astype(np.int).isin([3 ,4 ,6 ,7])].index
                part_meteo_df.loc[bad_ind ,col] = np.nan
                if col in ['pastWeather' ,'presentWeather' ,'cloudCoverTotal']:
                    part_meteo_df[col] = part_meteo_df[col].fillna(method='ffill').fillna(method='bfill')
                else:
                    part_meteo_df[col] = part_meteo_df[col].interpolate(method='linear')
            
            # ресемплирование
            part_meteo_df.set_index('datetime' ,inplace=True)

            resample_dict = {
                'cloudCoverTotal': 'max',
                'pastWeather': 'max',
                'presentWeather': 'max',
                'windDirection': 'median',
                'windSpeed': 'max',
                'maximumWindGustSpeed': 'max',
                'totalAccumulatedPrecipitation': 'sum',
                'airTemperature_max': 'max',
                'airTemperature_min': 'min',
                'soilTemperature': 'mean',
                'relativeHumidity': 'max',
                'pressureReducedToMeanSeaLevel': 'mean',
                'pressure': 'mean'
            }
            for col in ['cloudCoverTotal','pastWeather','presentWeather']:
                part_meteo_df[col] = part_meteo_df[col].round().astype(np.int)

            dict_keys = list(resample_dict.keys())
            for col in dict_keys:
                if col not in meteo_cols:
                    del resample_dict[col]
            part_meteo_df = part_meteo_df.resample('1D').agg(resample_dict)
            part_meteo_df.reset_index(inplace=True)
            part_meteo_df['identifier'] = identifier
            part_meteo_df.sort_values('datetime' ,inplace=True)
            for col in meteo_cols:
                if col in ['pastWeather' ,'presentWeather' ,'cloudCoverTotal']:
                    part_meteo_df[col] = part_meteo_df[col].fillna(method='ffill').fillna(method='bfill')
                else:
                    part_meteo_df[col] = part_meteo_df[col].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
            meteo_df = meteo_df.append(part_meteo_df)
    meteo_df.sort_values('datetime' ,inplace=True)
    return meteo_df