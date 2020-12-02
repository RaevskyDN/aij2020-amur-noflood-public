# -*- coding: utf-8 -*-

import pandas as pd
import os
import logging
from ..config import ALL_STATIONS,DATASETS_PATH,MAX_DATE
from ..utils.common import dateparse
from typing import Union
from datetime import date,datetime
from dateutil.relativedelta import relativedelta

MAX_LEVEL_DIFF = 1000 # максимальное значение разности между уровнями, при котором считаем данные невалидными

def read_hydro_level_data(fname: str, start_date: Union[str,date,datetime] = '1980-01-01',
                          end_date: Union[str,date,datetime]=MAX_DATE) -> pd.DataFrame:
    '''
    Чтение исходных данных из .csv файла с информацией по уровню воды
    Ресемплирование данных до суточных
    :param fname: str, путь до файла с данными
    :param start_date: str,datetime,date - начальная дата, меньше которой данные выкидываем
    :param end_date: str,datetime,date - конечная дата, больше которой данные выкидываем
    :return: pd.DataFrame
    '''
    try:
        df = pd.read_csv(fname,
                         sep=';',
                         dtype={'water_code': str},
                         skiprows=2, index_col=0,
                         names=['date', 'sealevel_avg', 'sealevel_min', 'sealevel_max', 'water_temp',
                                'water_code', '_'],
                         usecols=['date', 'sealevel_avg', 'sealevel_min', 'sealevel_max',
                                  'water_temp', 'water_code'],
                         date_parser=dateparse,
                         skipinitialspace=True
                         )
    except UnicodeDecodeError:
        df = pd.read_csv(fname,
                         encoding='cp1251',
                         sep=';',
                         dtype={'water_code': str},
                         skiprows=2, index_col=0,
                         names=['date', 'sealevel_avg', 'sealevel_min', 'sealevel_max', 'water_temp',
                                'water_code', '_'],
                         usecols=['date', 'sealevel_avg', 'sealevel_min', 'sealevel_max',
                                  'water_temp', 'water_code'],
                         date_parser=dateparse,
                         skipinitialspace=True
                         )

    df = df[(df.index <= end_date) & (df.index >= start_date)]
    df['sealevel_max'] = pd.to_numeric(df['sealevel_max'], errors='coerce')
    df['sealevel_min'] = pd.to_numeric(df['sealevel_min'])
    df['water_temp'] = pd.to_numeric(df['water_temp'].fillna('').astype(str).str.extract(r'(\d+\.{0,1}\d{0,2})')[0])
    df['water_code'] = df.water_code.str.strip().str.split(', ').astype(str)
    unique_vals = {val: n for n, val in enumerate(df['water_code'].unique())}
    df['water_code'] = df['water_code'].map(unique_vals)
    df = df[~df.index.duplicated(keep='last')]
    df = df.resample('D').agg({
        'sealevel_max': 'max',
        'sealevel_min': 'min',
        'water_temp': 'mean',
        'water_code': 'median'
    })

    # зануляем неадекватные значения - если уровень резко меняется
    df.sort_index(inplace=True)
    df = df[df['sealevel_max'].diff().fillna(0) <= MAX_LEVEL_DIFF]
    return df

def read_hydro_disch_data(fname: str, start_date: Union[str,date,datetime] = '1980-01-01',
                          end_date: Union[str,date,datetime]=MAX_DATE) -> pd.DataFrame:
    '''
    Чтение исходных данных из .csv файла с информацией по суточному расходу воды
    :param fname: str, путь до файла с данными
    :param start_date: str,datetime,date - начальная дата, меньше которой данные выкидываем
    :param end_date: str,datetime,date - конечная дата, больше которой данные выкидываем
    :return: pd.DataFrame
    '''
    try:
        df = pd.read_csv(fname,
                             sep=';', names=['date', 'water_flow', '_'], usecols=['date', 'water_flow'],
                             skiprows=2, index_col=0, date_parser=dateparse)
    except UnicodeDecodeError:
        df = pd.read_csv(fname, encoding='cp1251',
                             sep=';', names=['date', 'water_flow', '_'], usecols=['date', 'water_flow'],
                             skiprows=2, index_col=0, date_parser=dateparse)

    df = df[(df.index <= end_date) & (df.index >= start_date)]
    df['water_flow'] = pd.to_numeric(df['water_flow'], errors='coerce')
    return df

def read_hydro_snow_ice_data(fname: str, start_date: str = '1980-01-01', end_date: str=MAX_DATE) -> pd.DataFrame:
    '''
    Чтение исходных данных из .csv файла с информацией по снежнову и ледовому покровам
    Ресемплирование данных до суточных
    :param fname: str, путь до файла с данными
    :param start_date: str,datetime,date - начальная дата, меньше которой данные выкидываем
    :param end_date: str,datetime,date - конечная дата, больше которой данные выкидываем
    :return: pd.DataFrame
    '''
    try:
        df = pd.read_csv(fname,
                         sep=';', names=['date', 'ice_thickness', 'snow_height', 'ice_place', '_'],
                         usecols=['date', 'ice_thickness', 'snow_height', 'ice_place'],
                         skiprows=2, index_col=0, date_parser=dateparse)
    except UnicodeDecodeError:
        df = pd.read_csv(fname, encoding='cp1251',
                         sep=';', names=['date', 'ice_thickness', 'snow_height', 'ice_place', '_'],
                         usecols=['date', 'ice_thickness', 'snow_height', 'ice_place'],
                         skiprows=2, index_col=0, date_parser=dateparse)

    df = df[(df.index <= end_date) & (df.index >= start_date)]
    df['ice_thickness'] = pd.to_numeric(df['ice_thickness'], errors='coerce')
    df['snow_height'] = pd.to_numeric(df['snow_height'], errors='coerce')
    df = df.resample('D').agg({
        'ice_thickness': 'max',
        'snow_height': 'max'
    })
    return df

def read_hydro_archive(identifier: str, start_date: Union[str,date,datetime] = '1980-01-01',
                       end_date: Union[str,date,datetime]=MAX_DATE) -> pd.DataFrame:
    '''
    Мердж всех данных (уровень, суточный расход воды и снежный покров) с определенного гидропоста.
    Если данных по суточному расходу воды и/или снежному покрову нет, они игнорируются

    :param identifier: str, идентификатор гидропоста
    :param start_date: str,datetime,date - начальная дата, меньше которой данные выкидываем
    :param end_date: str,datetime,date - конечная дата, больше которой данные выкидываем
    :return: pd.DataFrame
    '''
    logger = logging.getLogger()
    hydro_level_fname = os.path.join(DATASETS_PATH,'hydro',f'{identifier}_daily.csv')
    df = read_hydro_level_data(hydro_level_fname,start_date,end_date)

    hydro_snow_ice_fname = os.path.join(DATASETS_PATH,'hydro',f'{identifier}_ice.csv')
    try:
        snow_ice_df = read_hydro_snow_ice_data(hydro_snow_ice_fname,start_date,end_date)
        df = df.merge(snow_ice_df, how='left', left_index=True, right_index=True)
    except FileNotFoundError:
        logger.warning(f"File not found {hydro_snow_ice_fname}")

    hydro_water_fname = os.path.join(DATASETS_PATH,'hydro',f'{identifier}_disch_d.csv')
    try:
        wat_df = read_hydro_disch_data(hydro_water_fname,start_date,end_date)
        df = df.merge(wat_df, how='left', left_index=True, right_index=True)
    except FileNotFoundError:
        logger.warning(f"File not found {hydro_water_fname}")

    df['identifier'] = identifier
    return df.sort_index().reset_index()


def read_hydro_all(start_date: Union[str,date,datetime] = '1980-01-01',
                   end_date: Union[str,date,datetime]=MAX_DATE) -> pd.DataFrame:
    '''
    Чтение всех входных данных по гидро по всем станциям из ALL_STATIONS
    :param start_date: str,datetime,date - начальная дата, меньше которой данные выкидываем
    :param end_date: str,datetime,date - конечная дата, больше которой данные выкидываем
    :return: pd.DataFrame, данные с информацией с гидропостов
    '''
    full_df = pd.DataFrame()
    for _id in ALL_STATIONS:
        df = read_hydro_archive(_id,start_date,end_date)
        full_df = pd.concat([full_df, df], ignore_index=True)
    full_df.drop_duplicates(['date','identifier'],keep='last',inplace=True)
    full_df.sort_values('date',inplace=True)
    return full_df