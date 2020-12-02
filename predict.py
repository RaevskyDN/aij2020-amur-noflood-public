# -*- coding: utf-8 -*-
from argparse import RawTextHelpFormatter, ArgumentParser
import pandas as pd
import os
from datetime import timedelta, datetime

from amurlevel_model.config import DAYS_FORECAST, ALL_STATIONS, NUMBER_OF_INFERENCE_STATIONS, DATASETS_PATH

from amurlevel_model.dataloaders.asunp import get_asunp_hydro_stations
from amurlevel_model.dataloaders.meteo import read_history_meteo
from amurlevel_model.dataloaders.hydro import read_hydro_all

from amurlevel_model.processing.merge_hydro_meteo import merge_hydro_meteo
from amurlevel_model.processing.merge_hydro_weatherforecast import merge_hydro_weatherforecast
from amurlevel_model.processing.merge_hydro_asunp import merge_hydro_asunp
from amurlevel_model.processing.merge_meteo_asunp import merge_meteo_asunp

from amurlevel_model.features.amur_features import make_dataset

from amurlevel_model.model.prepare_data import prepare_data
from amurlevel_model.model.model import build_model
from amurlevel_model.utils.common import set_logger

class EmptyHistoricalMeteo(Exception):
    pass

class EmptyHistoricalHydro(Exception):
    pass

def parse_args():
    parser = ArgumentParser(description='''
        Скрипт для проверки качества обученной модели для предсказания уровня воды.
        Предсказывается на 10 дней вперед от f_day
        Результаты сохраняются в файл level_{f_day}.csv

        Пример:  python predict.py -f_day 2020-11-01 -w /data/weights-aij2020amurlevel-2017.h5 (предсказания будут от 2020-11-01 до 2020-11-11 по модели обученной до 2018 года)
        python predict.py -f_day 2013-02-01 -w /data/weights-aij2020amurlevel-2012.h5 (предсказания будут от 2013-02-01 до 2013-02-11 по модели обученной до 2013 года)
        ''', formatter_class=RawTextHelpFormatter)

    parser.add_argument('-f_day', type=str, required=True,
                        help='Дата от которой считаем предсказания')
    parser.add_argument('-w','--weights', dest='w',type=str, required=True,
                        help='Путь до файла с весами модели')
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    logger = set_logger()
    args = parse_args()
    model = build_model()
    f_day = pd.to_datetime(args.f_day)
    l_day = f_day + timedelta(DAYS_FORECAST)
    l_day_history = f_day - timedelta(1)
    f_day_hydro = f_day - timedelta(DAYS_FORECAST + 4 * 365 + 1)  # объем данных необходимый для данных с гидространций
    f_day_meteo = f_day - timedelta(DAYS_FORECAST + 60)  # объем данных необходимый для метео
    l_day_meteo = min(datetime.today().date()-timedelta(1),l_day) # выгружаем исторические метео макс. до вчерашнего дня
    logger.info('prediction. Period ' + f_day.strftime('%Y-%m-%d') + ' - ' + (l_day-timedelta(days=1)).strftime('%Y-%m-%d'))

    asunp_hydro = get_asunp_hydro_stations()
    hydro_df = read_hydro_all(start_date=f_day_hydro, end_date=l_day_history)  # выгружаем данные с гидропостов
    logger.info(f'hydro shape {hydro_df.shape}')
    if hydro_df.empty:
        raise EmptyHistoricalHydro(f"There is no historical hydro data from {f_day_hydro.strftime('%Y-%m-%d')} - {f_day.strftime('%Y-%m-%d')}!")
    elif hydro_df['date'].max() < l_day_history:
        logger.warning(f"The last max date in hydro is {hydro_df['date'].max().strftime('%Y-%m-%d')} and prediction period is {f_day.strftime('%Y-%m-%d')} - {(l_day-timedelta(days=1)).strftime('%Y-%m-%d')}. You should update your historical hydro data or results will be worse")

    logger.info(f"unique identifiers {len(hydro_df['identifier'].unique())}")
    meteo_df = read_history_meteo(start_date=f_day_meteo, end_date=l_day_meteo)  # выгружаем метео
    logger.info(f'meteo shape {meteo_df.shape}')
    if meteo_df.empty:
        raise EmptyHistoricalMeteo(f"There is no historical meteo data from {f_day_meteo.strftime('%Y-%m-%d')} - {l_day.strftime('%Y-%m-%d')}!")
    elif meteo_df['datetime'].max() < l_day_history:
        logger.warning(f"The last max date ine meteo is {meteo_df['date'].max().strftime('%Y-%m-%d')} and prediction period is {f_day.strftime('%Y-%m-%d')} - {(l_day-timedelta(days=1)).strftime('%Y-%m-%d')}. You should update your historical meteo data or results will be worse")

    meteo_df = merge_meteo_asunp(meteo_df)
    logger.info(f'meteo shape after merge with asunp {meteo_df.shape}')
    hydro_df = merge_hydro_asunp(hydro_df, asunp_hydro)  # мержим гидро и asunp
    logger.info(f'hydro shape after merge with asunp {hydro_df.shape}')
    hydro_df = merge_hydro_meteo(hydro_df, meteo_df, asunp_hydro)  # мержим гидро и метео
    logger.info(f'hydro shape after merge with historical meteo {hydro_df.shape}')

    if l_day >= datetime.today().date():  # если прогнозируем будущее - добавляем прогноз погоды
        logger.info('Use weather forecast')
        hydro_df = merge_hydro_weatherforecast(hydro_df, asunp_hydro)
        logger.info(f'hydro shape after merge with forecast weather {hydro_df.shape}')

    amur_df = make_dataset(hydro_df)

    logger.info(f'amur_df shape {amur_df.shape}')
    inputs = prepare_data(amur_df, start_date=f_day, end_date=l_day)  # подготовка данных в формате модели
    logger.info(f'model inputs shape {inputs.shape}')

    # предсказание модели
    model = build_model()
    try:
        model.load_weights(args.w)
    except:
        model_fn = os.path.join(DATASETS_PATH, args.w)
        logger.info(f'There is no file {args.w}. Try to load from {model_fn}')
        model.load_weights(model_fn)

    preds = model.predict(inputs).reshape(-1,len(ALL_STATIONS))
    results_df = pd.DataFrame({'date': pd.date_range(f_day, l_day - timedelta(days=1), freq='1D')})

    for i in range(NUMBER_OF_INFERENCE_STATIONS):
        results_df[ALL_STATIONS[i]] = preds[:, i]
    fname = os.path.join(DATASETS_PATH, f"level_{f_day.strftime('%Y-%m-%d')}_{(l_day-timedelta(days=1)).strftime('%Y-%m-%d')}.csv")
    results_df.to_csv(fname, index=False)
    logger.info(f'Results saved to {fname}')