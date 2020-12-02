from argparse import RawTextHelpFormatter, ArgumentParser
import pandas as pd
import tensorflow as tf

from amurlevel_model.dataloaders.asunp import get_asunp_hydro_stations
from amurlevel_model.dataloaders.meteo import read_history_meteo
from amurlevel_model.dataloaders.hydro import read_hydro_all

from amurlevel_model.processing.merge_hydro_meteo import merge_hydro_meteo
from amurlevel_model.processing.merge_hydro_asunp import merge_hydro_asunp
from amurlevel_model.processing.merge_meteo_asunp import merge_meteo_asunp

from amurlevel_model.features.amur_features import make_dataset

from amurlevel_model.model.train_test_split import train_test_split
from amurlevel_model.model.model import build_model

from amurlevel_model.config import BATCH_SIZE,EPOCHS
from amurlevel_model.utils.common import set_logger

LAST_DAY = '2021-01-01' # дата до которой у нас есть данные в выборке, больше не выгружаем


def parse_args():
    parser = ArgumentParser(description='''
        Скрипт для обучения модели

        Пример:  python train.py -t_day 2015-01-01 -w /data/weights_2015.h5 (предсказания будут от 2020-11-01 до 2020-11-11)
        ''', formatter_class=RawTextHelpFormatter)

    parser.add_argument('-t_day', type=str, required=True,
                        help='Дата до которой включаем в обучающую выборку, после этой даты - тестовая')

    parser.add_argument('-w', type=str, required=True,
                        help='Название файла куда сохраним веса модели (в формате .h5)')

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    logger = set_logger()
    args = parse_args()
    f_day = pd.to_datetime(args.t_day)
    l_day = pd.to_datetime(LAST_DAY)

    asunp_hydro = get_asunp_hydro_stations()
    logger.info('training')
    hydro_df = read_hydro_all(end_date=l_day)  # выгружаем данные с гидропостов
    logger.info(f'hydro shape {hydro_df.shape}')
    logger.info(f"unique identifiers {len(hydro_df['identifier'].unique())}")
    meteo_df = read_history_meteo(end_date=l_day)  # выгружаем метео
    logger.info(f'meteo shape {meteo_df.shape}')

    meteo_df = merge_meteo_asunp(meteo_df)
    logger.info(f'meteo shape after merge with asunp {meteo_df.shape}')
    hydro_df = merge_hydro_asunp(hydro_df, asunp_hydro)  # мержим гидро и asunp
    logger.info(f'hydro shape after merge with asunp {hydro_df.shape}')
    hydro_df = merge_hydro_meteo(hydro_df, meteo_df, asunp_hydro)  # мержим гидро и метео
    logger.info(f'hydro shape after merge with historical meteo {hydro_df.shape}')

    amur_df = make_dataset(hydro_df,train=True)

    logger.info(f'amur_df shape {amur_df.shape}')
    X_train, y_train, X_test, y_test = train_test_split(amur_df, start_test_date=f_day, end_test_date=l_day)  # подготовка данных в формате модели
    logger.info(f'train shape {X_train.shape}')
    logger.info(f'test shape {X_test.shape}')

    # обучение модели
    model_fn = args.w

    model = build_model()
    sv_model = tf.keras.callbacks.ModelCheckpoint(model_fn, save_best_only=True, monitor='val_mae_inference')
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(factor=0.3)

    history_model = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[lr_callback, sv_model],
        verbose=2
    )

    logger.info(
        f"Min training loss={min(history_model.history['loss'])}, min validation loss={min(history_model.history['val_loss'])}")