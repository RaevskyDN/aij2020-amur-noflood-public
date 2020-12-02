# -*- coding: utf-8 -*-

import pandas as pd

from ..dataloaders.asunp import get_asunp_stations

def merge_meteo_asunp(meteo_df: pd.DataFrame) -> pd.DataFrame:
    '''
    Мердж между историческими метео и данными по местоположению метеостанций из asunp
    :param hydro_df: pd.DataFrame, датафрейм с историей метео
    :param asunp_hydro: GeoDataFrame, данные по местоположению метеостанций
    :return: pd.DataFrame
    '''
    asunp = get_asunp_stations()
    meteo_stations = asunp[asunp.meteo.isin(meteo_df['identifier'])].drop_duplicates('meteo' ,keep='last')
    meteo_df = meteo_df.merge(meteo_stations[['lat' ,'lon' ,'meteo']] ,left_on='identifier' ,right_on='meteo')
    return meteo_df