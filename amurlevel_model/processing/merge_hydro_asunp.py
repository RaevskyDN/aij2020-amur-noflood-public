# -*- coding: utf-8 -*-

import pandas as pd
from geopandas.geodataframe import GeoDataFrame
from typing import Optional

from ..dataloaders.asunp import get_asunp_hydro_stations

def merge_hydro_asunp(hydro_df: pd.DataFrame, asunp_hydro: Optional[GeoDataFrame]=None) -> pd.DataFrame:
    '''
    Мердж между данными с гидропостов и данными по местоположению гидропостов из asunp
    :param hydro_df: pd.DataFrame, датафрейм с данными из гидропостов
    :param asunp_hydro: GeoDataFrame, данные по местоположению гидропостов
    :return: pd.DataFrame
    '''
    if asunp_hydro is None:
        asunp_hydro = get_asunp_hydro_stations()
    return hydro_df.merge(asunp_hydro[['lat', 'lon']], left_on='identifier', right_index=True)