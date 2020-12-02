# -*- coding: utf-8 -*-

import geopandas as gpd
from geopandas.geodataframe import GeoDataFrame

from ..config import ASUNP_API_URL,ALL_STATIONS

def get_asunp_stations() -> GeoDataFrame:
    '''
    По API получает данные о всех cтанциях из ASUNP
    :return: GeoDataFrame
    '''
    asunp = gpd.read_file(ASUNP_API_URL, driver='GeoJSON')
    return asunp

def get_asunp_hydro_stations() -> GeoDataFrame:
    '''
    Фильтрация только по необходимым гидропостам
    :return: GeoDataFrame
    '''
    asunp = get_asunp_stations()
    hydro_posts = asunp[asunp.gidro.isin(ALL_STATIONS) &
            (asunp.ktoCategory.isin(['post_gidro', 'station_gidro']))]\
                     .sort_values(by=['lon'])\
                     .set_index('gidro', drop=False)
    hydro_posts = hydro_posts[~hydro_posts.index.duplicated(keep='first')]
    return hydro_posts
