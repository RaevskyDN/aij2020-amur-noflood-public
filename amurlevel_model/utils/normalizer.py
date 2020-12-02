import json
import os
from typing import Optional
from ..config import DATASETS_PATH

def get_normalizer_stats(fname: Optional[str]=None) -> dict:
    '''
    Получение данных со статистикой mean,std для каждой фичи для нормализации
    :param fname: str, путь до файла json cо статистикой mean,std для каждого поля
    :return: dict
    '''
    if fname is None:
        fname = os.path.join(DATASETS_PATH,'mean_std_stats.json')
    with open(fname, 'r', encoding='utf-8') as f:
        normalizer_stats = json.loads(f.read())
    return normalizer_stats