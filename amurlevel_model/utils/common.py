from datetime import datetime
import logging
import sys

def dateparse(date_str: str) -> datetime:
    '''
    В исходных данных есть очень большие даты, которые нельзя распарсить. Для них возвращает дефолт.
    :param x:
    :return:
    '''
    try:
        return datetime.strptime(date_str, '%d.%m.%Y')
    except:
        return datetime(2100, 12, 31)

def set_logger():
    '''
    Формат логирования
    Выводим все в stdout
    :return: logger
    '''
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logging.getLogger()