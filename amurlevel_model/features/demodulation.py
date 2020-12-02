# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import math
from typing import Tuple,List

def demodul(x: List[float], T: int) -> Tuple[List[float],List[float],List[float]]:
    '''
    Метод частотно-фазовой демодуляции для выделения периодических колебаний с периодом T
    Нужен для восстановления периодических составляющих с заданным периодом T из исходного ряда
    :param x: List[float], исходный ряд с наблюдениями
    :param T: int, период колебаний
    :return: tuple:
                  A - восстановленная динамическая амплитуда
                  tetta - восстановленная динамическая фаза
                  demodul - восстановленный ряд

    demodul == A*sin(tetta+(pi*2*T/365))
    '''
    m = int(1 / T)
    omega = T
    sealevelcos = np.array([x[i] * math.cos(omega * i * 2.0 * math.pi) for i in range(len(x))])
    sealevelsin = np.array([x[i] * math.sin(omega * i * 2.0 * math.pi) for i in range(len(x))])
    macos = np.zeros(len(x))
    masin = np.zeros(len(x))
    A = np.zeros(len(x))
    demodul = np.zeros(len(x))
    tetta = np.zeros(len(x))

    for i in range(len(x)):
        if i >= 2 * m - 2:
            a = 0.0
            b = 0.0
            for n, k in enumerate(range(i - 2 * m - 2, i)):
                if k == i - 2 * m - 2 or k == i - 1:
                    a += 0.5 * sealevelcos[k]
                    b += 0.5 * sealevelsin[k]
                elif k < len(x):
                    a += sealevelcos[k]
                    b += sealevelsin[k]
                elif k >= len(x):
                    a += sealevelcos[k - len(x)]
                    b += sealevelsin[k - len(x)]
        else:
            a = 0.0
            b = 0.0
            for n, k in enumerate(range(i, 2 * m + 2 + i)):
                if k == i or k == i + 1 + 2 * m:
                    a += 0.5 * sealevelcos[k]
                    b += 0.5 * sealevelsin[k]
                elif k >= 0:
                    a += sealevelcos[k]
                    b += sealevelsin[k]
                elif k < 0:
                    a += sealevelcos[len(x) + k]
                    b += sealevelsin[len(x) + k]
                    # print(n,2*m+1)
        macos[i] = a / (2 * m + 1)
        masin[i] = b / (2 * m + 1)
        A[i] = 2 * math.sqrt(macos[i] ** 2 + masin[i] ** 2)
        tetta[i] = np.arctan2(masin[i], macos[i])
        demodul[i] = 2.0 * masin[i] * math.sin(i * 2 * math.pi * omega) + 2.0 * macos[i] * math.cos(i * 2 * math.pi * omega)
    return A, tetta, demodul


def level_demodul_365(x: pd.Series) -> pd.Series:
    '''
    Применение метода демодуляции для периода T=365 дней
    :param x: pd.Series - исходный ряд с наблюдениями из общего датафрейма
    :return: pd.Series - восстановленный ряд для T=365 дней
    '''
    vals = (x - x.mean()).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill').values
    A, tetta, B = demodul(vals, 1 / 365)
    return pd.Series(B + x.mean(), index=x.index)


def level_demodul_121(x: List[float]) -> pd.Series:
    '''
    Применение метода демодуляции для периода T=121 дней
    :param x: pd.Series - исходный ряд с наблюдениями из общего датафрейма
    :return: pd.Series - восстановленный ряд для T=121 дней
    '''
    vals = (x - x.mean()).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill').values
    A, tetta, B = demodul(vals, 1 / 121)
    return pd.Series(B + x.mean(), index=x.index)