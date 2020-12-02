# -*- coding: utf-8 -*-

import tensorflow as tf
from ..config import NUMBER_OF_INFERENCE_STATIONS

def mae(y_true, y_pred):
    '''
    Метрика mean_absolute_error
    '''
    _mae = tf.abs(tf.subtract(y_pred, y_true))
    return tf.math.reduce_mean(tf.math.reduce_mean(_mae,axis=1),axis=1)


def mae_inference(y_true, y_pred):
    '''
    Метрика mean_absolute_error для станций, которые участвуют в инференсе
    '''
    _mae = tf.abs(tf.subtract(y_pred[:,:,:NUMBER_OF_INFERENCE_STATIONS], y_true[:,:,:NUMBER_OF_INFERENCE_STATIONS]))
    return tf.math.reduce_mean(tf.math.reduce_mean(_mae,axis=1),axis=1)


def rmse(y_true, y_pred):
    '''
    colwise rmse для оптимизации
    '''
    colwise_mse = tf.reduce_mean(tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred),axis=1)),axis=1)
    return colwise_mse

def mae_by_stations(y_true, y_pred):
    '''
    mean absolute error в разрезе станций
    '''
    _mae = tf.abs(tf.subtract(y_pred, y_true))
    return tf.math.reduce_mean(tf.math.reduce_mean(_mae,axis=1),axis=0)

def mae_by_time(y_true, y_pred):
    '''
    mean absolute error в разрезе дня прогноза
    '''
    _mae = tf.abs(tf.subtract(y_pred, y_true))
    return tf.math.reduce_mean(tf.math.reduce_mean(_mae,axis=2),axis=0)