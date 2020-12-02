# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K

from .metrics import rmse,mae,mae_inference
from ..config_features import NUMERICAL_FEATURES,CATEGORICAL_FEATURES,CAT_MAP
from ..config import DAYS_FORECAST,ALL_STATIONS


def lstm_layer(hidden_dim, dropout):
    return L.Bidirectional(
        L.LSTM(hidden_dim,
                             dropout=dropout,
                             return_sequences=True,
                             kernel_initializer='orthogonal'))

class Conv1BN():
    '''
    Архитектура - conv1D->Batchnorm->conv1D->Batchnorm->Droput
    :param filters: int, число фильтров
    :param input_shape: tuple, входной shape
    :param kernel_size: int, размер ядра для 1D свертки
    :param dilation_rate: int, dilation для 1D свертки
    '''
    def __init__(self,filters,input_shape,kernel_size,dilation_rate):
        self.conv1 = L.Conv1D(filters=filters,input_shape=input_shape,padding='same',
                                            kernel_size=kernel_size,dilation_rate=dilation_rate)
        self.drop1 = L.Dropout(0.1)
        self.bn1 = L.BatchNormalization()
        self.conv2 = L.Conv1D(filters=filters,padding='same',
                                            kernel_size=kernel_size,dilation_rate=dilation_rate)
        self.bn2 = L.BatchNormalization()

    def __call__(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.drop1(x)
        return x


def build_model(dropout=0.35, hidden_dim=256, embed_dim=1,
                numerical=len(NUMERICAL_FEATURES),
                categorical=len(CATEGORICAL_FEATURES)):
    '''
    Итоговая модель для предсказания уровня

    :param dropout: float, dropout используемый в модели
    :param hidden_dim: int, размерность скрытого слоя в LSTM
    :param embed_dim: int, размерность эмбединга для категориальных признаков
    :param numerical: int, количество численных признаков
    :param categorical: int, количество категориальных признаков
    :return: model
    '''
    K.clear_session()
    inputs = L.Input(shape=(DAYS_FORECAST, numerical + categorical))
    num_inputs = inputs[:, :, :numerical]

    if categorical > 0:
        embed_inputs = inputs[:, :, numerical:numerical + categorical]
        embed = L.Embedding(input_dim=len(CAT_MAP), output_dim=embed_dim)(embed_inputs)
        conv_embed_inputs = tf.reshape(embed, shape=(-1, embed.shape[1], embed.shape[2] * embed.shape[3]))

    in1 = L.Dense(1000, activation='relu')(num_inputs)
    in1 = L.BatchNormalization()(in1)
    dropped1 = L.Dropout(dropout)(in1)

    if categorical > 0:
        reshaped = tf.concat([conv_embed_inputs, dropped1], axis=2)
    else:
        reshaped = dropped1

    #print(f'reshaped shape {reshaped.shape}')

    hidden = lstm_layer(hidden_dim, dropout)(reshaped)
    hidden = lstm_layer(hidden_dim, dropout)(hidden)

    #print(hidden.shape)

    hidden = L.BatchNormalization()(hidden)
    out1 = L.Dense(800, activation='relu')(hidden)
    out1 = L.BatchNormalization()(out1)
    dropped = L.Dropout(dropout)(out1)

    out1 = L.Dense(800, activation='relu')(dropped)
    out1 = L.BatchNormalization()(out1)

    dropped2 = L.Dropout(dropout)(num_inputs)

    convbn = L.Dropout(dropout)(num_inputs)
    convbn = Conv1BN(filters=1000, kernel_size=3, dilation_rate=1, input_shape=convbn.shape)(convbn)

    # out1 = tf.concat([out1, dropped2], axis=2)
    out1 = tf.concat([out1, dropped2, convbn], axis=2)

    out = L.Dense(len(ALL_STATIONS), activation='linear')(out1)
    model = tf.keras.Model(inputs=inputs, outputs=out)

    adam = tf.optimizers.Adam(learning_rate=0.00008)

    model.compile(optimizer=adam, loss=rmse,metrics=[mae_inference, mae])

    return model