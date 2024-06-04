import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import os

from PIL import Image

import tensorflow as tf
from keras.datasets import cifar10
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense
from keras.utils import to_categorical

__all__ = ['model']


def model(model_name: str):
    if os.path.exists(model_name):
        return load_model(model_name)
    
    (x_train, y_train), (x_val, y_val) = cifar10.load_data()
    x_train, x_val = x_train / 255, x_val / 255
    y_train, y_val = to_categorical(y_train, 10), to_categorical(y_val, 10)

    model = Sequential([
        Flatten(input_shape=(32, 32, 3)),
        Dense(1000, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    model.fit(
        x_train,
        y_train,
        batch_size=64,
        epochs=10,
        validation_data=(x_val, y_val)
    )
    model.save(model_name)
    return model
