import os

__all__ = [
    'MODEL_NAME',
    'MODEL_PATH',
    'MODEL_FULLPATH',
    'CIFAR10_CLASSES'
]

MODEL_NAME: str = 'cifar10_model.h5'
MODEL_PATH: str = '.\\models'
MODEL_FULLPATH = os.path.join(MODEL_PATH, MODEL_NAME)

CIFAR10_CLASSES: list[str] = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]