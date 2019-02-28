# coding=utf-8

from model.densenet import DenseNet
from keras.optimizers import SGD
import keras
import tensorflow as tf
from keras.utils import multi_gpu_model
from keras.callbacks import TensorBoard, LearningRateScheduler
from model.model_saver import MultiGPUCheckpointCallback
from keras.losses import categorical_crossentropy
import argparse
from image_util import read_image
import os