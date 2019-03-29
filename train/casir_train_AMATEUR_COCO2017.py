import luccauchon.data.__MYENV__ as E
import logging

E.APPLICATION_LOG_LEVEL = logging.INFO

import os

os.environ['basedir_a'] = '/gpfs/home/cj3272/tmp/'
os.environ["CUDA_VISIBLE_DEVICES"] = '3'



from luccauchon.data.Generators import AmateurDataFrameDataGenerator
import luccauchon.data.Generators as generators

from segmentation_models import Unet
from segmentation_models import FPN
from segmentation_models import PSPNet
from segmentation_models import Linknet
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.losses import cce_jaccard_loss
from segmentation_models.metrics import iou_score
from segmentation_models.metrics import jaccard_score

import keras
import PIL
import numpy as np
import scipy

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

print('keras.__version__=' + str(keras.__version__))
print('tf.__version__=' + str(tf.__version__))
print('PIL.__version__=' + str(PIL.__version__))
print('np.__version__=' + str(np.__version__))
print('scipy.__version__=' + str(scipy.__version__))
print('Using GPU ' + str(os.environ["CUDA_VISIBLE_DEVICES"]) + '  Good luck...')
import sys
from pathlib import Path

print('Using conda env: ' + str(Path(sys.executable).as_posix().split('/')[-3]) + ' [' + str(Path(sys.executable).as_posix()) + ']')

import keras


BACKBONE = 'inceptionresnetv2'
BACKBONE = 'seresnet152'
BACKBONE = 'resnet34'

class_ids = [0, 1]
dim_image = (256, 256, 3)
batch_size=18
number_elements = None
dataset_dir='/gpfs/groups/gc056/APPRANTI/cj3272/dataset/22FEV2019/GEN_segmentation/'

if BACKBONE == 'seresnet152':
    dim_image = (384, 384, 3)


categories = ['person']#, 'tie']
data_dir_source_coco = '/gpfs/home/cj3272/56/APPRANTI/cj3272/dataset/coco/'
dim_image = (256, 256, 3)
batch_size = 64
from luccauchon.data.Generators import COCODataFrameDataGenerator
train_generator = COCODataFrameDataGenerator(data_dir_source_coco=data_dir_source_coco, categories=categories, batch_size=batch_size, dim_image=dim_image, data_type_source_coco='train2017')
val_generator = COCODataFrameDataGenerator(data_dir_source_coco=data_dir_source_coco, categories=categories, batch_size=batch_size, dim_image=dim_image, data_type_source_coco='val2017')

df_train, df_val = generators.amateur_train_val_split(dataset_dir=dataset_dir, class_ids=class_ids, number_elements=number_elements)
train_generator = AmateurDataFrameDataGenerator(df_train, classes_id=class_ids, batch_size=batch_size, dim_image=dim_image)
val_generator = AmateurDataFrameDataGenerator(df_val, classes_id=class_ids, batch_size=batch_size, dim_image=dim_image)

# preprocess input
# from segmentation_models.backbones import get_preprocessing
# preprocess_input = get_preprocessing(BACKBONE)
# x_train = preprocess_input(x_train)
# x_val = preprocess_input(x_val)

# define model

model = Unet(BACKBONE, classes=len(class_ids), encoder_weights='imagenet')
#model = FPN(BACKBONE, classes=len(class_ids), encoder_weights='imagenet')
#model = PSPNet(BACKBONE, classes=len(class_ids), encoder_weights='imagenet')
#model = Linknet(BACKBONE, classes=len(class_ids), encoder_weights='imagenet')

model.compile('Adam', loss=cce_jaccard_loss, metrics=[jaccard_score])
model.summary()

modelCheckpoint = keras.callbacks.ModelCheckpoint(filepath='segmod_weights.{epoch:02d}-{val_loss:.4f}.hdf5',
                                                  monitor='val_loss',
                                                  verbose=0, save_best_only=False, save_weights_only=False,
                                                  mode='auto', period=1)
reduceLROnPlateau = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, verbose=1,
                                                      mode='auto', min_delta=0.001, cooldown=0, min_lr=10e-7)


model.fit_generator(generator=train_generator, steps_per_epoch=None, epochs=10, verbose=2,
                    callbacks=[reduceLROnPlateau, modelCheckpoint],
                    validation_data=val_generator, validation_steps=None, class_weight=None, max_queue_size=10,
                    workers=8, use_multiprocessing=False, shuffle=True, initial_epoch=0)


os._exit(0)
from segmentation_models import Unet
from segmentation_models.utils import set_trainable

''' 
Some times, it is useful to train only randomly initialized decoder in order not to damage weights of properly trained encoder with huge gradients during first steps of training. 
In this case, all you need is just pass freeze_encoder = True argument while initializing the model.
'''

model = Unet(backbone_name='resnet34', classes=len(class_ids), encoder_weights='imagenet', freeze_encoder=True)
model.compile('Adam', loss=cce_jaccard_loss, metrics=[jaccard_score])

# pretrain model decoder
model.fit_generator(generator=train_generator, steps_per_epoch=None, epochs=2, verbose=1, callbacks=None,
                    validation_data=val_generator, validation_steps=None, class_weight=None, max_queue_size=10,
                    workers=2, use_multiprocessing=False, shuffle=True, initial_epoch=0)

# release all layers for training
set_trainable(model) # set all layers trainable and recompile model

# continue training
model.fit_generator(generator=train_generator, steps_per_epoch=None, epochs=100, verbose=1, callbacks=None,
                    validation_data=val_generator, validation_steps=None, class_weight=None, max_queue_size=10,
                    workers=2, use_multiprocessing=False, shuffle=True, initial_epoch=0)