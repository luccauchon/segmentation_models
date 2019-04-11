###############################################################################
# Initialisation.
###############################################################################
import os
import sys

gpu_id = int(sys.argv[1])
assert gpu_id <= 3 and gpu_id >= 0
os.environ['basedir_a'] = '/gpfs/home/cj3272/tmp/'
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
import luccauchon.data.__MYENV__ as E
import logging

E.APPLICATION_LOG_LEVEL = logging.DEBUG
import sys

sys.path.append('../')  # To find local version of the library
import keras
import PIL
import numpy as np
import scipy as scipy
from pathlib import Path
from luccauchon.data.Generators import COCODataFrameDataGenerator
from segmentation_models import PSPNet
from segmentation_models.losses import cce_jaccard_loss
from segmentation_models.metrics import jaccard_score
from segmentation_models.utils import set_trainable

###############################################################################
# Set tf backend to allow memory to grow, instead of claiming everything
###############################################################################
import tensorflow as tf


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

###############################################################################
# Information des librairies.
###############################################################################
print('keras.__version__=' + str(keras.__version__))
print('tf.__version__=' + str(tf.__version__))
print('PIL.__version__=' + str(PIL.__version__))
print('np.__version__=' + str(np.__version__))
print('scipy.__version__=' + str(scipy.__version__))
print('Using GPU ' + str(os.environ["CUDA_VISIBLE_DEVICES"]) + '  Good luck... you\'ll need it...')
print('Using conda env: ' + str(Path(sys.executable).as_posix().split('/')[-3]) + ' [' + str(Path(sys.executable).as_posix()) + ']')

###############################################################################
# Configuration de l'entrainement.
###############################################################################
backbone = str(sys.argv[2])
dim_image = (int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
batch_size = int(sys.argv[6])
model_dir = './logs/'
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
model_checkpoint_prefix = str(sys.argv[7]) + '.' + backbone + '.'

###############################################################################
# Configuration de la source des donnees.
###############################################################################
data_dir_source_coco = '/gpfs/home/cj3272/56/APPRANTI/cj3272/dataset/coco/'
train_generator = COCODataFrameDataGenerator(data_dir_source_coco=data_dir_source_coco, batch_size=batch_size, dim_image=dim_image, data_type_source_coco='train2017')
val_generator = COCODataFrameDataGenerator(data_dir_source_coco=data_dir_source_coco, batch_size=batch_size, dim_image=dim_image, data_type_source_coco='val2017')
number_of_classes = len(train_generator.cat_ids)
assert len(train_generator.cat_ids) == len(val_generator.cat_ids)

###############################################################################
# Configuration du modele et entrainement.
###############################################################################
assert 0 != number_of_classes and 0 < number_of_classes

modelCheckpoint = keras.callbacks.ModelCheckpoint(filepath=model_dir + model_checkpoint_prefix + '_weights.{epoch:02d}-{val_loss:.4f}.hdf5',
                                                  monitor='val_loss',
                                                  verbose=0, save_best_only=False, save_weights_only=False,
                                                  mode='auto', period=1)
reduceLROnPlateau = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, verbose=1,
                                                      mode='auto', min_delta=0.001, cooldown=0, min_lr=10e-7)

''' 
Some times, it is useful to train only randomly initialized decoder in order not to damage weights of properly trained encoder with huge gradients during first steps of training. 
In this case, all you need is just pass freeze_encoder = True argument while initializing the model.
'''
model = PSPNet(backbone, classes=number_of_classes, encoder_weights='imagenet', activation='softmax', freeze_encoder=True)
model.compile('Adam', loss=cce_jaccard_loss, metrics=[jaccard_score])
model.summary()
print('Backbone=' + backbone + '  dim_image=(' + str(dim_image) + ')  batch_size=(' + str(batch_size) + ')  model_checkpoint_prefix=(' + str(model_checkpoint_prefix) + ')')

# pretrain model decoder
model.fit_generator(generator=train_generator, steps_per_epoch=None, epochs=2, verbose=2, callbacks=[modelCheckpoint, reduceLROnPlateau],
                    validation_data=val_generator, validation_steps=None, class_weight=None, max_queue_size=10,
                    workers=8, use_multiprocessing=False, shuffle=True, initial_epoch=0)

# release all layers for training
set_trainable(model)  # set all layers trainable and recompile model

# continue training
model.fit_generator(generator=train_generator, steps_per_epoch=None, epochs=10, verbose=2, callbacks=[modelCheckpoint, reduceLROnPlateau],
                    validation_data=val_generator, validation_steps=None, class_weight=None, max_queue_size=10,
                    workers=8, use_multiprocessing=True, shuffle=True, initial_epoch=0)
