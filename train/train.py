###############################################################################
# Initialisation.
###############################################################################
import os
import sys
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-a", "--architecture", dest="architecture", help="specify the architecture", type=str, choices=['FPN', 'Linknet', 'PSP', 'Unet'], required=True)
parser.add_argument("-b", "--backbone", dest="backbone", help="specify the backbone", type=str, required=True,
                    choices=['vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'seresnet18', 'seresnet34', 'seresnet50', 'seresnet101', 'seresnet152',
                             'resnext50', 'resnext101', 'seresnext50', 'seresnext101', 'senet154', 'densenet121', 'densenet169', 'densenet201', 'inceptionv3', 'inceptionresnetv2',
                             'mobilenet', 'mobilenetv2'])
parser.add_argument("-blbs", "--baseline_batch_size", dest="baseline_batch_size", help="batch_size of precomiled dataset", type=int, required=False, default=6)
parser.add_argument("-bs", "--batch_size", dest="batch_size", help="", type=int, required=False, default=4)
parser.add_argument("-c", "--cat_names", dest="cat_names", help="specify coco categories to use", type=str, required=False, default=None)
parser.add_argument("-ca", "--classes_id_amateur", dest="classes_id_amateur", help="specify amateur categories to use", type=str, required=False, default='1', choices=['1', '2'])
parser.add_argument("-d", "--depth", dest="depth", help="", type=int, required=True)
parser.add_argument("-ds", "--dataset", dest="dataset", help="specify which datatset to use", type=str, required=True, choices=['coco2017', 'amateur'])
parser.add_argument("-e", "--epoch", dest="epoch", help="number of epochs", type=int, required=False, default=5)
parser.add_argument("-ex", "--experience_id", dest="experience_id", help="name of the directory where the results are saved", type=str, required=True)
parser.add_argument("-g", "--gpu_id", dest="gpu_id", help="specify the gpu to use", type=int, choices=range(4), required=True)
parser.add_argument("-he", "--height", dest="height", help="", type=int, required=True)
parser.add_argument("-p", "--precompiled", dest="precompiled", help="whether or not use a precompiled dataset", type=int, choices=range(1), required=False, default=0)
parser.add_argument("-t", "--temp_dir", dest="temp_dir", help="specify temporary directory to use", type=str, required=True)
parser.add_argument("-tr", "--threads", dest="threads", help="specify number of threads to use by keras to process input data", type=int, required=False, default=0,
                    choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
parser.add_argument("-w", "--width", dest="width", help="", type=int, required=True)

args = parser.parse_args()

os.environ['basedir_a'] = args.temp_dir
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
import luccauchon.data.__MYENV__ as E
import logging

E.APPLICATION_LOG_LEVEL = logging.INFO

LOG = E.setup_logger(logger_name=__name__, _level=E.APPLICATION_LOG_LEVEL)
import sys

sys.path.append('../')  # To find local version of the library
import keras
import PIL
import numpy as np
import scipy as scipy
from pathlib import Path
from luccauchon.data.Generators import COCODataFrameDataGenerator
from luccauchon.data.Generators import COCODataFramePreCompiledDataGenerator
from segmentation_models import PSPNet
from segmentation_models import FPN
from segmentation_models import Linknet
from segmentation_models import Unet
from segmentation_models.losses import cce_jaccard_loss
from segmentation_models.losses import jaccard_distance_l
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
LOG.info('keras.__version__=' + str(keras.__version__))
LOG.info('tf.__version__=' + str(tf.__version__))
LOG.info('PIL.__version__=' + str(PIL.__version__))
LOG.info('np.__version__=' + str(np.__version__))
LOG.info('scipy.__version__=' + str(scipy.__version__))
LOG.info('Using GPU ' + str(os.environ["CUDA_VISIBLE_DEVICES"]) + '  Good luck... you\'ll need it...')
LOG.info('Using conda env: ' + str(Path(sys.executable).as_posix().split('/')[-3]) + ' [' + str(Path(sys.executable).as_posix()) + ']')

###############################################################################
# Configuration de l'entrainement.
###############################################################################
gpu_id = args.gpu_id
backbone = args.backbone
dim_image = (args.width, args.height, args.depth)
batch_size = args.batch_size
architecture = args.architecture
baseline_batch_size = args.baseline_batch_size
precompiled = args.precompiled
nb_threads = args.threads
cat_names = None
if args.cat_names is not None:
    cat_names = [item for item in args.cat_names.split(',')]
classes_id_amateur = None
if args.classes_id_amateur is not None:
    classes_id_amateur = [int(item) for item in args.classes_id_amateur.split(',')]
model_dir = './logs/' + args.experience_id + '/'
if not os.path.isdir(model_dir):
    os.mkdir(model_dir)
else:
    LOG.warn('Directory already exists: ' + model_dir)
nb_epoch = args.epoch
dataset = args.dataset

model_checkpoint_prefix = architecture + '.' + backbone + '.w' + str(dim_image[0]) + '.h' + str(dim_image[1]) + '.d' + str(dim_image[2]) + '.bs' + str(batch_size) + '.' + str(
    dataset)

###############################################################################
# Configuration de la source des donnees.
###############################################################################
data_dir_source_coco = '/gpfs/home/cj3272/56/APPRANTI/cj3272/dataset/coco/'
data_dir_source_amateur_nt = 'F:/AMATEUR/segmentation/22FEV2019/GEN_segmentation/'
data_dir_source_amateur = '/gpfs/groups/gc014a/AMATEUR/dataset/segmentation/22FEV2019/GEN_segmentation/'

if 1 == precompiled and dataset == 'coco2017':
    train_generator = COCODataFramePreCompiledDataGenerator(data_dir_source_coco=data_dir_source_coco, batch_size=batch_size, dim_image=dim_image,
                                                            data_type_source_coco='train2017',
                                                            baseline_batch_size=baseline_batch_size)
    val_generator = COCODataFramePreCompiledDataGenerator(data_dir_source_coco=data_dir_source_coco, batch_size=batch_size, dim_image=dim_image, data_type_source_coco='val2017',
                                                          baseline_batch_size=baseline_batch_size)
elif 0 == precompiled and dataset == 'coco2017':
    train_generator = COCODataFrameDataGenerator(data_dir_source_coco=data_dir_source_coco, batch_size=batch_size, dim_image=dim_image, data_type_source_coco='train2017',
                                                 cat_names=cat_names)
    val_generator = COCODataFrameDataGenerator(data_dir_source_coco=data_dir_source_coco, batch_size=batch_size, dim_image=dim_image, data_type_source_coco='val2017',
                                               cat_names=cat_names)
elif dataset == 'amateur':
    import luccauchon.data.Generators as generators

    if os.name == 'nt':
        df_train, df_val = generators.amateur_train_val_split(dataset_dir=data_dir_source_amateur_nt, class_ids=classes_id_amateur, number_elements=None)
    else:
        df_train, df_val = generators.amateur_train_val_split(dataset_dir=data_dir_source_amateur, class_ids=classes_id_amateur, number_elements=None)
    from luccauchon.data.Generators import AmateurDataFrameDataGenerator

    train_generator = AmateurDataFrameDataGenerator(df_train, classes_id=classes_id_amateur, batch_size=batch_size, dim_image=dim_image)
    train_generator.cat_ids = classes_id_amateur
    val_generator = AmateurDataFrameDataGenerator(df_val, classes_id=classes_id_amateur, batch_size=batch_size, dim_image=dim_image)
    val_generator.cat_ids = classes_id_amateur

else:
    assert False
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
reduceLROnPlateau = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1,
                                                      mode='auto', min_delta=0.01, cooldown=0, min_lr=10e-7)

''' 
Some times, it is useful to train only randomly initialized decoder in order not to damage weights of properly trained encoder with huge gradients during first steps of training. 
In this case, all you need is just pass freeze_encoder = True argument while initializing the model.
'''
freeze_encoder = True
if os.name == 'nt':
    freeze_encoder = False

if number_of_classes > 1:
    if architecture == 'PSP':
        model = PSPNet(backbone, input_shape=dim_image, classes=number_of_classes, encoder_weights='imagenet', activation='softmax', freeze_encoder=freeze_encoder)
    elif architecture == 'FPN':
        model = FPN(backbone, input_shape=dim_image, classes=number_of_classes, encoder_weights='imagenet', activation='softmax', freeze_encoder=freeze_encoder)
    else:
        assert False
    model.compile('Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.summary()
else:
    assert 1 == number_of_classes
    if architecture == 'PSP':
        model = PSPNet(backbone, input_shape=dim_image, classes=1, encoder_weights='imagenet', activation='sigmoid', freeze_encoder=freeze_encoder)
    elif architecture == 'FPN':
        model = FPN(backbone, input_shape=dim_image, classes=1, encoder_weights='imagenet', activation='sigmoid', freeze_encoder=freeze_encoder)
    elif architecture == 'Linknet':
        model = Linknet(backbone, input_shape=dim_image, classes=1, encoder_weights='imagenet', activation='sigmoid', freeze_encoder=freeze_encoder)
    elif architecture == 'Unet':
        model = Unet(backbone, input_shape=dim_image, classes=1, encoder_weights='imagenet', activation='sigmoid', freeze_encoder=freeze_encoder)
    else:
        assert False
    # model.compile('Adam', loss='binary_crossentropy', metrics=['binary_accuracy'])
    model.compile('Adam', loss='jaccard_distance_l', metrics=['iou_score'])
    model.summary()

if os.name == 'nt':
    LOG.info('GPU=(' + str(gpu_id) + ')  Architecture=' + architecture + '  Backbone=' + backbone + '  dim_image=' + str(dim_image) + '  batch_size/baseline_batch_size=(' + str(
        batch_size) + '/' + str(baseline_batch_size) + ')  model_checkpoint_prefix=(' + str(model_checkpoint_prefix) + ')  use coco2017 precompiled dataset=' + str(
        precompiled) + '  #_threads=' + str(nb_threads) + '  models_directory=' + model_dir + '  dataset=' + dataset)

    model.fit_generator(generator=train_generator, steps_per_epoch=None, epochs=nb_epoch, verbose=1, callbacks=None,
                        validation_data=val_generator, validation_steps=None, class_weight=None, max_queue_size=10,
                        workers=0, use_multiprocessing=False, shuffle=True, initial_epoch=0)
else:
    LOG.info('GPU=(' + str(gpu_id) + ')  Architecture=' + architecture + '  Backbone=' + backbone + '  dim_image=' + str(dim_image) + '  batch_size/baseline_batch_size=(' + str(
        batch_size) + '/' + str(baseline_batch_size) + ')  model_checkpoint_prefix=(' + str(model_checkpoint_prefix) + ')  use coco2017 precompiled dataset=' + str(
        precompiled) + '  #_threads=' + str(nb_threads) + '  models_directory=' + model_dir + '  dataset=' + dataset)

    # pretrain model decoder
    model.fit_generator(generator=train_generator, steps_per_epoch=None, epochs=2, verbose=1, callbacks=None,
                        validation_data=val_generator, validation_steps=None, class_weight=None, max_queue_size=10,
                        workers=nb_threads, use_multiprocessing=True, shuffle=True, initial_epoch=0)

    # release all layers for training
    set_trainable(model)  # set all layers trainable and recompile model
    model.summary()
    LOG.info('GPU=(' + str(gpu_id) + ')  Architecture=' + architecture + '  Backbone=' + backbone + '  dim_image=' + str(dim_image) + '  batch_size/baseline_batch_size=(' + str(
        batch_size) + '/' + str(baseline_batch_size) + ')  model_checkpoint_prefix=(' + str(model_checkpoint_prefix) + ')  use coco2017 precompiled dataset=' + str(
        precompiled) + '  #_threads=' + str(nb_threads) + '  models_directory=' + model_dir + '  dataset=' + dataset)

    # continue training
    model.fit_generator(generator=train_generator, steps_per_epoch=None, epochs=nb_epoch, verbose=1, callbacks=[modelCheckpoint, reduceLROnPlateau],
                        validation_data=val_generator, validation_steps=None, class_weight=None, max_queue_size=10,
                        workers=nb_threads, use_multiprocessing=True, shuffle=True, initial_epoch=0)
