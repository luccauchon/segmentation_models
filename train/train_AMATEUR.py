import os

os.environ['basedir_a'] = 'C:/temp2/'

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

BACKBONE = 'resnet34'
BACKBONE = 'inceptionresnetv2'
BACKBONE = 'seresnet152'

class_ids = [0, 1]
dim_image = (256, 256, 3)


if BACKBONE == 'seresnet152':
    dim_image = (384, 384, 3)

df_train, df_val = generators.amateur_train_val_split(dataset_dir='G:/AMATEUR/segmentation/22FEV2019/GEN_segmentation/', class_ids=class_ids, number_elements=16)

train_generator = AmateurDataFrameDataGenerator(df_train, classes_id=class_ids, batch_size=4, dim_image=dim_image)
val_generator = AmateurDataFrameDataGenerator(df_val, classes_id=class_ids, batch_size=4, dim_image=dim_image)




# preprocess input
# from segmentation_models.backbones import get_preprocessing
# preprocess_input = get_preprocessing(BACKBONE)
# x_train = preprocess_input(x_train)
# x_val = preprocess_input(x_val)

# define model

model = Unet(BACKBONE, classes=len(class_ids), encoder_weights='imagenet')
model = FPN(BACKBONE, classes=len(class_ids), encoder_weights='imagenet')
model = PSPNet(BACKBONE, classes=len(class_ids), encoder_weights='imagenet')
model = Linknet(BACKBONE, classes=len(class_ids), encoder_weights='imagenet')

model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])
model.summary()

modelCheckpoint = keras.callbacks.ModelCheckpoint(filepath='segmod_weights.{epoch:02d}-{val_loss:.4f}.hdf5',
                                                  monitor='val_loss',
                                                  verbose=0, save_best_only=False, save_weights_only=False,
                                                  mode='auto', period=1)
reduceLROnPlateau = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, verbose=1,
                                                      mode='auto', min_delta=0.001, cooldown=0, min_lr=10e-7)


model.fit_generator(generator=train_generator, steps_per_epoch=None, epochs=10, verbose=1,
                    callbacks=[reduceLROnPlateau, modelCheckpoint],
                    validation_data=val_generator, validation_steps=None, class_weight=None, max_queue_size=10,
                    workers=2, use_multiprocessing=False, shuffle=True, initial_epoch=0)


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