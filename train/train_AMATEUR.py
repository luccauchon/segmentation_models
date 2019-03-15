import os
os.environ['basedir_a'] = 'C:/temp2/'


from luccauchon.data.Generators import AmateurDataFrameDataGenerator
import luccauchon.data.Generators as generators

from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score


df_train, df_val = generators.amateur_train_val_split('G:/AMATEUR/segmentation/22FEV2019/GEN_segmentation/', number_elements=16)

dim_image = (256, 256, 3)
BACKBONE = 'resnet34'
BACKBONE = 'inceptionresnetv2'
#preprocess_input = get_preprocessing(BACKBONE)
# preprocess input
#x_train = preprocess_input(x_train)
#x_val = preprocess_input(x_val)

# define model
model = Unet(BACKBONE, encoder_weights='imagenet')
model.compile('Adam', loss=bce_jaccard_loss, metrics=[iou_score])
model.summary()
train_generator = AmateurDataFrameDataGenerator(df_train, batch_size=4, dim_image=dim_image)
val_generator = AmateurDataFrameDataGenerator(df_val, batch_size=4, dim_image=dim_image)

model.fit_generator(generator=train_generator, steps_per_epoch=None, epochs=10, verbose=1, callbacks=None,
                    validation_data=val_generator, validation_steps=None, class_weight=None, max_queue_size=10,
                    workers=2, use_multiprocessing=False, shuffle=True, initial_epoch=0)
