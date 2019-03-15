import os
os.environ['basedir_a'] = 'C:/temp2/'

import train

from train.data.Generators import AmateurDataFrameDataGenerator
import train.data.Generators as generators

from segmentation_models import Unet
from segmentation_models.backbones import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score


df_train, df_val = generators.amateur_train_val_split('G:/AMATEUR/segmentation/22FEV2019/GEN_segmentation/', number_elements=32)


BACKBONE = 'resnet34'
preprocess_input = get_preprocessing(BACKBONE)