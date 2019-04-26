import os

os.environ['basedir_a'] = 'F:/Temp2/'

import keras
import segmentation_models
import time
import skimage
import numpy as np
import scipy
import PIL

print('PIL:' + str(PIL.__version__))
model_base_dir = 'F:/AMATEUR/segmentation_models/'
images_dir = 'F:/AMATEUR/fichiers_pour_tests/'
results_dir = 'F:/AMATEUR/results/'

t1 = time.time()
class_ids = [1]
dim_img = (64, 64, 3)
dim_w = dim_img[0]
dim_h = dim_img[1]
model = keras.models.load_model(model_base_dir + 'Unet.seresnet18.w64.h64.d3.bs32.amateur_weights.03-0.0368.hdf5')
model.summary()
t2 = time.time()

from os import listdir
from os.path import isfile, join

onlyfiles = [f for f in listdir(images_dir) if isfile(join(images_dir, f))]

for image_file in onlyfiles:
    t1 = time.time()
    the_image = skimage.io.imread(join(images_dir, image_file))
    the_image_height = the_image.shape[0]
    the_image_width = the_image.shape[1]

    input_image = np.array(PIL.Image.fromarray(the_image).resize((dim_w, dim_h), resample=PIL.Image.NEAREST))
    input_image = input_image[np.newaxis, ...]
    res = model.predict(input_image)

    for index, mask_n in enumerate(class_ids):
        my_mask = res[..., index]
        my_mask = np.squeeze(my_mask, axis=0)
        my_mask_pil = PIL.Image.fromarray(my_mask).resize((the_image_width, the_image_height),
                                                          resample=PIL.Image.NEAREST)
        my_mask = np.array(my_mask_pil)
        assert my_mask.shape[0] == the_image.shape[0]
        assert my_mask.shape[1] == the_image.shape[1]
        skimage.io.imsave(results_dir + image_file + '.mask.classe_id_' + str(mask_n) + '.png',
                          (255 * my_mask).astype(np.uint8))

        my_mask = my_mask[..., np.newaxis]
        my_mask = (255 * my_mask).astype(np.uint8)
        selected_pixels = (my_mask > 64).all(axis=2)
        the_image_merged = skimage.io.imread(join(images_dir, image_file))
        assert selected_pixels.shape[0] == the_image_merged.shape[0]
        assert selected_pixels.shape[1] == the_image_merged.shape[1]
        the_image_merged[selected_pixels] = 0
        skimage.io.imsave(results_dir + image_file + '.merged.classe_id_' + str(mask_n) + '.png', the_image_merged)
