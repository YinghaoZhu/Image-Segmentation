from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import glob
import skimage.io as io
import skimage.transform as trans
from skimage import img_as_ubyte

# in oder to use binary-entropy we need to adjust the input images
def adjust_data(img, mask):
    img = (255 -img)/255.0

    mask = mask / 255
    # Notice: your pixel val must be 0/255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return (img, mask)




def trainGenerator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode="grayscale",
                   mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                   flag_multi_class=False, num_class=2, save_to_dir=None, target_size=(256, 256), seed=1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path, 
        classes=[image_folder], 
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir, 
        save_prefix=image_save_prefix,
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)
    train_generator = zip(image_generator, mask_generator)
    for (img, mask) in train_generator:
       
        img, mask = adjust_data(img, mask) 
        yield (img, mask)
# base on test set
def val_generator(batch_size, train_path, image_folder, mask_folder, aug_dict, image_color_mode="grayscale",
                   mask_color_mode="grayscale", image_save_prefix="image", mask_save_prefix="mask",
                   flag_multi_class=False, num_class=2, save_to_dir=None, target_size=(256, 256), seed=1):
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path, 
        classes=[image_folder],
        class_mode=None, 
        color_mode=image_color_mode, 
        target_size=target_size, 
        batch_size=batch_size,
        save_to_dir=save_to_dir, 
        save_prefix=image_save_prefix,
        seed=seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes=[mask_folder],
        class_mode=None,
        color_mode=mask_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        save_to_dir=save_to_dir,
        save_prefix=mask_save_prefix,
        seed=seed)
    train_generator = zip(image_generator, mask_generator)  
    for (img, mask) in train_generator:
        
        img, mask = adjust_data(img, mask)  # [2,256,256]
        yield (img, mask)



flag_multi_class = False
def testGenerator(test_path, target_size=(256, 256), as_gray=True):
    for i in range(1, 11):
        img = io.imread(os.path.join(test_path, "%d.tif" % i), as_gray=as_gray)
        print(img.shape)
        img = trans.resize(img, target_size)
        img = img / 255
        img = np.reshape(img, (1,) + img.shape)
       
        print(img.shape)
        yield img



def saveResult(save_path, npyfile, flag_multi_class=False, num_class=2):

    for i,item in enumerate(npyfile):
        img = item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)
