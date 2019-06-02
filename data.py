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
        train_path,  # 训练数据文件夹路径
        classes=[image_folder],  # 类别文件夹,对哪一个类进行增强
        class_mode=None,  # 不返回标签
        color_mode=image_color_mode,  # 灰度，单通道模式
        target_size=target_size,  # 转换后的目标图片大小
        batch_size=batch_size,  # 每次产生的（进行转换的）图片张数
        save_to_dir=save_to_dir,  # 保存的图片路径
        save_prefix=image_save_prefix,  # 生成图片的前缀，仅当提供save_to_dir时有效
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
    train_generator = zip(image_generator, mask_generator)  # 组合成一个生成器
    for (img, mask) in train_generator:
        # 由于batch是2，所以一次返回两张，即img是一个2张灰度图片的数组，[2,256,256]
        img, mask = adjust_data(img, mask)  # 返回的img依旧是[2,256,256]
        yield (img, mask)




flag_multi_class = False
def testGenerator(test_path, target_size=(256, 256), as_gray=True):
    for i in range(1, 11):
        img = io.imread(os.path.join(test_path, "%d.tif" % i), as_gray=as_gray)
        print(img.shape)
        img = trans.resize(img, target_size)
        img = img / 255
        img = np.reshape(img, (1,) + img.shape)
        # 将测试图片扩展一个维度，与训练时的输入[2,256,256]保持一致
        print(img.shape)
        yield img




# # 上面函数是给出测试后的输出之后，为输出涂上不同的颜色，多类情况下才起作用，两类的话无用
#
def saveResult(save_path, npyfile, flag_multi_class=False, num_class=2):

    for i,item in enumerate(npyfile):
        img = item[:,:,0]
        io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)
