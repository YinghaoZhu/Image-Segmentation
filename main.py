from data import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from Kerastest import *
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')


#image_train, mask_train = geneTrainNpy("/data/cephfs/punim0619/dataset/trainingset/img_crop", "/data/cephfs/punim0619/dataset/trainingset/seg_crop")
#test_input = geneTestNpy('/data/cephfs/punim0619/dataset/testset/test_img')

myGene = trainGenerator(8,'/data/cephfs/punim0619/yinghao/trainingset/','image','groundtruth',data_gen_args, save_to_dir = None)

model = unet()
#model_checkpoint = ModelCheckpoint('yinghao.hdf5', monitor='loss', verbose=1, save_best_only=True)
model.fit_generator(myGene,steps_per_epoch=300,epochs=10) #,callbacks=[model_checkpoint])

testGene = testGenerator("/Users/zhuyinghao/Desktop/dataset/yinghao/testset/image")
results = model.predict_generator(testGene, 8, verbose=1)

saveResult('~/ComputingProject/output_images', results)

