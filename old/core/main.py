from skimage import io, transform
import glob
import os
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping

#讀取圖片
def readImage(path, img_height, img_width, img_channel, writeClassNamePath = None):
    Classes = []
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    imgs = []
    labels = []
    for idx, folder in enumerate(cate):
        for im in glob.glob(folder+'/*.jpg'):
            print('reading the images:%s'%(im))
            img = io.imread(im)
            img_resize = transform.resize(img, (img_height, img_width))
            # --------------------------------------------------------------------
            if(img_channel > 1):
                if(img_resize.shape != (img_height, img_width, img_channel)):
                    print('img_resize.shape error:%s'%(im))
                else:
                    # img_resize = img_resize / 255
                    imgs.append(img_resize)
                    labels.append(idx)
            else:
                if(img_resize.shape != (img_height, img_width)):
                    print('img_resize.shape error:%s'%(im))
                else:
                    # img_resize = img_resize / 255
                    imgs.append(img_resize)
                    labels.append(idx)
            # --------------------------------------------------------------------
    
    Classes = writeLabels(path, writeClassNamePath)
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32), Classes

#寫檔(classes)
def writeLabels(readPath, writePath):
    Classes = [x for x in os.listdir(readPath) if os.path.isdir(readPath + x)]
    fw = open(writePath, "w")
    for i in range(0, len(Classes), 1):
        fw.write(str(Classes[i]) + "\n")
    fw.close()
    return Classes

def saveTrainModels_gen(model, saveModelPath, saveTensorBoardPath, epochs, batch_size,
                    x_train, y_train, x_val, y_val, 
                    rotation_range = 60, width_shift_range = 0.3, height_shift_range = 0.3, 
                    shear_range = 0.3, zoom_range = 0.3, horizontal_flip = True, fill_mode = 'nearest'):
	# DataGen
    train_datagen = ImageDataGenerator(rescale = 1/255,
                                rotation_range = rotation_range,
                                width_shift_range = width_shift_range,
                                height_shift_range = height_shift_range,
                                shear_range = shear_range,
                                zoom_range = zoom_range,
                                horizontal_flip = horizontal_flip,
                                fill_mode = fill_mode)
    # DataGen
    val_datagen = ImageDataGenerator(rescale = 1/255)

    #設置TensorBoard
    tbCallBack = TensorBoard(log_dir = saveTensorBoardPath, batch_size = batch_size,
                            write_graph = True, write_grads = True, write_images = True,
                            embeddings_freq = 0, embeddings_layer_names = None, embeddings_metadata = None)

    #設置checkpoint
    checkpoint = ModelCheckpoint(
                            monitor = 'val_loss', verbose = 1, 
                            save_best_only = True, mode = 'min',
                            filepath = ('%s_{epoch:02d}_{loss:.4f}_{val_loss:.4f}_{accuracy:.2f}_{val_accuracy:.2f}.h5' %(saveModelPath)))

    #設置ReduceLROnPlateau
    Reduce = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.9, patience = 9, cooldown = 1, verbose = 1)

    #設置EarlyStopping
    Early = EarlyStopping(monitor = 'val_loss', patience = 27, verbose = 1)

    callbacks_list = [checkpoint, tbCallBack, Reduce, Early]

    #訓練模型
    model.fit_generator(train_datagen.flow(x_train, y_train, batch_size = batch_size),
                steps_per_epoch = len(x_train)//batch_size,
                epochs = epochs,
                verbose = 1,
                shuffle = True,
                validation_data = val_datagen.flow(x_val, y_val),
                callbacks = callbacks_list)
    