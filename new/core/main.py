# from skimage import io,transform
# import glob
# import os
import numpy as np
import pandas as pd
# from PIL import Image

# from sklearn.preprocessing import MultiLabelBinarizer
# from sklearn.model_selection import train_test_split

# import keras
# from keras.utils import np_utils
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from sklearn.utils import shuffle

#createFilePD
def createFilePD(Path, dataSplitRatio=0.8):
    Data1_columnNames = ["Id","Target"]
    Data1 = pd.read_csv(Path, names = Data1_columnNames, dtype={'Id': str,'Target':int})
    
    #隨機順序
    Data1 = shuffle(Data1)

    #切割資料
    num_example = Data1.shape[0]
    s = int(num_example * dataSplitRatio)
    train = Data1[:s].reset_index(drop=True)
    val = Data1[s:].reset_index(drop=True)
    
    print('--------------------')
    print('train shape : ', train.shape)
    print('val shape : ', val.shape)
    print('--------------------')

    print('--------------------')
    #print labels number
    unique, counts = np.unique(train['Target'].values.reshape(-1, 1), return_counts=True)
    print('y_train：', dict(zip(unique, counts)))

    unique, counts = np.unique(val['Target'].values.reshape(-1, 1), return_counts=True)
    print('y_val：', dict(zip(unique, counts)))
    print('--------------------')
    
    train['Target'] = train['Target'].astype(int)
    val['Target'] = val['Target'].astype(int)
    
    return train, val

def saveTrainModels_gen(model, saveModelPath, saveTensorBoardPath, epochs, batch_size,
                    train, val, DataDirPath, img_height, img_width, columns):
    # DataGen
    train_datagen = ImageDataGenerator(
            rescale=1./255, rotation_range=20, zoom_range=0.15,
    width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
    horizontal_flip=True, fill_mode="nearest")

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train,
        directory=DataDirPath,
        x_col="Id",# Filenames
        y_col=["Target"],#columns
        shuffle=True,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='other')
        
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=val,
        directory=DataDirPath,
        x_col="Id",# Filenames
        y_col=["Target"],#columns
        shuffle=True,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='other')

    #設置TensorBoard
    tbCallBack = TensorBoard(log_dir = saveTensorBoardPath, batch_size = batch_size,
                            write_graph = True, write_grads = True, write_images = True,
                            embeddings_freq = 0, embeddings_layer_names = None, embeddings_metadata = None)

    #設置checkpoint
    checkpoint = ModelCheckpoint(
                            monitor = 'val_loss', verbose = 1, 
                            save_best_only = True, mode = 'min',
                            filepath = ('%s_{epoch:02d}_{loss:.2f}_{accuracy:.2f}_{val_loss:.2f}_{val_accuracy:.2f}.h5' %(saveModelPath)))

    #設置ReduceLROnPlateau
    Reduce = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.9, patience = 5, cooldown = 1, verbose = 1, mode = 'min')

    #設置EarlyStopping
    Early = EarlyStopping(monitor = 'val_loss', patience = 15, verbose = 1, mode = 'min')

    callbacks_list = [checkpoint, tbCallBack, Reduce, Early]

    #訓練模型
    model.fit_generator(train_generator,
                steps_per_epoch = len(train)//batch_size,# len(train)//batch_size
                epochs = epochs,
                verbose = 1,
                shuffle = True,
                validation_data = val_generator,
                validation_steps = len(val)//batch_size, 
                callbacks = callbacks_list)
    