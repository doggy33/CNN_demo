from skimage import io, transform
import glob
import os
import numpy as np

from tensorflow.keras.utils import to_categorical
from core.main import readImage, saveTrainModels_gen

from core.Model.KerasApplication import buildKerasAppModel

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    # 參數設定
    img_height, img_width, img_channel = 224, 224, 3 #224, 224, 3
    batch_size = 2
    epochs = 50
    dataSplitRatio = 0.8
    readDataPath = "./Data/Train/"
    saveModelPath = "./Model/Keras"
    saveTensorBoardPath = "./Model/Tensorboard/"
    writeLabelPath = "./Model/Keras_Classes.txt"
    num_GPU = 1

    dirList = ['./Model', './Model/Tensorboard']
    for Path in dirList:
        if not os.path.exists(Path):
            os.makedirs(Path)

    # 載入資料
    data, label, Classes = readImage(readDataPath, img_height, img_width, img_channel, writeLabelPath)
    
    # 順序隨機
    num_example = data.shape[0]
    arr = np.arange(num_example)
    np.random.shuffle(arr)
    data = data[arr]
    label = label[arr]
    
    # 切割資料
    s = np.int(num_example * dataSplitRatio)
    x_train = data[:s]
    y_train = label[:s]
    x_val = data[s:]
    y_val = label[s:]
    
    # 重新調整大小
    if(img_channel == 1):
        x_train = x_train.reshape(x_train.shape[0], img_height, img_width, img_channel)
        x_val = x_val.reshape(x_val.shape[0], img_height, img_width, img_channel)

    print('x_train shape:', x_train.shape)
    print('x_val shape:', x_val.shape)
    print(x_train.shape[0], 'train samples')
    print(x_val.shape[0], 'validation samples')

    # 將數字轉為 One-hot 向量
    y_train = to_categorical(y_train, len(Classes))
    y_val = to_categorical(y_val, len(Classes))
    
    # 建構模型
    model = buildKerasAppModel(img_height, img_width, img_channel, len(Classes), num_GPU)

    # 訓練及保存模型
    saveTrainModels_gen(model, saveModelPath, saveTensorBoardPath, epochs, batch_size, x_train, y_train, x_val, y_val)