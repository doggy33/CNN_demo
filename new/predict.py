from skimage import io,transform
import glob
import numpy as np
import re

from tensorflow.keras.models import load_model

#讀取圖片
def readImage(readPath, img_height, img_width, readClassesPath = None):
    imgs = []
    labels = []
    fileName = []

    for im in glob.glob(readPath + '*.jpg'):
        print('reading the images:%s'%(im))
        img = io.imread(im)
        img_resize = transform.resize(img, (img_height, img_width))
        img_resize = img_resize
        imgs.append(img_resize)
        fileName.append(im.split('\\')[-1:])

    if(readClassesPath != None):
        # label
        f = open(readClassesPath, "r")
        readText = f.read()
        labels = re.split(',|\n',readText)
        f.close()

        return np.asarray(imgs,np.float32), labels, fileName
    else:
        return np.asarray(imgs,np.float32), fileName


if __name__ == "__main__":
    #參數設定
    img_height, img_width, img_channel = 224, 224 , 3
    readDataPath = "./Data/Test/"
    readClassPath = "./Data/Classes.txt"
    loadModelPath = "./Model/Model_50_0.02_1.00_0.20_1.00.h5"

    #載入資料
    data, label, fileName = readImage(readDataPath, img_height, img_width, readClassPath)

    #載入模型
    model = load_model(loadModelPath)

    #predict
    result = model.predict(data)
    for i in range(0, result.shape[0], 1):
        print('%s：%s'%(fileName[i][0], str(label[np.argmax(result[i])])))
    