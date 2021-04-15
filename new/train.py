import os
from core.main import createFilePD, saveTrainModels_gen
from core.Model.KerasApplication import buildKerasAppModel

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == "__main__":
    #參數設定
    img_height, img_width, img_channel = 224, 224, 3 #224, 224, 3
    classes = ['Target']
    num_classes = 2 #binary
    batch_size = 4
    epochs = 50
    dataSplitRatio = 0.8
    readDataPath = "./Data/train.csv"
    DataDirPath = "./Data/Train/"
    savePath = "./Model/"
    saveModelPath = "./Model/Model"
    saveTensorBoardPath = "./Model/Tensorboard/"

    if not os.path.exists(savePath):
         os.mkdir(savePath)
    if not os.path.exists(saveTensorBoardPath):
         os.mkdir(saveTensorBoardPath)

    #Revicing the bug of TensorBoard of TF2
    tfPath01 = saveTensorBoardPath + '/train'
    tfPath02 = saveTensorBoardPath + '/train/plugins'
    tfPath03 = saveTensorBoardPath + '/train/plugins/profile'
    if not os.path.exists(tfPath01):
        os.mkdir(tfPath01)
    if not os.path.exists(tfPath02):
        os.mkdir(tfPath02)
    if not os.path.exists(tfPath03):
        os.mkdir(tfPath03)

    #載入資料
    train, val = createFilePD(readDataPath, dataSplitRatio=dataSplitRatio)
    
    print(train.head(5))
    
    # 建構模型
    model = buildKerasAppModel(img_height, img_width, img_channel, num_classes)
    
    #訓練及保存模型
    saveTrainModels_gen(model, saveModelPath, saveTensorBoardPath, epochs, batch_size, 
                    train, val, DataDirPath, img_height, img_width, classes)