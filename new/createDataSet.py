import glob
import os

if __name__ == "__main__":
	# 參數設定
	writePath = './Data/train.csv'
	rootPath = './Data/Train/'
	fw = open(writePath, "w")
	rootDirList = [rootPath + x for x in os.listdir(rootPath) if os.path.isdir(rootPath + x)]
	for idx, folder in enumerate(rootDirList):
		for imgPath in glob.glob(folder+'/*.jpg'):
			folderName = folder.split('/')
			ImgFilaName = os.path.basename(imgPath)
			fw.write(folderName[-1] + "/" + ImgFilaName + "," + folderName[-1] + "\n")
	fw.close()
