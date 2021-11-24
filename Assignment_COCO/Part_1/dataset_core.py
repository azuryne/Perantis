import os
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def create_dataset(dataSetFilename, srcPaths):

    # list of image 
    imgList = []
    labelList = []

    # read image file from folder 
    for srcPath in srcPaths:
        for fname in sorted(os.listdir(srcPath)):
            filePath = os.path.join(srcPath, fname)
            # read
            img = cv.imread(filePath)

            # get last character 
            fname_no_ext = os.path.splitext(fname)[0]
            #label = fname_no_ext[-1]
            label = fname_no_ext

            #append
            imgList.append(img)
            labelList.append(label) 

    # save dataset
   # dataSetFilename = 'azu_dataset'

    images = np.array(imgList, dtype='object')
    labels = np.array(labelList, dtype='object')
    np.savez_compressed(dataSetFilename, images= images, labels = labels)

    return True
 



# if __name__ == "__main__":

#     create_dataset()

#     dataSetFilename = 'azuDataset.npz'

#     if create_dataset():

#         data = np.load(dataSetFilename, allow_pickle = True)

#         imgList = data['images']
#         labelList = data['labels']

#         for i in range (0, 5):
#             img = imgList[i]
#             label = labelList[i]

#             imgRGB = img[:,:,::-1]
#             plt.imshow(imgRGB)
#             plt.title(label)
#             plt.show()

    