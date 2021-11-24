from dataset_core import create_dataset
import matplotlib.pyplot as plt
import numpy as np

srcPaths = ['dataset_new/img_1', 'dataset_new/img_2', 'dataset_new/img_3']
dataSetFilename = 'cv_dataset.npz'

if create_dataset(dataSetFilename, srcPaths):

    data = np.load(dataSetFilename, allow_pickle = True)

    imgList = data['images']
    labelList = data['labels']
    print(imgList.shape)
    print(labelList)

a, b = 3, 3
count = 1

# show first 5 picture 
for i in range (0, 9):
    img = imgList[i]
    label = labelList[i]
    imgRGB = img[:,:,::-1]
    plt.subplot(a,b, count)
    plt.imshow(imgRGB)
    plt.title(label)
    count += 1

plt.tight_layout()
plt.show()

