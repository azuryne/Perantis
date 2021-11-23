import numpy as np 
import PIL.Image
import os 
import matplotlib.pyplot as plt
import cv2 as cv

src = "image"

for fname in sorted(os.listdir(src)):
    filePath = os.path.join(src, fname)
    label = np.asarray(PIL.Image.open(filePath))
    img = cv.imread(filePath)
    imgRGB = img[:,:,::-1]
    plt.imshow(imgRGB)
    plt.title(f"{fname}")
    plt.show()
    print(label.dtype)
    print(label.shape)
    