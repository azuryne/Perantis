import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

filename = 'digits.png'
imgGray = cv.imread(filename, cv.IMREAD_GRAYSCALE)

print(imgGray.shape)

#### get all the digits
IMG_SIZE = 20

# Resize
rowNum = imgGray.shape[0] / IMG_SIZE
colNum = imgGray.shape[1] / IMG_SIZE

rows = np.vsplit(imgGray, rowNum)  #split each row first

digits = []
for row in rows:
    rowCells = np.hsplit(row, colNum)  #after splitting row, split each col
    for digit in rowCells:
        digits.append(digit)   #each cell rep a particular digit

# convert list to np.array
digits = np.array(digits)
print('digits', digits.shape)

# labels
DIGITS_CLASS = 10
repeatNum = len(digits) / DIGITS_CLASS
labels = np.repeat(np.arange(DIGITS_CLASS), repeatNum)
print('labels', labels.shape)

#### get features
features = []
for digit in digits:
    img_pixel = np.float32(digit.flatten())  #flatten 20 by 20 pixel to 1D array of 400 pixel
    features.append(img_pixel)

features = np.squeeze(features)
print('features', features.shape)

# shuffle features and labels
# seed random for constant random value
rand = np.random.RandomState(321)
shuffle = rand.permutation(features.shape[0])
features, labels = features[shuffle], labels[shuffle]

# split into training and testing
splitRatio = [2, 1]
sumRatio = sum(splitRatio)
partition = np.array(splitRatio) * len(features) // sumRatio
partition = np.cumsum(partition)

featureTrain, featureTest = np.array_split(features, partition[:-1])
labelTrain, labelTest = np.array_split(labels, partition[:-1])

print('featureTrain', featureTrain.shape)
print('featureTest', featureTest.shape)
print('labelTrain', labelTrain.shape)
print('labelTest', labelTest.shape)

# Train the KNN model:
print('Training KNN model')
knn = cv.ml.KNearest_create()
knn.train(featureTrain, cv.ml.ROW_SAMPLE, labelTrain)

# Test the created model:
k = 4
ret, prediction, neighbours, dist = knn.findNearest(featureTest, k)

# Compute the accuracy:
accuracy = (np.squeeze(prediction) == labelTest).mean() * 100
print("Accuracy when k=4: {}".format(accuracy))
print()

# Getting input from user:
number_list = []
n = int(input("Enter the combination random numbers (1-4): "))

print("\n")

if n <= 4:
    if n != 0:
        for i in range(0, n):
            print("Enter number at index", i)
            item = int(input())
            number_list.append(item)
        print("User list is:", number_list, "\n")
else:
    print("Please re-enter")

#### Prediction:
import random

np.random.seed(2)

filename = 'digits.png'

imgGray = cv.imread(filename, cv.IMREAD_GRAYSCALE)

print("Image shape:", imgGray.shape)

IMG_SIZE = 20

# Resize
rowNum = imgGray.shape[0] / IMG_SIZE
colNum = imgGray.shape[1] / IMG_SIZE

rows = np.vsplit(imgGray, rowNum)

digits = []
for row in rows:
    rowCells = np.hsplit(row, colNum)
    for digit in rowCells:
        digits.append(digit)

    ##convert list to np.array
digits = np.array(digits)
print('digits', digits.shape)
print('len digit', len(digits))

own_features = []
for digit in digits:
    img_pixel = np.float32(digit.flatten())
    own_features.append(img_pixel)

own_features = np.squeeze(own_features)
print('features', own_features.shape)

new_features = random.sample(list(own_features), n)
new_features1 = np.asarray(new_features)
print("Random numbers features shape: ", new_features1.shape)

labels1 = np.array(number_list)
print('Labels shape: ', labels1.shape)
print(labels1)

ret, prediction, neighbours, dist = knn.findNearest(new_features1, k)

# Compute the accuracy:
accuracy = (np.squeeze(prediction) == labels1).mean() * 100

print('Accuracy of number combination: {:.2f}%'.format(accuracy))
print("Input from user: ", labels1)
print("Predicted label: ", prediction.flatten())
print()

