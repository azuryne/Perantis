import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt 


def knnModel():
    filepath = '/Users/azureennaja/Desktop/Perantis/Project 1/Question2/digits.png'
    imgGray = cv.imread(filepath, cv.IMREAD_GRAYSCALE)

    print(imgGray.shape)

    SIZE = 20

    #Resize
    rowNum = imgGray.shape[0] / SIZE
    colNum = imgGray.shape[1] / SIZE

    rows = np.vsplit(imgGray, rowNum)

    digits = []
    for row in rows:
        rowCells = np.hsplit(row, colNum)
        for digit in rowCells:
            digits.append(digit)

    digits = np.array(digits)
    print('digits', digits.shape)

    #labels
    Digit_class = 10
    repeatNum = len(digits) / Digit_class
    labels = np.repeat(np.arange(Digit_class), repeatNum)
    print('labels', labels.shape)

    features = []
    for digit in digits: 
        img_pixel = np.float32(digit.flatten())
        features.append(img_pixel)
    
    features = np.squeeze(features)
    print("Features:", features.shape)

    #Shuffle features and labels 
    #Seed random for constant random value 

    rand = np.random.RandomState(321)
    shuffle = rand.permutation(features.shape[0])
    features, labels = features[shuffle], labels[shuffle]

    #Split into training and testing 

    splitRatio = [2,1]
    sumRatio = sum(splitRatio)
    partition = np.array(splitRatio) * len(features) // sumRatio
    partition = np.cumsum(partition)

    featureTrain, featureTest = np.array_split(features, partition[:-1])
    labelTrain, labelTest = np.array_split(labels, partition[:-1])

    print("featureTrain:", featureTrain.shape)
    print("featureTest:", featureTest.shape)
    print("labelTrain:", labelTrain.shape)
    print("labelTest:", labelTest.shape)

    #Train the Knn Model 
    print("Train the Knn Model")
    knn = cv.ml.KNearest_create()
    knn.train(featureTrain, cv.ml.ROW_SAMPLE, labelTrain)

    k = 4
    ret, prediction, neighbours, dist = knn.findNearest(featureTest, k)

    # Compute the accuracy:
    accuracy = (np.squeeze(prediction) == labelTest).mean() * 100
    print(f"Accuracy when k=4: {accuracy:.2f}%")
    
    return knn 