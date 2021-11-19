import cv2 as cv
import numpy as np 
import matplotlib.pyplot as plt 
from random import randbytes, shuffle
from numpy.random.mtrand import rand
import os 
import random


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


def test_userInput(filename, rand_num, knn):
    imgGray = cv.imread(filename, cv.IMREAD_GRAYSCALE)

    print("Image Shape:", imgGray.shape)

    SZ = 20 

    # resize
    rowNum = imgGray.shape[0] / SZ
    colNum = imgGray.shape[1] / SZ

    rows = np.vsplit(imgGray, rowNum)

    digits = []

    for row in rows:
        rowCells = np.hsplit(row, colNum)
        for digit in rowCells:
            digits.append(digit)

    # convert list to np.array 
    digits = np.array(digits)
    print("digits shape:", digits.shape)
    print("len digits", len(digits))

    #labels
    
    labels = np.asarray(rand_num)
    print(labels)

    print("labels", labels.shape)

    new_features = []
    for digit in digits: 
        img_pixel = np.float32(digit.flatten())
        new_features.append(img_pixel)
    
    new_features = np.squeeze(new_features)
    print("Features:", new_features.shape)

    k = 4

    ret, prediction, neighbours, dist = knn.findNearest(new_features, k)

    #Compute the accuracy
    accuracy = (labels == np.squeeze(prediction).flatten()).mean() * 100
    print(f"Accuracy of number combination: {accuracy:.2f}%")
    print(prediction.flatten())
    print()


def main():
    knn = knnModel()
    number_list = data_extraction()
    print("\n--Number Combination--\n")

    rand_num = []

    n = int(input("Enter the length combination random numbers (2-4): "))

    print("\n")

    if n <= 4:
        if n !=0:
            for i in range(0, n):
                print("Enter number at index", i)
                item = int(input())
                rand_num.append(item)
            print("User list is:", rand_num, "\n")
    else:
        print("Please re-enter")    

    print()

    print("shape:", rand_num)

    user_input(numbers_list=number_list, rand_num=rand_num)

    test_userInput("from_user.jpg", rand_num, knn)
    img = cv.imread("from_user.jpg")
    img = img[:,:,::-1]
    plt.imshow(img)
    plt.show()

    return rand_num


#Extract the list of digits from digits image 

def data_extraction():
    img = cv.imread('/Users/azureennaja/Desktop/Perantis/Project 1/Question2/digits.png')
    print("Image dimension", img.shape)

    numbers_list = []  #to create list for storing digits

    column = img.shape[0]
    row = img.shape[1]

    for column in range(0, 2000, 20):
        for row in range(0, 1000, 20):
            digits = img[column: column+20, row:row+20]
            numbers_list.append(digits)
    numbers_list = cv.vconcat(numbers_list)

    return numbers_list

#Get the user input to generate image consist of rand num insert by user
def user_input(numbers_list, rand_num):

    np.random.seed(2)

    # create an empty list to contain array of image based on rand_num / feature vector
    rand_num_img = []

    for x in range(0, len(rand_num), 1):
        digit1 = int(rand_num[x])
        a = digit1 * 5000
        b = a + 5000  #because col is twice the size of row
        value = []
        for y in range(a, b, 20):
            value.append(y)
            
  
        
        start_i = value[np.random.randint(0, len(value))]
        img = numbers_list[start_i: start_i + 20, :]
        rand_num_img.append(img) #list

    new_img = cv.hconcat(rand_num_img)
    cv.imwrite("from_user.jpg", new_img)



if __name__ == "__main__":
    main()








