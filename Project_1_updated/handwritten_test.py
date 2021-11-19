from random import randbytes, shuffle
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from numpy.random.mtrand import rand
import generate_image
from model import knnModel

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
    number_list = generate_image.data_extraction()
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

    generate_image.user_input(numbers_list=number_list, rand_num=rand_num)

    test_userInput("from_user.jpg", rand_num, knn)
    img = cv.imread("from_user.jpg")
    img = img[:,:,::-1]
    plt.imshow(img)
    plt.show()

    return rand_num



if __name__ == "__main__":
    main()








