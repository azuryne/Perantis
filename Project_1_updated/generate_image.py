import os 
import random
import cv2 as cv 
import matplotlib.pyplot as plt 
import numpy as np

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

    

               










