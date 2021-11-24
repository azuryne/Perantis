import cv2 as cv 
import matplotlib.pyplot as plt 
import os

for i in range (1,4):
    imageFile = f"img_{i}.png"
    img = cv.imread(imageFile)

    if not os.path.exists("dataset_new"):
        os.mkdir("dataset_new")

    folderPath = os.path.splitext(imageFile)[0]
    folderPath = f"dataset_new/{folderPath}"

    if not os.path.exists(folderPath):
        os.mkdir(folderPath)

point_list1 = []
point_list1.append(((348, 380), (572, 539), 'goke'))
point_list1.append(((659, 219), (880, 347), 'azureen'))
point_list1.append(((24, 15), (247, 163), 'saseendran'))
point_list1.append(((33, 227), (264, 357), 'mahmuda'))
point_list1.append(((636, 22), (887, 157), 'numan'))
point_list1.append(((309, 21), (580, 165), 'inamul'))
point_list1.append(((341, 215), (581, 360), 'gavin'))
point_list1.append(((31, 368), (243, 521), 'jincheng'))
point_list1.append(((663, 387), (871, 543), 'afiq'))

point_list2 = []
point_list2.append(((2319, 366), (2829, 676), 'goke'))
point_list2.append(((824, 341), (1369, 666), 'azureen'))
point_list2.append(((1564, 746), (2079, 1091), 'saseendran'))
point_list2.append(((854, 796), (1314, 1086), 'mahmuda'))
point_list2.append(((109,321), (709, 681), 'numan'))
point_list2.append(((234, 1265), (524, 1496), 'jincheng'))
point_list2.append(((1649, 376), (2099, 681), 'afiq'))

point_list3 = []
point_list3.append(((236, 55), (386, 148), 'goke'))
point_list3.append(((431, 41), (588, 149), 'azureen'))
point_list3.append(((439, 161), (601, 259), 'mahmuda'))
point_list3.append(((240,154), (397, 259), 'numan'))
point_list3.append(((70, 52), (210, 149), 'inamul'))
point_list3.append(((48, 168), (203, 260), 'afiq'))


count = 1 

# ls_of_ls = []

# for ls in ls_of_ls: 
for w in point_list1:

    ((x1, y1), (x2, y2), label) = w

    if y2 < y1: 
        y = y2
        y2 = y1 
        y1 = y

    if x2 < x1: 
        x = x2
        x2 = x1 
        x1 = x
    
    img1 = cv.imread('img_1.png')
    img_cropped = img1[y1:y2, x1:x2].copy()
    print(img_cropped.shape)

    saveFileName = 'dataset_new/img_1' + '/' + label + '.png'

    cv.imwrite(saveFileName, img_cropped)

    count+=1

for t in point_list2:

    ((x1, y1), (x2, y2), label) = t

    if y2 < y1: 
        y = y2
        y2 = y1 
        y1 = y

    if x2 < x1: 
        x = x2
        x2 = x1 
        x1 = x

    img2 = cv.imread('img_2.png')
    img_cropped = img2[y1:y2, x1:x2].copy()
    print(img_cropped.shape)

    saveFileName = 'dataset_new/img_2' + '/' + label + '.png'

    cv.imwrite(saveFileName, img_cropped)

    count+=1

for v in point_list3:

    ((x1, y1), (x2, y2), label) = v

    if y2 < y1: 
        y = y2
        y2 = y1 
        y1 = y

    if x2 < x1: 
        x = x2
        x2 = x1 
        x1 = x

    img3 = cv.imread('img_3.png')
    img_cropped = img3[y1:y2, x1:x2].copy()
    print(img_cropped.shape)

    saveFileName = 'dataset_new/img_3' + '/' + label + '.png'

    cv.imwrite(saveFileName, img_cropped)

    count+=1

        #ls_of_ls.append(list(point_list1), list(point_list2, list(point_list3)))