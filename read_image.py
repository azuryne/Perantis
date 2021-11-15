import cv2 as cv 

img = cv.imread('/Users/azureennaja/Desktop/Perantis/my_project/ca2.jpeg')
cv.imshow("Display Picture", img)

k = cv.waitKey(10000)

if k == ord("q"):
    print("thanks for open the image")
    
cv.destroyAllWindows()
cv.waitKey(1)