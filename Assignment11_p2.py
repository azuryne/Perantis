import sys
import cv2 as cv

print("commandline argument array \n", sys.argv)
print()
print("first argument: ", sys.argv[0])
print("second argument: ", sys.argv[1])
print("third argument: ", sys.argv[2])

filePath = sys.argv[1]

capture = cv.VideoCapture(filePath)

# check if connected
if capture.isOpened() is False:
    print("Error opening camera 0")
    exit()

while capture.isOpened():

    # capture frames, if read correctly ret is True
    ret, frame = capture.read()

    if not ret:
        print("Didn't receive frame. Stop ")
        break

    # display frame
    cv.imshow("Camera frame", frame)

    k = cv.waitKey(10)

    # check if key is q then exit
    if k == ord("q"):
        break

capture.release()
cv.destroyAllWindows()
cv.waitKey(1)
