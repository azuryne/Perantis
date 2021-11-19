#PROJECT 1 Q3 PART 1

# IMAGE DETECTION USING CAFFE (Tensorflow)

import cv2 as cv
import numpy as np

def read_image(filepath):
    img = cv.imread(filepath)

    # image dimension
    h, w = img.shape[:2]

    # load model
    modelFile = '/Users/azureennaja/Desktop/Perantis/Project 1/Project_1_Q3/opencv_face_detector_uint8.pb'
    configFile = '/Users/azureennaja/Desktop/Perantis/Project 1/Project_1_Q3/opencv_face_detector.pbtxt.txt'
    model = cv.dnn.readNetFromTensorflow(modelFile, configFile)

    # preprocessing
    # image resize to 300x300 by substraction mean vlaues [104., 117., 123.]
    blob = cv.dnn.blobFromImage(img, 1.0, (300, 300), [
        104., 117., 123.], False, False)

    # set blob asinput and detect face
    model.setInput(blob)
    detections = model.forward()

    faceCounter = 0
    # draw detections above limit confidence > 0.7
    for i in range(0, detections.shape[2]):
        # confidence
        confidence = detections[0, 0, i, 2]
        #
        if confidence > 0.7:
            # face counter
            faceCounter += 1
            # get coordinates of the current detection
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Draw the detection and the confidence:
            cv.rectangle(img, (x1, y1), (x2, y2), (255, 154, 0), 3)
            text = "{:.2f}%".format(confidence * 100)
            y = y1 - 10 if y1 - 10 > 10 else y1 + 10
            cv.putText(img, text, (x1, y), cv.FONT_HERSHEY_SIMPLEX, 0.50, (25, 235, 25), 2)

            cv.imshow('Image', img)

            cv.imwrite("friends.png", img)

            k = cv.waitKey(1000)

            if k == ord("q"):
                break

    cv.destroyAllWindows()
    cv.waitKey(1)


read_image("/Users/azureennaja/Desktop/Perantis/Project 1/Project_1_Q3/class18.jpeg")