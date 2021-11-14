import cv2 as cv
import sys

# Load the cascade
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# To capture video from existing video.
filePath = sys.argv[1]
cap = cv.VideoCapture(filePath)

while True:
    # Read the frame
    _, img = cap.read()

    # Convert to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Display
    cv.imshow('Video', img)

    k = cv.waitKey(10)

    # check if key is q then exit
    if k == ord("q"):
        break

cv.destroyAllWindows()
cv.waitKey(1)

