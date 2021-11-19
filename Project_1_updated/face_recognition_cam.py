import cv2 as cv 
import face_recognition

def cam_face_recog(filepath):

    #open the cam 
    capture = cv.VideoCapture(filepath)

    # load cascade model classifier
    # model already trained
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_eye.xml')

    videoWidth = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    videoHeight = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*'MPEG')
    video_out = cv.VideoWriter('myVid2.avi', fourcc, 20.0, (videoWidth,  videoHeight))

    while True:
        # Read the frame
        _, img = capture.read()

        # Convert to grayscale
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        face_landmarks_list_68 = face_recognition.face_landmarks(img)

        # Draw all detected landmarks:
        for face_landmarks in face_landmarks_list_68:
            for facial_feature in face_landmarks.keys():
                for p in face_landmarks[facial_feature]:
                    cv.circle(img, p, 2, (0, 255, 0), -1)

        # Detect the faces
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)


        # Draw the rectangle around each face
        for (x, y, w, h) in faces:
            cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]

            # Detect the eyes 
            eyes = eye_cascade.detectMultiScale(roi_gray)

            # Draw rectangle around the eyes
            for (ex, ey, ew, eh) in eyes:
                cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255), 2)

        video_out.write(img)
        
        # Display
        cv.imshow('Video', img)

        k = cv.waitKey(100)

        # check if key is q then exit
        if k == ord("q"):
            break

    capture.release()
    video_out.release()
    cv.destroyAllWindows()
    cv.waitKey(1)


cam_face_recog(0)

    



