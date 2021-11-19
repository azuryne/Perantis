import face_recognition
import cv2 as cv
import sys
import os

def read_video(filePath):

    # Open the input movie file
    capture = cv.VideoCapture(filePath)

    ret, frame = capture.read()
    h, w, _ = frame.shape

    fourcc = cv.VideoWriter_fourcc(*"MPEG")
    writers = cv.VideoWriter(f"Two.avi", fourcc, 20.0, (w, h))

    # Load some sample pictures and learn how to recognize them.
    zayn_image = face_recognition.load_image_file("/Users/azureennaja/Desktop/Perantis/Project 1/Project_1_Q3/zayn.jpeg")
    zayn_face_encoding = face_recognition.face_encodings(zayn_image)[0]

    harry_image = face_recognition.load_image_file("/Users/azureennaja/Desktop/Perantis/Project 1/Project_1_Q3/harry.jpeg")
    harry_face_encoding = face_recognition.face_encodings(harry_image)[0]

    liam_image = face_recognition.load_image_file("/Users/azureennaja/Desktop/Perantis/Project 1/Project_1_Q3/liam.jpeg")
    liam_face_encoding = face_recognition.face_encodings(liam_image)[0]

    niall_image = face_recognition.load_image_file("/Users/azureennaja/Desktop/Perantis/Project 1/Project_1_Q3/niall.jpeg")
    niall_face_encoding = face_recognition.face_encodings(niall_image)[0]

    louis_image = face_recognition.load_image_file("/Users/azureennaja/Desktop/Perantis/Project 1/Project_1_Q3/louis.jpeg")
    louis_face_encoding = face_recognition.face_encodings(louis_image)[0]

    jimmy_image = face_recognition.load_image_file("/Users/azureennaja/Desktop/Perantis/Project 1/Project_1_Q3/jimmy.jpeg")
    jimmy_face_encoding = face_recognition.face_encodings(jimmy_image)[0]

    known_faces = [
        zayn_face_encoding,
        harry_face_encoding,
        liam_face_encoding,
        niall_face_encoding,
        louis_face_encoding,
        jimmy_face_encoding
    ]

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    frame_number = 0

    while True:
        # Grab a single frame of video
        ret, frame = capture.read()
        frame_number += 1

        # Quit when the input video file ends
        if not ret:
            break

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(frame)
        face_encodings = face_recognition.face_encodings(frame, face_locations)

        face_names = []

        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

            name = None
            if match[0]:
                name = "Zayn"
            elif match[1]:
                name = "Harry"
            elif match[2]:
                name = "Liam"
            elif match[3]:
                name = "Niall"
            elif match[4]:
                name = "Louis"
            elif match[5]:
                name = "Jimmy"

            else:
                name = "Unknown"

            face_names.append(name)

        # Label the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            if not name:
                continue

            # Draw a box around the face
            cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 1)

            # Draw a label with a name below the face
            cv.rectangle(frame, (left, bottom - 10), (right, bottom), (0, 0, 255), cv.FILLED)
            font = cv.FONT_HERSHEY_DUPLEX
            cv.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

            cv.imshow("Camera frame", frame)
            writers.write(frame)

        k = cv.waitKey(10)

        # check if key is q then exit
        if k == ord("q"):
            break

    capture.release()
    writers.release()
    cv.destroyAllWindows()
    cv.waitKey(1)

#video = sys.argv[1]
#read_video("/Users/azureennaja/Desktop/Perantis/Project 1/Project_1_Q3/1D_1_1.mp4")



