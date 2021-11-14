import face_recognition
import cv2 as cv

# Open the input movie file
capture = cv.VideoCapture(0)

# Load some sample pictures and learn how to recognize them.
azu_image = face_recognition.load_image_file("Passport1.jpg")
azu_face_encoding = face_recognition.face_encodings(azu_image)[0]

known_faces = [
    azu_face_encoding,

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

    # top border
    frame[:15, :] = [10, 10, 100]

    # bottom border
    frame[-15:, :] = [80, 200, 80]

    # left border
    frame[:, :20] = [5, 100, 200]

    # right border
    frame[:, -20:] = [200, 10, 60]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    face_names = []

    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

        name = None
        if match[0]:
            name = "Azureen"
        else:
            name = "Unknown"

        face_names.append(name)

    # Label the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            continue

        # Draw a box around the face
        cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv.FILLED)
        font = cv.FONT_HERSHEY_DUPLEX
        cv.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        cv.imshow("Camera frame", frame)

    k = cv.waitKey(10)

    # check if key is q then exit
    if k == ord("q"):
        break

capture.release()
cv.destroyAllWindows()
cv.waitKey(1)
