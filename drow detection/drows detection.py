from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import cv2
import dlib
import winsound

frequency = 3400
duration = 1000
count = 0

earthresh = 0.25  # dist between the vert eye coordinate threshold
earframe = 48  # consecutive frames for the eye closure
shapepred = "shape_predictor_68_face_landmarks.dat"  # file


def eyeAspectRatio(eye):
    # Vertical
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # Horiz
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


cam = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()  # instead of haar cascade frontl face algo
predictor = dlib.shape_predictor(shapepred)  # loading the file

# getting the coordinates of left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

while True:
    _, frame = cam.read()
    frame = imutils.resize(frame, width=1000)
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(grayscale, 0)
    for rect in rects:
        shape = predictor(grayscale, rect)
        shape = face_utils.shape_to_np(shape)

        lftco = shape[lStart:lEnd]
        rghtco = shape[rStart:rEnd]

        lefteye = eyeAspectRatio(lftco)
        righteye = eyeAspectRatio(rghtco)

        ear = (lefteye + righteye) / 2.0

        lefteyehull = cv2.convexHull(lftco)
        righteyehull = cv2.convexHull(rghtco)
        cv2.drawContours(frame, [lefteyehull], -1, (0, 0, 255), 1)
        cv2.drawContours(frame, [righteyehull], -1, (0, 0, 255), 1)

        if ear < earthresh:
            count += 1
        else:
            count = 0
        if count >= earframe:
            cv2.putText(frame, "DROWSINESS DETECTED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            winsound.Beep(frequency, duration)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
cam.release()
cv2.destroyAllWindows()
















