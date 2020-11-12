import cv2 as cv
import dlib
import numpy as np
from tensorflow import keras

model = keras.models.load_model('model/fer.66.99.h5')

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

detector = dlib.get_frontal_face_detector()

# 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
class_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

while True:
    ret, frame = cap.read()

    if frame is None:
        print("frame is None")
        continue

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    frame = cv.resize(frame, (640, 360))
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    face_list = detector(gray, 1)
    print('len(face_list)', len(face_list))

    for index, face in enumerate(face_list):
        face_left = face.left()
        face_top = face.top()
        face_right = face.right()
        face_bottom = face.bottom()
        print('(face_left, face_top, face_right, face_bottom)', (face_left, face_top, face_right, face_bottom))

        if face_left < 0 or face_top < 0 or face_right < 0 or face_bottom < 0:
            print('face_left < 0 or face_top < 0 or face_right < 0 or face_bottom < 0')
            continue

        # rectangle
        cv.rectangle(
            frame,
            (face_left, face_top),
            (face_right, face_bottom),
            (0, 0, 255),
            3
        )

        # face rgb
        # face_cut = frame[face_top:face_bottom, face_left:face_right]
        # cv.imshow('face_cut' + str(+index), face_cut)

        # face gray
        face_gray_cut = gray[face_top:face_bottom, face_left:face_right]
        print('---')
        print('face_gray_cut', face_gray_cut)
        print('len(face_gray_cut)', len(face_gray_cut))
        if len(face_gray_cut) <= 0:
            continue
        # cv.imshow('face_gray_cut' + str(+index), face_gray_cut)

        # face gray resize
        face_gray_cut_resize = cv.resize(face_gray_cut, (48, 48))
        # cv.imshow('face_gray_cut_resize' + str(+index), face_gray_cut_resize)

        # expand dims (48, 48) -> (48, 48, 1)
        face_gray_for_predict = np.expand_dims(face_gray_cut_resize, -1)

        # expand dims (48, 48, 1) -> (1, 48, 48, 1)
        face_gray_for_predict = np.expand_dims(face_gray_for_predict, 0)

        # model predict
        predict_list = model.predict(face_gray_for_predict)[0]
        print('predict_list', predict_list)

        # class index
        class_index = np.ndarray.argmax(predict_list)
        print('class_index', class_index)

        # result text
        class_text = class_list[class_index]
        probability = str(round(predict_list[class_index], 2))
        result_text = class_text + ' ' + probability
        cv.putText(frame, result_text, (face_left, face_top), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv.LINE_AA)

    cv.imshow('camera', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
