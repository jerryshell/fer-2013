import json

import cv2 as cv
import dlib
import numpy as np
import requests

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

detector = dlib.get_frontal_face_detector()

# 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral
class_list = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

while True:
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    if frame is None:
        print("frame is None")
        continue

    frame = cv.resize(frame, (640, 360))
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    face_list = detector(gray, 1)
    # print('len(face_list)', len(face_list))

    for index, face in enumerate(face_list):
        face_left = face.left()
        face_top = face.top()
        face_right = face.right()
        face_bottom = face.bottom()
        # print('(face_left, face_top, face_right, face_bottom)', (face_left, face_top, face_right, face_bottom))

        if face_left < 0 or face_top < 0 or face_right < 0 or face_bottom < 0:
            # print('face_left < 0 or face_top < 0 or face_right < 0 or face_bottom < 0')
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
        # print('---')
        # print('face_gray_cut', face_gray_cut)
        # print('len(face_gray_cut)', len(face_gray_cut))
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
        post_data = {
            "instances": face_gray_for_predict.tolist()
        }
        # print('post_data', post_data)
        response = requests.post(url='http://127.0.0.1:8501/v1/models/fer:predict', json=post_data)
        # print('response.text', response.text)

        responseJson = json.loads(response.text)
        # print('responseJson', responseJson)
        print('responseJson[\'predictions\']', responseJson['predictions'])

        # class index
        predictions = np.array(responseJson['predictions'])
        class_index = np.ndarray.argmax(predictions)
        print('class_index', class_index)

        # result text
        result_text = class_list[class_index] + ' ' + str(predictions[0][class_index])
        cv.putText(frame, result_text, (face_left, face_top), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv.LINE_AA)

    cv.imshow('camera', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
