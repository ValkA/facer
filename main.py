import dlib
import cv2
import json
from gtts import gTTS
import os
import sys
import time

tts = gTTS(text='Hello Stranger', lang='en')
tts.save("hello.mp3")

predictor_path = "shape_predictor_5_face_landmarks.dat"
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"
faces_folder_path = "data"

with open('./data/data.json') as f:
    data = json.load(f)
faces = data['faces']

detector = dlib.get_frontal_face_detector() #a detector to find the data
sp = dlib.shape_predictor(predictor_path) #a shape predictor to find face landmarks so we can precisely localize the face
facerec = dlib.face_recognition_model_v1(face_rec_model_path) #face recognition model

video_capture = cv2.VideoCapture(0) #Webcam object
video_capture.set(3, 320)
video_capture.set(4, 240)


def euclidean_dist(vector_x, vector_y):
    if len(vector_x) != len(vector_y):
        raise Exception('Vectors must be same dimensions')
    return sum((vector_x[dim] - vector_y[dim]) ** 2 for dim in range(len(vector_x)))


def learn_new_face(face_descriptor, frame, d):
    x, y, w, h = [d.left(), d.top(), d.width(), d.height()]
    face_roi = frame[y:y+h, x:x+w]

    cv2.imshow("camera", face_roi)
    cv2.waitKey(1)
    save = input("do you want to save that face? [y/N]")
    if(save != 'y'):
        return None

    name = input("Enter name of the person")
    filename = name+".png"
    faces.append({"desc":face_descriptor ,"name": name, "image": filename})

    cv2.imwrite("./data/faces/"+filename, face_roi)
    return faces[-1]


def get_face_data(frame, d):
    shape = sp(frame, d)
    face1desc = facerec.compute_face_descriptor(frame, shape)  # 100
    face1desc = list(face1desc[dim] for dim in range(len(face1desc))) #convert to python tuple

    for face2 in faces:
        if euclidean_dist(face1desc, face2["desc"]) < 0.36:
            return face2

    if learn_new_faces == 'y':
        return learn_new_face(face1desc, frame, d)

    return None


learn_new_faces = input("Do you want to save new faces? [y/N]")
while True:
    ret, frame = video_capture.read()
    dets = detector(frame, 1)
    print("Number of data detected: {}".format(len(dets)))

    # Now process each face we found.
    for k, d in enumerate(dets):
        # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(  k, d.left(), d.top(), d.right(), d.bottom()))
        # Get the landmarks/parts for the face in box d.

        face_data = get_face_data(frame, d)
        if face_data is None:
            print("Hi Stranger")
            os.system("afplay hello.mp3")
            time.sleep(5)
            continue

        cv2.putText(frame, face_data["name"], (d.left(), d.top()), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        if "message" in face_data:
            cv2.putText(frame, face_data["message"], (d.left(), d.top()+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("camera", frame) #Display the frame
    if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
        break

with open('./data/data.json', 'w+') as f:
    json.dump(data, f)


