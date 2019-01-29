import dlib
import cv2
import json
import pyttsx
tts = pyttsx.init()

KEYFRAME = 25*1 #each KEYFRAME frames

cv2.namedWindow("camera", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

predictor_path = "shape_predictor_5_face_landmarks.dat"
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"
faces_folder_path = "data"

try:
    with open('./data/data.json') as f:
        data = json.load(f)
except Exception:
    data = {'faces':[]}
faces = data['faces']

detector = dlib.get_frontal_face_detector() #a detector to find the data
sp = dlib.shape_predictor(predictor_path) #a shape predictor to find face landmarks so we can precisely localize the face
facerec = dlib.face_recognition_model_v1(face_rec_model_path) #face recognition model

video_capture = cv2.VideoCapture(0) #Webcam object
video_capture.set(3, 640)
video_capture.set(4, 480)


def euclidean_dist(vector_x, vector_y):
    if len(vector_x) != len(vector_y):
        raise Exception('Vectors must be same dimensions')
    return sum((vector_x[dim] - vector_y[dim]) ** 2 for dim in range(len(vector_x)))


def learn_new_face(face_descriptor, frame, d):
    x, y, w, h = [d.left(), d.top(), d.width(), d.height()]
    face_roi = frame[y:y+h, x:x+w]

    cv2.imshow("face", face_roi)
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
counter = 0
tracked_faces = {}

tts.say('Good morning!')

while True:
    ret, frame = video_capture.read()

    if counter == KEYFRAME:
        counter = 0
        tracked_faces = {}
        dets = detector(frame, 1)
        for k, d in enumerate(dets):
            face_data = get_face_data(frame, d)
            if face_data is not None:
                face_data = dict(face_data)
                face_data['tracker'] = cv2.TrackerMedianFlow_create()
                face_data['tracker'].init(frame, (d.left(), d.top(), d.width(), d.height()))
                tracked_faces[face_data["name"]] = face_data

                tts.say(face_data["name"])
                if face_data.get('message'):
                    tts.say(str(face_data.get('message')))

                cv2.rectangle(frame, (d.left(),d.top()), (d.right(),d.bottom()), (255,0,0), 1)
                cv2.putText(frame, face_data["name"], (d.left(), d.top()), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                if "message" in face_data:
                    cv2.putText(frame, face_data["message"], (d.left(), d.top()+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        tts.runAndWait()
    else:
        for name, face_data in tracked_faces.items():
            success, d = face_data['tracker'].update(frame)
            if success:
                cv2.rectangle(frame, (int(d[0]),int(d[1])), (int(d[0]+d[2]),int(d[1]+d[3])), (0,0,255), 1)
                cv2.putText(frame, face_data["name"], (int(d[0]), int(d[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                if "message" in face_data:
                    cv2.putText(frame, face_data["message"], (int(d[0]), int(d[1]) + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("camera", frame) #Display the frame
    if cv2.waitKey(1) & 0xFF == ord('q'): #Exit program when the user presses 'q'
        break
    counter+=1

with open('./data/data.json', 'w+') as f:
    json.dump(data, f)


