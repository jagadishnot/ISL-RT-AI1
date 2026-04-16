import cv2
import mediapipe as mp

video="data/videos/Videos_Sentence_Level/are you free today/are you free today.mp4"

cap=cv2.VideoCapture(video)

mp_holistic=mp.solutions.holistic
mp_draw=mp.solutions.drawing_utils

with mp_holistic.Holistic(model_complexity=2) as holistic:

    ret,frame=cap.read()

    frame=cv2.resize(frame,(1280,960))

    image=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    results=holistic.process(image)

    print("Face:",results.face_landmarks is not None)
    print("Left hand:",results.left_hand_landmarks is not None)
    print("Right hand:",results.right_hand_landmarks is not None)
    print("Pose:",results.pose_landmarks is not None)