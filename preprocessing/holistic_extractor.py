import cv2
import mediapipe as mp
import time

# --------------------------------------------
# Initialize MediaPipe
# --------------------------------------------

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

drawing_spec = mp_draw.DrawingSpec(
    thickness=1,
    circle_radius=1,
    color=(0,255,0)
)

# --------------------------------------------
# Start Webcam
# --------------------------------------------

cap = cv2.VideoCapture(0)

prev_time = 0

# --------------------------------------------
# Holistic Model
# --------------------------------------------

with mp_holistic.Holistic(

    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=True,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.4

) as holistic:

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        # Mirror view
        frame = cv2.flip(frame, 1)

        # Improve detection
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=15)

        # Convert to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image.flags.writeable = False

        results = holistic.process(image)

        image.flags.writeable = True

        # --------------------------------------------
        # Draw Face
        # --------------------------------------------

        if results.face_landmarks:

            mp_draw.draw_landmarks(
                frame,
                results.face_landmarks,
                mp_holistic.FACEMESH_TESSELATION,
                drawing_spec,
                drawing_spec
            )

        # --------------------------------------------
        # Draw Left Hand
        # --------------------------------------------

        if results.left_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                drawing_spec,
                drawing_spec
            )

        # --------------------------------------------
        # Draw Right Hand
        # --------------------------------------------

        if results.right_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS,
                drawing_spec,
                drawing_spec
            )

        # --------------------------------------------
        # Draw Pose (Shoulders + Body)
        # --------------------------------------------

        if results.pose_landmarks:

            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS,
                drawing_spec,
                drawing_spec
            )

            # Highlight shoulders
            h, w, _ = frame.shape

            left_shoulder = results.pose_landmarks.landmark[11]
            right_shoulder = results.pose_landmarks.landmark[12]

            cv2.circle(frame,
                       (int(left_shoulder.x*w), int(left_shoulder.y*h)),
                       8,(255,0,0),-1)

            cv2.circle(frame,
                       (int(right_shoulder.x*w), int(right_shoulder.y*h)),
                       8,(255,0,0),-1)

        # --------------------------------------------
        # FPS
        # --------------------------------------------

        curr_time = time.time()

        fps = 1/(curr_time-prev_time) if curr_time!=prev_time else 0

        prev_time = curr_time

        # --------------------------------------------
        # UI Text
        # --------------------------------------------

        cv2.putText(frame,
                    f"FPS: {int(fps)}",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,0),
                    2)

        cv2.putText(frame,
                    "ISL Detection Running",
                    (20,75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255,255,255),
                    2)

        cv2.putText(frame,
                    "Press ESC to Exit",
                    (20,105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200,200,200),
                    1)

        # --------------------------------------------
        # Show Window
        # --------------------------------------------

        cv2.imshow("ISL Holistic Detection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break


# --------------------------------------------
# Release
# --------------------------------------------

cap.release()
cv2.destroyAllWindows()