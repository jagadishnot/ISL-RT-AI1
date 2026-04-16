import sys
import os
import cv2
import torch
import numpy as np
import mediapipe as mp
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.cslt_model import CSLTModel
from training.tokenizer import build_vocab
from tts.speak import speak

# ---------------------------------------------------------
# DEVICE
# ---------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Running on:", device)

# ---------------------------------------------------------
# VOCAB
# ---------------------------------------------------------

vocab, word_to_idx, idx_to_word = build_vocab()

# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

model = CSLTModel(len(vocab))
model.load_state_dict(
    torch.load("best_cslt_gnn.pth", map_location=device, weights_only=True)
)
model = model.to(device)
model.eval()
print("Model loaded")

# ---------------------------------------------------------
# MEDIAPIPE
# ---------------------------------------------------------

mp_holistic    = mp.solutions.holistic
mp_drawing     = mp.solutions.drawing_utils
mp_draw_styles = mp.solutions.drawing_styles

FACE_DIM       = 478 * 3
HAND_DIM       = 21  * 3
POSE_DIM       = 33  * 3
TOTAL_FEATURES = FACE_DIM + HAND_DIM + HAND_DIM + POSE_DIM  # 1659

# ---------------------------------------------------------
# LANDMARK EXTRACTION
# ---------------------------------------------------------

def extract_landmarks(results):

    landmarks = []

    if results.face_landmarks:
        for lm in results.face_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * FACE_DIM)

    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * HAND_DIM)

    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * HAND_DIM)

    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * POSE_DIM)

    return landmarks


# ---------------------------------------------------------
# NORMALIZE
# ---------------------------------------------------------

def normalize(seq):

    mask = np.any(seq != 0, axis=1)

    if mask.sum() == 0:
        return seq

    valid           = seq[mask]
    mean            = valid.mean(axis=0)
    std             = valid.std(axis=0)
    std[std < 1e-6] = 1.0
    seq[mask]       = (seq[mask] - mean) / std

    return seq


# ---------------------------------------------------------
# MOTION DETECTION
# only predict when hands are actively moving
# ---------------------------------------------------------

def is_signing(sequence, threshold=0.015):

    if len(sequence) < 10:
        return False

    recent = np.array(sequence[-10:])

    # focus on hand landmarks only (face index 1434 onward)
    hands  = recent[:, FACE_DIM : FACE_DIM + HAND_DIM + HAND_DIM]

    motion = np.diff(hands, axis=0).std()

    return motion > threshold


# ---------------------------------------------------------
# CTC DECODER
# ---------------------------------------------------------

def decode(output):

    log_probs = output.log_softmax(dim=2)
    pred      = torch.argmax(log_probs, dim=2)[0]

    prev  = -1
    words = []

    for p in pred:
        p = p.item()
        if p != prev and p != 0:
            words.append(idx_to_word.get(p, ""))
        prev = p

    return " ".join(words)


# ---------------------------------------------------------
# CAMERA
# ---------------------------------------------------------

cap       = cv2.VideoCapture(0)
sequence  = []
sentence  = ""
FPS       = 0
prev_time = time.time()

last_spoken      = ""
last_spoken_time = time.time()
SPEAK_DELAY      = 2.5

# signing state for UI indicator
signing_active = False

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    refine_face_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True

        landmarks = extract_landmarks(results)

        if len(landmarks) != TOTAL_FEATURES:
            landmarks = [0.0] * TOTAL_FEATURES

        sequence.append(landmarks)
        sequence = sequence[-120:]

        # check if person is actively signing
        signing_active = is_signing(sequence)

        if len(sequence) == 120 and signing_active:

            seq_np = np.array(sequence, dtype=np.float32)
            seq_np = normalize(seq_np)

            x = torch.tensor(seq_np, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(x)

            sentence = decode(output)

            now = time.time()
            if (sentence and
                sentence != last_spoken and
                now - last_spoken_time > SPEAK_DELAY):
                speak(sentence)
                last_spoken      = sentence
                last_spoken_time = now

        elif not signing_active:
            # clear prediction when not signing
            sentence = ""

        # -----------------------------------------------------
        # DRAW LANDMARKS
        # -----------------------------------------------------

        mp_drawing.draw_landmarks(
            frame, results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )
        mp_drawing.draw_landmarks(
            frame, results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS
        )

        # -----------------------------------------------------
        # UI
        # -----------------------------------------------------

        current_time = time.time()
        FPS          = 1 / max(current_time - prev_time, 1e-6)
        prev_time    = current_time

        # signing status indicator
        status_color = (0, 255, 0) if signing_active else (0, 0, 255)
        status_text  = "Signing..." if signing_active else "Waiting..."
        cv2.circle(frame, (frame.shape[1] - 30, 30), 12, status_color, -1)
        cv2.putText(frame, status_text,
                    (frame.shape[1] - 130, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # prediction
        cv2.putText(frame, f"Prediction: {sentence}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # FPS
        cv2.putText(frame, f"FPS: {int(FPS)}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # exit hint
        cv2.putText(frame, "Press ESC to exit",
                    (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow("ISL Real-Time Translator", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()