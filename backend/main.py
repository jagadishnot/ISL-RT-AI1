"""
ISL-RT-AI1  —  FastAPI Backend
Routes:
  WS   /ws/stream       — send JPEG frames as base64, receive predictions
  POST /api/translate   — upload video file -> prediction
  GET  /api/vocab       — full vocabulary list
  GET  /health          — health check

Run:
  uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import sys
import tempfile
import asyncio
import json
import base64
import traceback

import cv2
import torch
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.cslt_model import CSLTModel
from training.tokenizer import build_vocab

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="ISL-RT-AI1 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model once at startup ────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[ISL] Device: {device}")

vocab, word_to_idx, idx_to_word = build_vocab()

model = CSLTModel(len(vocab))
model.load_state_dict(
    torch.load("best_cslt_gnn.pth", map_location=device, weights_only=True)
)
model.to(device).eval()
print(f"[ISL] Model loaded — vocab: {len(vocab)}")

# ── Constants ─────────────────────────────────────────────────────────────────

FACE_DIM       = 478 * 3
HAND_DIM       = 21  * 3
POSE_DIM       = 33  * 3
TOTAL_FEATURES = FACE_DIM + HAND_DIM + HAND_DIM + POSE_DIM   # 1659
MAX_FRAMES     = 120

# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_landmarks(results) -> list:
    lm = []
    for part, dim in [
        (results.face_landmarks,       FACE_DIM),
        (results.left_hand_landmarks,  HAND_DIM),
        (results.right_hand_landmarks, HAND_DIM),
        (results.pose_landmarks,       POSE_DIM),
    ]:
        if part:
            for p in part.landmark:
                lm.extend([p.x, p.y, p.z])
        else:
            lm.extend([0.0] * dim)
    return lm


def normalize(seq: np.ndarray) -> np.ndarray:
    mask = np.any(seq != 0, axis=1)
    if mask.sum() == 0:
        return seq
    valid = seq[mask]
    mean  = valid.mean(axis=0)
    std   = valid.std(axis=0)
    std[std < 1e-6] = 1.0
    seq[mask] = (seq[mask] - mean) / std
    return seq


def is_signing(sequence: list, threshold: float = 0.015) -> bool:
    if len(sequence) < 10:
        return False
    recent = np.array(sequence[-10:])
    hands  = recent[:, FACE_DIM: FACE_DIM + HAND_DIM + HAND_DIM]
    return float(np.diff(hands, axis=0).std()) > threshold


def decode_output(output: torch.Tensor) -> str:
    pred = torch.argmax(output.log_softmax(dim=2), dim=2)[0]
    prev, words = -1, []
    for p in pred:
        p = p.item()
        if p != prev and p != 0:
            words.append(idx_to_word.get(p, ""))
        prev = p
    return " ".join(w for w in words if w)


def run_inference(sequence: list) -> str:
    seq_np = normalize(np.array(sequence, dtype=np.float32))
    x = torch.tensor(seq_np, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        return decode_output(model(x))


def process_video_path(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"error": "Could not open video"}

    sequence     = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            image   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            lm      = extract_landmarks(results)
            sequence.append(lm if len(lm) == TOTAL_FEATURES else [0.0] * TOTAL_FEATURES)

    cap.release()

    if len(sequence) < 10:
        return {"prediction": "", "frames": len(sequence), "error": "Video too short"}

    if len(sequence) > MAX_FRAMES:
        idx      = np.linspace(0, len(sequence) - 1, MAX_FRAMES, dtype=int)
        sequence = [sequence[i] for i in idx]
    else:
        sequence += [[0.0] * TOTAL_FEATURES] * (MAX_FRAMES - len(sequence))

    pred = run_inference(sequence)
    return {
        "prediction": pred,
        "frames":     total_frames,
        "confidence": 0.95 if pred else 0.0,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "device": str(device), "vocab_size": len(vocab)}


@app.get("/api/vocab")
async def get_vocab():
    return {"vocab": list(word_to_idx.keys()), "size": len(vocab)}


@app.post("/api/translate")
async def translate_video(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[-1] or ".mp4"
    tmp    = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        tmp.write(await file.read())
        tmp.close()
        result = await asyncio.to_thread(process_video_path, tmp.name)
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


@app.websocket("/ws/stream")
async def websocket_stream(ws: WebSocket):
    """
    Client sends:   { "frame": "<base64 JPEG>" }
    Server replies: { "prediction": "...", "signing": true/false }
    """
    await ws.accept()
    sequence  = []
    last_pred = ""
    print("[WS] Client connected")

    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            b64 = msg.get("frame")
            if not b64:
                continue

            img_bytes             = base64.b64decode(b64)
            img_arr               = np.frombuffer(img_bytes, np.uint8)
            frame                 = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            image                  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable  = False
            results                = holistic.process(image)
            image.flags.writeable  = True

            lm = extract_landmarks(results)
            sequence.append(lm if len(lm) == TOTAL_FEATURES else [0.0] * TOTAL_FEATURES)
            sequence = sequence[-MAX_FRAMES:]

            signing = is_signing(sequence)

            if len(sequence) == MAX_FRAMES and signing:
                last_pred = await asyncio.to_thread(run_inference, list(sequence))
            elif not signing:
                last_pred = ""

            await ws.send_text(json.dumps({
                "prediction": last_pred,
                "signing":    signing,
            }))

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")
        traceback.print_exc()
    finally:
        holistic.close()