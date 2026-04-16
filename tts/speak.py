"""
speak.py — Windows-compatible TTS using fresh subprocess per call.
Bypasses the pyttsx3 background-thread bug on Windows where
runAndWait() silently fails after the first call.
"""

import queue
import threading
import subprocess
import sys

_q = queue.Queue(maxsize=1)
_started = False

def _worker():
    while True:
        text = _q.get()
        if text is None:
            break
        try:
            # Spawn a completely fresh Python process each time
            # This bypasses the pyttsx3 COM thread restriction on Windows
            script = (
                "import pyttsx3;"
                "engine = pyttsx3.init();"
                "engine.setProperty('rate', 150);"
                "engine.setProperty('volume', 1.0);"
                f"engine.say({repr(text)});"
                "engine.runAndWait();"
                "engine.stop()"
            )
            subprocess.run(
                [sys.executable, "-c", script],
                timeout=15,
                capture_output=True
            )
        except subprocess.TimeoutExpired:
            pass
        except Exception as e:
            print(f"[TTS] Error: {e}")
        finally:
            _q.task_done()

def _ensure_started():
    global _started
    if not _started:
        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        _started = True

def speak(text: str):
    """Non-blocking. Skips if already speaking."""
    if not text or not text.strip():
        return
    _ensure_started()
    try:
        _q.put_nowait(text.strip())
    except queue.Full:
        pass  # already speaking, skip

def stop():
    """Graceful shutdown."""
    try:
        _q.put_nowait(None)
    except queue.Full:
        pass