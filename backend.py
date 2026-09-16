
import os
import io
import base64
import tempfile
from pathlib import Path

import numpy as np
import librosa
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

SAMPLE_RATE = 16000
AUDIO_DURATION = 4
NUM_SAMPLES = SAMPLE_RATE * AUDIO_DURATION
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256

MODEL_PATH = Path(os.environ.get("MODEL_PATH", Path(__file__).resolve().parent / "best.pt"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SyntheticSpeechCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_model():
    if not MODEL_PATH.exists():
        return None

    checkpoint = torch.load(str(MODEL_PATH), map_location=DEVICE, weights_only=False)

    # Accept the exact training checkpoint format used in the project,
    # raw state_dicts, and a few common checkpoint wrappers.
    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    else:
        state = None
        if isinstance(checkpoint, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                if key in checkpoint:
                    state = checkpoint[key]
                    break
            if state is None and checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                state = checkpoint
        if state is None:
            raise RuntimeError("Unsupported model checkpoint format.")

        state = {
            (k[7:] if k.startswith("module.") else k): v
            for k, v in state.items()
        }
        model = SyntheticSpeechCNN()
        model.load_state_dict(state, strict=True)

    model.to(DEVICE)
    model.eval()
    return model


try:
    model = load_model()
    MODEL_ERROR = None
except Exception as e:
    model = None
    MODEL_ERROR = str(e)

app = FastAPI(title="Synthetic Speech Detector", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def decode_audio(audio_bytes: bytes, filename: str):
    """
    Decode audio using a temporary file with the original extension.

    This mirrors the Colab inference path:
        librosa.load(path, sr=16000, mono=True)
    """
    suffix = Path(filename or "audio.wav").suffix.lower()

    allowed = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm", ".aac"}
    if suffix not in allowed:
        suffix = ".wav"

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            temp_path = tmp.name

        audio, _ = librosa.load(
            temp_path,
            sr=SAMPLE_RATE,
            mono=True
        )

        if len(audio) == 0:
            raise ValueError("No audio samples found.")

        return audio

    except Exception as e:
        raise ValueError(f"Could not decode this audio file: {e}")

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

def make_features(audio_bytes: bytes, filename: str):
    audio = decode_audio(audio_bytes, filename)
    original_duration = len(audio) / SAMPLE_RATE

    # Same 4-second preprocessing used during project inference.
    if len(audio) < NUM_SAMPLES:
        audio = np.pad(audio, (0, NUM_SAMPLES - len(audio)))
    else:
        audio = audio[:NUM_SAMPLES]

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)

    x = torch.tensor(
        mel_norm, dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0).to(DEVICE)

    return x, mel_db, original_duration


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "device": str(DEVICE),
        "model_error": MODEL_ERROR,
    }


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)):
    global model

    if model is None:
        detail = "Model is not loaded. Put your checkpoint in this folder as best.pt."
        if MODEL_ERROR:
            detail += f" Checkpoint error: {MODEL_ERROR}"
        raise HTTPException(503, detail)

    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty audio file.")

    try:
        x, mel_db, duration = make_features(data, file.filename or "audio.wav")

        with torch.inference_mode():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()

        bonafide = float(probs[0])
        spoof = float(probs[1])
        prediction = "BONAFIDE" if bonafide >= spoof else "SPOOF"
        confidence = max(bonafide, spoof)

        # Compact normalized spectrogram for display.
        lo, hi = np.percentile(mel_db, [2, 98])
        spec = np.clip((mel_db - lo) / max(hi - lo, 1e-8), 0, 1)
        spec_u8 = (spec * 255).astype(np.uint8)

        return {
            "prediction": prediction,
            "confidence": round(confidence * 100, 2),
            "bonafide_probability": round(bonafide * 100, 2),
            "spoof_probability": round(spoof * 100, 2),
            "duration": round(duration, 2),
            "filename": file.filename,
            "spectrogram": base64.b64encode(spec_u8.tobytes()).decode("ascii"),
            "spectrogram_shape": list(spec_u8.shape),
        }

    except Exception as e:
        raise HTTPException(400, f"Could not analyze this audio: {e}")
