"""
Download required MediaPipe model files into the models/ directory.

Run once before starting the application:
    python scripts/download_models.py
"""

import pathlib
import urllib.request
import sys

MODELS = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    ),
}

MODELS_DIR = pathlib.Path(__file__).parent.parent / "models"


def download_all() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    for filename, url in MODELS.items():
        dest = MODELS_DIR / filename
        if dest.exists():
            print(f"  [OK] {filename} already present ({dest.stat().st_size:,} bytes)")
            continue
        print(f"  [..] Downloading {filename} ...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"done ({dest.stat().st_size:,} bytes)")
        except Exception as exc:
            print(f"\nERROR: {exc}", file=sys.stderr)
            sys.exit(1)
    print("All models ready.")


if __name__ == "__main__":
    download_all()
