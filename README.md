# 🤖 Gesture AI Assistant

A powerful desktop application that lets you control your computer using **hand gestures** and **voice commands**. Built with Python, OpenCV, MediaPipe, and PyAutoGUI.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-orange)
![OpenAI](https://img.shields.io/badge/ChatGPT-Voice-white?logo=openai)

---

## ✨ Key Features

| Feature | Details |
|---|---|
| **Virtual Mouse** | Move the cursor smoothly with your index finger, left-click, and scroll up/down. |
| **Gesture Macros** | Trigger system actions (media controls, volume, screenshots, app launch) instantly. |
| **Voice Assistant** | Say "assistant" to chat with ChatGPT. It responds with text-to-speech. |
| **Smart HUD** | Live on-screen overlays show current gesture, mouse state, confidence, and FPS. |
| **Debouncing** | Built-in cooldowns and confidence thresholds prevent accidental triggers. |

---

## ✋ Gesture Controls

Control your PC with these simple hand shapes. A reference card is always displayed on-screen.

| Gesture | Action |
|---------|--------|
| **☝️ Pointing** (Index only) | **Move Mouse** |
| **✌️ Two Fingers** | **Left Click** |
| **🤟 Three Fingers** | **Scroll Mode** (Move hand up/down) |
| **🖖 Four Fingers** (No thumb) | **Open Chrome** |
| **🖐️ Open Palm** | **Screenshot** |
| **✊ Fist** | **Play / Pause Media** |
| **👍 Thumbs Up** | **Volume Up** |
| **👎 Thumbs Down** | **Volume Down** |

---

## 🎙️ Voice Assistant

The assistant listens in the background. 
1. Say **"assistant"** to wake it up.
2. Ask your question or give a command (e.g., "assistant, what's the weather today?").
3. The AI (ChatGPT) will process your request and speak the answer aloud.
4. Voice feedback is also provided for gesture actions (e.g., it says "Screenshot taken").

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.9+**
- A working **webcam**
- An **OpenAI API Key** (for the voice assistant)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/Gesture-AI-Assistant.git
cd Gesture-AI-Assistant

# 2. Create a virtual environment (recommended)
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Run

```bash
python main.py
```

Press **Q** or **Esc** to quit the application.

---

## ⚙️ Configuration

All settings live in [`src/config.py`](src/config.py) and `.env`. Key options:

| Parameter | Default | Description |
|---|---|---|
| `CAMERA_INDEX` | `0` | Webcam device index |
| `GESTURE_CONFIDENCE_THRESHOLD` | `0.6` | Minimum confidence to trigger actions |
| `MOUSE_SMOOTHING` | `0.35` | Higher = smoother but more lag |
| `CHATGPT_MODEL` | `gpt-4o-mini` | LLM model for the voice assistant |
| `VOICE_WAKE_WORD` | `"assistant"` | Trigger phrase for voice commands |

---

## 📁 Project Structure

```text
Gesture-AI-Assistant/
├── main.py                 # Application entry point & orchestration
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variables template
└── src/
    ├── config.py           # Tunable constants & thresholds
    ├── hand_tracker.py     # MediaPipe wrapper for reliable tracking
    ├── gesture_recognizer.py # Logic to turn landmarks into gesture labels
    ├── mouse_controller.py # Maps landmarks to PyAutoGUI movements
    ├── actions.py          # Action dispatcher with debouncing
    ├── voice_assistant.py  # Speech recognition & Text-to-Speech
    ├── chatgpt_service.py  # OpenAI API integration
    └── display.py          # OpenCV drawing (HUD, skeleton, text)
```

---

## 📄 License

This project is released under the [MIT License](LICENSE).
