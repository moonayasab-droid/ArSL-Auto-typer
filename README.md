# Arabic Sign Language (ArSL) Auto-Typer

An end-to-end accessibility system that translates Arabic Sign Language (ArSL) gestures into real-time typed Arabic text. This project bridges custom hardware prototyping with computer vision, dual-hand landmark extraction, and machine learning classification to automate text input seamlessly into any active application.

## 🚀 Key Features
- **Dual-Hand Tracking:** Powered by Google MediaPipe to accurately capture and process 3D landmarks for up to two hands simultaneously ($EXPECTED_FEATURES = 126$).
- **Custom Machine Learning Model:** Trained on a personalized dataset to classify specific ArSL hand gestures and letters.
- **Real-Time Automation:** Automatically maps predictions to Arabic characters, prevents duplicate spam via intelligent cooldown timers, and uses `pyperclip` and `pyautogui` to paste text directly into active windows (like Notepad).
- **Hardware Integration:** Designed to work hand-in-hand with custom physical glove prototypes for enhanced gesture capture.

## 🛠️ Tech Stack & Libraries
- **Python 3.11**
- **OpenCV (`cv2`)** - Real-time video capture and frame processing.
- **MediaPipe** - Robust hand landmark detection and tracking.
- **Scikit-learn** - Machine learning classification and probability prediction (`model.predict_proba`).
- **Pyautogui & Pyperclip** - System-level automation and clipboard management for text injection.

## 📂 Project Structure
```text
ArSL_AutoTyper/
│
├── ArSL_AutoTyper.py       # Main real-time classification and auto-typing script
├── collect_data.py         # Script for gathering custom gesture data
├── train_model.py          # Machine learning training pipeline
├── model.p                 # Serialized machine learning model and dictionaries
├── classes.py              # Class labels mapping
└── README.md               # Project documentation
