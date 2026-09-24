# Technical Design Document: Arabic Sign Language (ArSL) Auto-Typer

## 1. System Overview
The Arabic Sign Language (ArSL) Auto-Typer is an end-to-end computer vision and automation pipeline designed to bridge communication barriers. It captures dual-hand gestures via webcam, extracts 3D spatial landmarks, classifies the sign using a trained machine learning model, and translates it into real-time typed Arabic text and synthesized audio.

---

## 2. Architecture & Pipeline

The system operates across three distinct modular phases:

### Phase 1: Data Gathering (`collect_samples.py`)
* Captures raw video frames from the webcam using OpenCV (`cv2`).
* Organizes and saves image samples per gesture/class to build a robust custom dataset.

### Phase 2: Feature Extraction (`collect_data.py`)
* Utilizes Google MediaPipe Hands to process raw frames.
* **The 126-Feature Array:** MediaPipe detects up to two hands simultaneously, identifying **21 key landmarks per hand**. Each landmark contains 3 spatial coordinates $(x, y, z)$.
$$\text{21 landmarks} \times 3 \text{ coordinates} = 63 \text{ features per hand}$$
$$\text{63 features} \times 2 \text{ hands} = 126 \text{ total features per frame}$$
* Serializes the extracted landmark arrays and labels into a dataset file (`data.pickle`).

### Phase 3: Model Training (`train_model.py`)
* Loads the processed landmark features and trains a machine learning classifier (using scikit-learn).
* Evaluates class probabilities (`predict_proba`) to ensure high-confidence predictions.
* Saves the optimized model parameters and dictionaries to `model.p`.

---

## 3. Real-Time Inference & Automation (`ArSL_AutoTyper.py`)
* **Live Capture Loop:** Streams frames at $\ge 25\text{-}30\text{ FPS}$, passing each frame through MediaPipe for real-time 126-feature extraction.
* **Classification & Cooldown:** Predicts the matching sign. Implements intelligent cooldown timers to prevent character duplication spam during sustained gestures.
* **System-Level Injection:** Uses `PyAutoGUI` / `keyboard` libraries to automatically inject the translated character into active applications (such as Notepad or text fields).
* **Audio Feedback:** Triggers text-to-speech (TTS) audio playback simultaneously to give the user auditory confirmation of the signed letter.

---

## 4. Limitations & Future Scope
* **Lighting & Occlusion:** Susceptible to performance drops in low-light conditions or during severe hand self-occlusion.
* **Next Steps:** Expanding the dataset to include diverse lighting angles and implementing adaptive thresholding for robust low-light tracking.


