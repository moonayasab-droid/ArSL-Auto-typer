# ♿ Arabic Sign Language (ArSL) Auto-Typer

**🔗 Live Web Page & Documentation:** [moonayasab-droid.github.io/ArSL-Auto-typer/](https://moonayasab-droid.github.io/ArSL-Auto-typer/)

![System Demo](demo.gif)

An end-to-end accessibility system that translates Arabic Sign Language (ArSL) gestures into real-time typed Arabic text and synthesized audio playback, bridging custom computer vision hardware prototyping with system-level automation.

---

## 🔑 Key Features

* **Dual-Hand Tracking:** Powered by Google MediaPipe to accurately capture and process 3D landmarks for up to two hands simultaneously ($21 \text{ landmarks per hand} \times 3 \text{ coordinates} = 63 \text{ features per hand} \times 2 \text{ hands} = 126 \text{ total features}$).
* **Custom Machine Learning Model:** Trained on a personalized dataset to classify specific ArSL hand gestures and letters.
* **Real-Time Automation:** Automatically maps predictions to Arabic characters, prevents duplicate spam via intelligent cooldown timers, and uses system-level automation to paste text directly into active windows (like Notepad).
* **Audio Synthesis Feedback:** Integrates real-time audio playback to ensure users can confirm translated letters audibly.

---

## 🛠 Tech Stack & Libraries

* **Python 3.11** - Core programming language.
* **OpenCV (`cv2`)** - Real-time video capture and frame processing.
* **MediaPipe** - Robust hand landmark detection and tracking.
* **Scikit-learn** - Machine learning classification and probability prediction (`model.predict_proba`).
* **PyAutoGUI / Keyboard** - System-level automation for direct text injection.

---

## 📂 Project Structure

```text
ArSL_AutoTyper/
│
├── ArSL_AutoTyper.py # Main real-time classification and auto-typing script
├── collect_samples.py # Phase 1: Gathers raw gesture image/video frames
├── collect_data.py # Phase 2: Extracts 3D landmark arrays from raw frames
├── train_model.py # Phase 3: Machine learning training pipeline
├── model.p # Serialized machine learning model and dictionaries
├── classes.py # Class labels mapping
└── README.md # Project documentation

