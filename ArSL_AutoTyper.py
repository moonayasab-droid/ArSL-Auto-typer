import pyperclip
import pickle
import cv2
import mediapipe as mp
import numpy as np
import time
import pyautogui

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Paste it right here:
print("Starting auto-typer in 3 seconds... Click into your Notepad window NOW!")
time.sleep(3) 

HAND_DESCRIPTIONS = {
    'alif': "Thumb pointing straight up, fingers closed into a fist",
    'ba': "Index finger pointing straight up, other fingers curled",
    'ta': "Index and middle fingers pointing up together",
    'tha': "Index, middle, and ring fingers extended upward",
    'jiem': "Fingers curved downward in a claw shape",
    'ha': "Fingers curled loosely inward",
    'kha': "Hand held horizontally with fingers straight across",
    'dal': "Index finger curved pointing sideways like a hook",
    'thal': "Thumb and index finger pinching slightly open",
    'ra': "Single finger curved downward like a hook",
    'zay': "Finger curved downward with a tilted wrist",
    'seen': "All fingers spread wide and open vertically",
    'shien': "All five fingers spread wide apart, palm facing outward",
    'sad': "Fingers folded into a tight fist with thumb resting across",
    'dhad': "Fist shape with thumb extended out to the side",
    'tah': "Thumb touching index finger tip, other fingers extended up",
    'thalh': "Index finger pointing up with thumb touching side",
    'ayn': "Fingers curved into an open circle or 'C' shape",
    'ghayn': "Fingers curved into a 'C' shape rotated sideways",
    'fa': "Thumb and index finger forming a closed circle, others up",
    'qaf': "Fingers curled with index knuckle prominent",
    'kaf': "Fingers held up straight and tightly pressed together",
    'lam': "Index finger and thumb forming an L-shape",
    'miem': "Pinky finger pointing up while others are curled",
    'noon': "Index and middle fingers forming a curved horn shape",
    'ha_alt': "Fingers looped into a small circle shape",
    'waw': "Thumb curved downward pointing to a closed fist",
    'ya': "Pinky finger extended outward to the side"
}
ARABIC_LETTERS = {
    'alif': 'ا', 'ayn': 'ع', 'ba': 'ب', 'dad': 'ض', 'dal': 'د',
    'dhad': 'ض', 'fa': 'ف', 'ghayn': 'غ', 'ha': 'ح', 'ha_alt': 'ه',
    'haa': 'ح', 'jiem': 'ج', 'kaf': 'ك', 'kha': 'خ', 'khaa': 'خ',
    'lam': 'ل', 'meem': 'م', 'miem': 'م', 'no_sign': '', 'noon': 'ن',
    'qaf': 'ق', 'ra': 'ر', 'raa': 'ر', 'sad': 'ص', 'seen': 'س',
    'sheen': 'ش', 'shien': 'ش', 'ta': 'ت', 'taa': 'ت', 'taa_grid': 'ت',
    'tah': 'ط', 'tha': 'ث', 'thal': 'ذ', 'thalh': 'ظ', 'waw': 'و',
    'ya': 'ي', 'zay': 'ز'
}  
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Index 0 failed, trying index 1...")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) 
mp_drawing = mp.solutions.drawing_utils

last_typed_time = 0 
cooldown = 1.5
EXPECTED_FEATURES = 126

print("\n--- ULTIMATE AUTO-TYPER RUNNING ---")
print("Open Notepad or any text box, sign a letter, and watch it type!")
print("Press 'q' on the video window to exit.\n")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    display_text = "No Sign / Adjust Hand"
    description_text = "Show a clear hand shape"
    text_color = (0, 0, 255)

    if results.multi_hand_landmarks:
        all_landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for lm in hand_landmarks.landmark:
                all_landmarks.extend([lm.x, lm.y, lm.z])

        if len(results.multi_hand_landmarks) == 1:
            all_landmarks.extend([0.0] * 63)

        # Safety check — never crash on a shape mismatch, just skip this frame
        if len(all_landmarks) == EXPECTED_FEATURES:
            flattened_features = np.asarray(all_landmarks).reshape(1, -1)
            probabilities = model.predict_proba(flattened_features)[0]
            best_match_idx = np.argmax(probabilities)
            confidence = probabilities[best_match_idx]
            predicted_letter = model.classes_[best_match_idx]
            np.argmax(probabilities)
            confidence = probabilities[best_match_idx]
            predicted_letter = model.classes_[best_match_idx]
            if confidence > 0.55 and predicted_letter != 'no_sign':
                display_text = f"Letter: {predicted_letter} ({int(confidence*100)}%)"
                description_text = HAND_DESCRIPTIONS.get(predicted_letter.lower(), "Good hand alignment")
                text_color = (0, 255, 0)

                current_time = time.time()
                if current_time - last_typed_time > cooldown:
                    arabic_char = ARABIC_LETTERS.get(predicted_letter, predicted_letter)
                    pyperclip.copy(arabic_char)
                    pyautogui.hotkey('ctrl', 'v')
                    last_typed_time = current_time

    cv2.putText(frame, display_text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 3, cv2.LINE_AA)
    cv2.putText(frame, f"Tip: {description_text}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('ArSL Auto-Typer', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 