import cv2
import os
import mediapipe as mp
import numpy as np

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
ALPHABET_SIGN_LIST = list(HAND_DESCRIPTIONS.keys())

SAMPLES_PER_LETTER = 100
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.6)
mp_drawing = mp.solutions.drawing_utils


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Index 0 failed, trying index 1...")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

for TARGET_LABEL in ALPHABET_SIGN_LIST:
    DATA_DIR = os.path.join("ArSL_Data", TARGET_LABEL)
    os.makedirs(DATA_DIR, exist_ok=True)

    existing_files = os.listdir(DATA_DIR)
    sample_count = len([f for f in existing_files if f.endswith('.npy')])
    collecting = False
    letter_done = False
    description = HAND_DESCRIPTIONS.get(TARGET_LABEL, "")

    print(f"\n--- NOW ON: {TARGET_LABEL.upper()} ---")
    print(f"Hand shape: {description}")
    print("Press 's' to start recording. Auto-stops after " + str(SAMPLES_PER_LETTER) + " samples.")
    print("Press 'n' to skip early. Press 'q' to quit.\n")

    while cap.isOpened() and not letter_done:
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if collecting:
                all_landmarks = []
                for hand_landmarks in results.multi_hand_landmarks:
                    for lm in hand_landmarks.landmark:
                        all_landmarks.extend([lm.x, lm.y, lm.z])

                if len(results.multi_hand_landmarks) == 1:
                    all_landmarks.extend(list(np.zeros(63)))
                    arr = np.array(all_landmarks)
                file_path = os.path.join(DATA_DIR, f"{TARGET_LABEL}_{sample_count}.npy")
                np.save(file_path, arr)
                sample_count += 1

                if sample_count % SAMPLES_PER_LETTER == 0:
                    collecting = False
                    letter_done = True
                    print(f"Done! Collected {SAMPLES_PER_LETTER} samples for '{TARGET_LABEL}'.")

        status_text = "RECORDING..." if collecting else "Press 's' to start"
        status_color = (0, 255, 0) if collecting else (0, 0, 255)

        cv2.putText(frame, f"Letter: {TARGET_LABEL}  |  Samples: {sample_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Shape: {description}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(frame, status_text, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame, "s = start/stop | n = skip | q = quit", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.namedWindow("ArSL Data Collector", cv2.WINDOW_NORMAL)
        cv2.imshow("ArSL Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            collecting = not collecting
        elif key == ord('n'):
            print(f"Skipped early.")
            collecting = False
            letter_done = True
        elif key == ord('q'):
            print("\nQuitting data collection early.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

cap.release()
cv2.destroyAllWindows()
print("\nAll letters processed! Now run train_model.py to retrain.") 


