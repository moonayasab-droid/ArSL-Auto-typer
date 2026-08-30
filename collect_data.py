import cv2

print("Opening webcam...")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Index 0 failed, trying index 1...")
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

is_running = True
fail_count = 0

while is_running and cap.isOpened():
    ret, frame = cap.read()

    if not ret or frame is None:
        fail_count += 1
        if fail_count > 30:
            print("Camera not returning frames. Exiting.")
            break
        continue

    fail_count = 0
    cv2.imshow('ArSL Data Collection', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Quitting data collection...")
        is_running = False

cap.release()
cv2.destroyAllWindows()