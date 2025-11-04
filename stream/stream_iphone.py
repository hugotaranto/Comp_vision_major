

import cv2
import time
import os

# ==============================
# USER SETTINGS
# ==============================
# Replace this with the IP & port shown in the DroidCam app on your iPhone
IPHONE_STREAM_URL = "http://192.168.1.103:4747/video"

# Folder where images will be saved
SAVE_DIR = "captured_photos"

# Capture interval in seconds (set to 0 to disable automatic saving)
#0 means we disable all automatic saving. 
CAPTURE_INTERVAL = 0       # e.g. 5 → save one photo every 5 seconds
# ==============================


def ensure_folder(path):
    """Create folder if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    print("[INFO] Connecting to iPhone stream...")
    cap = cv2.VideoCapture(IPHONE_STREAM_URL)
    if not cap.isOpened():
        print("[ERROR] Could not connect to stream.")
        return

    ensure_folder(SAVE_DIR)
    print(f"[INFO] Saving photos to: {os.path.abspath(SAVE_DIR)}")
    print("[INFO] Press 'q' to quit, or 's' to save a manual snapshot.")

    last_capture_time = time.time()
    img_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Lost connection to stream.")
            break

        # Display the live video
        cv2.imshow("iPhone Stream", frame)

        # --- Manual snapshot ---
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            filename = os.path.join(SAVE_DIR, f"manual_{img_count:04d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"[SAVED] {filename}")
            img_count += 1

        # --- Timed snapshot ---
        if CAPTURE_INTERVAL > 0:
            now = time.time()
            if now - last_capture_time >= CAPTURE_INTERVAL:
                filename = os.path.join(SAVE_DIR, f"auto_{img_count:04d}.jpg")
                cv2.imwrite(filename, frame)
                print(f"[AUTO-SAVED] {filename}")
                last_capture_time = now
                img_count += 1

        # --- Exit key ---
        if key == ord('q'):
            print("[INFO] Quitting stream.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
