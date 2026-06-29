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

# 0 disables all automatic saving
CAPTURE_INTERVAL = 0
# ==============================

def ensure_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    print("[INFO] Connecting to iPhone stream...")
    cap = cv2.VideoCapture(IPHONE_STREAM_URL)
    if not cap.isOpened():
        print("[ERROR] Could not connect to stream.")
        return

    ensure_folder(SAVE_DIR)
    print(f"[INFO] Saving PNG photos to: {os.path.abspath(SAVE_DIR)}")
    print("[INFO] Press 'q' to quit, or 's' to save a manual snapshot.")

    img_count = 0
    last_capture_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Lost connection to stream.")
            break

        cv2.imshow("iPhone Stream", frame)

        key = cv2.waitKey(10) & 0xFF
        # --- Manual snapshot ---
        if key == ord('s'):
            filename = os.path.join(SAVE_DIR, f"manual_{img_count:04d}.png")
            success = cv2.imwrite(filename, frame)  # PNG is lossless
            if success:
                print(f"[SAVED] {filename}")
            else:
                print("[ERROR] Failed to save PNG image.")
            img_count += 1

        # --- Timed snapshot (optional) ---
        if CAPTURE_INTERVAL > 0:
            now = time.time()
            if now - last_capture_time >= CAPTURE_INTERVAL:
                filename = os.path.join(SAVE_DIR, f"auto_{img_count:04d}.png")
                success = cv2.imwrite(filename, frame)
                if not success:
                    print(f"[ERROR] Failed to auto-save {filename}")
                print(f"[AUTO-SAVED] {filename}")
                last_capture_time = now
                img_count += 1

        # --- Exit ---
        if key == ord('q'):
            print("[INFO] Quitting stream.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
