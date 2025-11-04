import cv2

cap = cv2.VideoCapture("http://192.168.1.103:4747/video")  # replace with your iPhone IP:port
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Direct iPhone Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()



# import cv2

# cap = cv2.VideoCapture("http://192.168.1.103:4747/video")
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     edges = cv2.Canny(frame, 100, 200)
#     cv2.imshow("Edges", edges)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()
