import cv2
live_stream = cv2.VideoCapture(0)
if not live_stream.isOpened():
    print("Error opening the webcam")

# setting the video frame size
live_stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
live_stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True :
    ret,frame=live_stream.read()
    if ret:
        cv2.imshow("Myself",cv2.Canny(frame,100,100))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break;
    else:
        break;

live_stream.release();
cv2.destroyAllWindows()
