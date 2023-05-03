import cv2

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier('./libraries/haarcascade_frontalface_default.xml')

# Open a video stream
cap = cv2.VideoCapture(0)

# Loop over the frames from the video stream
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Loop over the detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face on the original frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop the face from the original frame
        face = frame[y:y + h, x:x + w]

        # Save the cropped face to disk
        cv2.imwrite('./images/face.jpg', face)

        # Exit the loop
        break

    # Display the original frame with detected faces
    cv2.imshow('frame', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()
