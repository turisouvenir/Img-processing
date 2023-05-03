import cv2
import numpy as np

# Load the pre-trained face detection model
face_cascade = cv2.CascadeClassifier('./libraries/haarcascade_frontalface_default.xml')

# Load the hat image
hat = cv2.imread('./images/funny-hat.png', cv2.IMREAD_UNCHANGED)

# Open a video stream
cap = cv2.VideoCapture(0)

# Loop over the frames from the video stream
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5)

    # Loop over the detected faces
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face on the original frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop the face from the original frame
        face = frame[y:y + h, x:x + w]

        # Resize the hat image to fit the size of the detected face
        hat_resized = cv2.resize(hat, (w, h))

        # Create a mask for the hat image
        hat_mask = hat_resized[:, :, 3]

        # Create a mask for the face
        face_mask = np.zeros_like(gray)

        # Fill the face mask with ones in the region where the face is
        face_mask[y:y + h, x:x + w] = 1

        # Invert the face mask
        face_mask_inv = cv2.bitwise_not(face_mask)

        # Apply the hat mask to the hat image
        hat_resized = hat_resized[:, :, :3]
        hat_masked = cv2.bitwise_and(hat_resized, hat_resized, mask=hat_mask)

        # Apply the face mask to the original frame
        face_masked = cv2.bitwise_and(frame, frame, mask=cv2.cvtColor(face_mask_inv, cv2.COLOR_GRAY2BGR))

        # Overlay the masked hat image onto the masked face
        face_with_hat = cv2.add(face_masked, hat_masked)

        # Replace the original face region with the face with the hat
        frame[y:y + h, x:x + w] = face_with_hat

    # Display the original frame with detected faces and hat
    cv2.imshow('frame', frame)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the video stream and close all windows
cap.release()
cv2.destroyAllWindows()
