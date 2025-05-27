import cv2

# Load Haar cascade files for face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Start webcam capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from webcam.")
        break

    # Convert frame to grayscale (required for detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=2, minNeighbors=4)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame , (x, y), (x + w, y + h), (255, 0, 0), 1)

        # Focus on the face area (region of interest)
        face_gray = gray[y:y + h, x:x + w]
        face_color = frame[y:y + h, x:x + w]

        # Detect eyes inside the face area
        eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=7)

        for (ex, ey, ew, eh) in eyes[:10]:  # Limit to 2 eyes
            cv2.rectangle(face_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 1)

    # Display the result
    cv2.imshow("Face & Eye Detection", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
