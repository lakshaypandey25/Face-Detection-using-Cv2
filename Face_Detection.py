import cv2

def main():
    # Load the Haar Cascade classifier file
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Check if the classifier was loaded correctly
    if face_cascade.empty():
        print("Error loading haarcascade_frontalface_default.xml")
        return

    # Capture video from the default camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error opening video capture")
        return

    while True:
        # Read the frame from the camera
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the grayscale image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        
        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Display the output frame
        cv2.imshow('Face Detection', frame)
        
        # Break the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close the windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
