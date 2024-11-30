import cv2
import numpy as np

# Helper function to detect and count fingers
def count_fingers(contour, drawing):
    hull = cv2.convexHull(contour, returnPoints=False)
    if len(hull) > 3:  # Only proceed if the hull is large enough
        defects = cv2.convexityDefects(contour, hull)
        if defects is None:
            return 0

        count = 0 

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            a = np.linalg.norm(np.array(start) - np.array(end))
            b = np.linalg.norm(np.array(far) - np.array(start))
            c = np.linalg.norm(np.array(far) - np.array(end))
            angle = np.arccos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # Cosine Rule

            if angle <= np.pi / 2:  # Angle less than 90 degrees, finger detected
                count += 1
                cv2.circle(drawing, far, 8, [211, 84, 0], -1)

        return count
    return 0

# Start capturing video from webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame to avoid mirrored view
    frame = cv2.flip(frame, 1)

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the frame  
    blur = cv2.GaussianBlur(gray, (35, 35), 0)

    # Apply threshold to get a binary image -- 
    _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the largest contour (if any)
    if len(contours) > 0:
        max_contour = max(contours, key=cv2.contourArea)

        # Create an empty canvas to draw results on
        drawing = np.zeros(frame.shape, np.uint8)

        # Draw the contour
        cv2.drawContours(drawing, [max_contour], -1, (0, 255, 0), 2)

        # Count the number of fingers
        finger_count = count_fingers(max_contour, drawing)

        # Display the count of fingers on the frame
        cv2.putText(frame, f"Fingers: {finger_count + 1}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Show the original frame and the processed result
    cv2.imshow('Hand Detection', frame)
    cv2.imshow('Thresh', thresh)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
