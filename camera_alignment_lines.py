import cv2


CAMERA_ID = 0

cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)

if not cap.isOpened():
    raise RuntimeError("Could not open camera")

print("Camera opened.")
print("Press q or ESC to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        break

    height, width = frame.shape[:2]

    center_x = width // 2
    center_y = height // 2

    # Vertical center line
    cv2.line(
        frame,
        (center_x, 0),
        (center_x, height),
        (0, 255, 0),
        2
    )

    # Horizontal center line
    cv2.line(
        frame,
        (0, center_y),
        (width, center_y),
        (0, 255, 0),
        2
    )

    # Small center point
    cv2.circle(
        frame,
        (center_x, center_y),
        5,
        (0, 0, 255),
        -1
    )

    cv2.putText(
        frame,
        f"Center: ({center_x}, {center_y})",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Camera Alignment Lines", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()