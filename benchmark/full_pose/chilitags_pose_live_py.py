import cv2
import numpy as np
import chilitags_py

CALIBRATION_FILE = "calibration_data.npz"
MARKER_SIZE_METERS = 0.097
AXIS_LENGTH = MARKER_SIZE_METERS * 0.5

data = np.load(CALIBRATION_FILE)
cam_matrix = data["mtx"]
dist_coeffs = data["dist"]

marker_3d_edges = np.array([
    [-MARKER_SIZE_METERS / 2,  MARKER_SIZE_METERS / 2, 0],
    [ MARKER_SIZE_METERS / 2,  MARKER_SIZE_METERS / 2, 0],
    [ MARKER_SIZE_METERS / 2, -MARKER_SIZE_METERS / 2, 0],
    [-MARKER_SIZE_METERS / 2, -MARKER_SIZE_METERS / 2, 0]
], dtype=np.float32)

detector = chilitags_py.Detector()

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    print("Could not open camera")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read frame")
        break

    detections = detector.detect(frame)

    for det in detections:
        marker_id = det["id"]
        corners = np.array(det["corners"], dtype=np.float32)

        success, rvec, tvec = cv2.solvePnP(
            marker_3d_edges,
            corners,
            cam_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:
            x, y, z = tvec.flatten()
            distance = np.linalg.norm(tvec)

            cv2.polylines(frame, [corners.astype(int)], True, (0, 255, 0), 2)

            cv2.drawFrameAxes(
                frame,
                cam_matrix,
                dist_coeffs,
                rvec,
                tvec,
                AXIS_LENGTH
            )

            cv2.putText(frame, f"ID: {marker_id}", tuple(corners[0].astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.putText(frame, f"Z: {z:.3f} m", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(frame, f"Distance: {distance:.3f} m", (30, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Chilitags Python Wrapper Pose", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
