import cv2
import numpy as np


# -----------------------------
# Load camera calibration
# -----------------------------
try:
    with np.load("calibration_data.npz") as data:
        cam_matrix = data["mtx"]
        dist_coeffs = data["dist"]
except FileNotFoundError:
    raise RuntimeError(
        "Please run camera calibration first and save 'calibration_data.npz'"
    )


# -----------------------------
# Marker settings
# -----------------------------
MARKER_SIZE_METERS = 0.097
AXIS_LENGTH = MARKER_SIZE_METERS * 0.5

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

detector = cv2.aruco.ArucoDetector(
    aruco_dict,
    cv2.aruco.DetectorParameters()
)


# -----------------------------
# 3D marker corner coordinates
# Marker is centered at origin
# -----------------------------
half_size = MARKER_SIZE_METERS / 2

marker_3d_edges = np.array([
    [-half_size,  half_size, 0],
    [ half_size,  half_size, 0],
    [ half_size, -half_size, 0],
    [-half_size, -half_size, 0]
], dtype=np.float32)


# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam")


def draw_text(frame, text, position, color=(255, 255, 255)):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA
    )


while True:
    ret, frame = cap.read()

    if not ret:
        break

    corners, ids, _ = detector.detectMarkers(frame)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for i in range(len(ids)):
            marker_2d_corners = corners[i][0].astype(np.float32)

            success, rvec, tvec = cv2.solvePnP(
                marker_3d_edges,
                marker_2d_corners,
                cam_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                continue

            # Draw 3D coordinate axes on marker
            cv2.drawFrameAxes(
                frame,
                cam_matrix,
                dist_coeffs,
                rvec,
                tvec,
                AXIS_LENGTH
            )

            marker_id = int(ids[i][0])

            # Flatten values for display
            tx, ty, tz = tvec.flatten()
            rx, ry, rz = rvec.flatten()

            # Text panel
            x0, y0 = 20, 40

            draw_text(frame, f"Marker ID: {marker_id}", (x0, y0), (0, 255, 255))
            draw_text(frame, "6-DoF Pose Output", (x0, y0 + 35), (255, 255, 255))

            draw_text(
                frame,
                f"tvec [m]  X:{tx:.3f}  Y:{ty:.3f}  Z:{tz:.3f}",
                (x0, y0 + 70),
                (0, 255, 0)
            )

            draw_text(
                frame,
                f"rvec [rad] X:{rx:.3f}  Y:{ry:.3f}  Z:{rz:.3f}",
                (x0, y0 + 105),
                (255, 200, 0)
            )

            draw_text(
                frame,
                "6-DoF = 3 translation axes + 3 rotation axes",
                (x0, y0 + 140),
                (255, 255, 255)
            )

    cv2.imshow("6-DoF Fiducial Marker Pose Estimation", frame)

    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()