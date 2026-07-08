import cv2
import numpy as np
import stag


try:
    with np.load("calibration_data.npz") as data:
        cam_matrix = data["mtx"]
        dist_coeffs = data["dist"]
except FileNotFoundError:
    raise RuntimeError(
        "Please run camera calibration first and save 'calibration_data.npz'"
    )


MARKER_SIZE_METERS = 0.097
STAG_LIBRARY_HD = 21

half_size = MARKER_SIZE_METERS / 2

# Marker corners in 3D
marker_3d_edges = np.array([
    [-half_size,  half_size, 0],
    [ half_size,  half_size, 0],
    [ half_size, -half_size, 0],
    [-half_size, -half_size, 0]
], dtype=np.float32)


cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    raise RuntimeError("Could not open camera.")


MAP_SIZE = 600
MAP_CENTER = MAP_SIZE // 2
PIXELS_PER_CM = 3


while True:
    ret, frame = cap.read()

    if not ret:
        break

    corners, ids, rejected_corners = stag.detectMarkers(
        frame,
        STAG_LIBRARY_HD
    )

    map_img = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)

    cv2.line(
        map_img,
        (0, MAP_CENTER),
        (MAP_SIZE, MAP_CENTER),
        (100, 100, 100),
        1
    )

    cv2.line(
        map_img,
        (MAP_CENTER, 0),
        (MAP_CENTER, MAP_SIZE),
        (100, 100, 100),
        1
    )

    cv2.circle(
        map_img,
        (MAP_CENTER, MAP_CENTER),
        8,
        (0, 255, 0),
        -1
    )

    cv2.putText(
        map_img,
        "STag (0,0)",
        (MAP_CENTER + 10, MAP_CENTER - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1
    )

    if ids is not None and len(ids) > 0:
        for i in range(len(ids)):
            marker_2d_corners = np.array(
                corners[i],
                dtype=np.float32
            ).reshape(4, 2)

            marker_id = int(np.array(ids[i]).flatten()[0])

            pts = marker_2d_corners.astype(int)

            cv2.polylines(
                frame,
                [pts],
                isClosed=True,
                color=(0, 255, 0),
                thickness=2
            )

            cv2.putText(
                frame,
                f"STag ID: {marker_id}",
                tuple(pts[0]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            success, rvec, tvec = cv2.solvePnP(
                marker_3d_edges,
                marker_2d_corners,
                cam_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                continue

            cv2.drawFrameAxes(
                frame,
                cam_matrix,
                dist_coeffs,
                rvec,
                tvec,
                MARKER_SIZE_METERS * 0.5
            )

            distance_cm = np.linalg.norm(tvec) * 100

            R, _ = cv2.Rodrigues(rvec)

            # Camera position from the marker
            camera_pos = -R.T @ tvec

            cam_x = camera_pos[0][0] * 100
            cam_y = camera_pos[1][0] * 100
            cam_z = camera_pos[2][0] * 100

            R_cam = R.T

            yaw_rad = np.arctan2(
                R_cam[0, 2],
                R_cam[2, 2]
            )

            yaw_deg = np.degrees(yaw_rad) + 180

            if yaw_deg > 180:
                yaw_deg -= 360

            map_x = int(MAP_CENTER + cam_x * PIXELS_PER_CM)
            map_y = int(MAP_CENTER + cam_z * PIXELS_PER_CM)

            if 0 <= map_x < MAP_SIZE and 0 <= map_y < MAP_SIZE:
                cv2.circle(
                    map_img,
                    (map_x, map_y),
                    8,
                    (0, 255, 255),
                    -1
                )

                cv2.line(
                    map_img,
                    (MAP_CENTER, MAP_CENTER),
                    (map_x, map_y),
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    map_img,
                    f"Cam ({cam_x:.1f},{cam_z:.1f})",
                    (map_x + 10, map_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1
                )

            cv2.putText(
                map_img,
                f"Yaw {yaw_deg:.1f} deg",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

            x_text = int(marker_2d_corners[0][0])
            y_text = int(marker_2d_corners[0][1]) - 10

            cv2.putText(
                frame,
                f"Dist: {distance_cm:.1f} cm",
                (x_text, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Cam X:{cam_x:.1f}",
                (x_text, y_text - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Cam Y:{cam_y:.1f}",
                (x_text, y_text - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Cam Z:{cam_z:.1f}",
                (x_text, y_text - 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Yaw:{yaw_deg:.1f}",
                (x_text, y_text - 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )

    cv2.imshow("STag Pose Estimation", frame)
    cv2.imshow("Top Down Navigation Map", map_img)

    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()