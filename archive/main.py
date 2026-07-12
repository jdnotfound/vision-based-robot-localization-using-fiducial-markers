import cv2
import numpy as np
import time
import json
import argparse


MARKER_SIZE_METERS = 0.097


APRILTAG_DICTS = {
    "16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


def load_calibration(path):
    try:
        with np.load(path) as data:
            cam_matrix = data["mtx"]
            dist_coeffs = data["dist"]
        return cam_matrix, dist_coeffs
    except FileNotFoundError:
        raise RuntimeError(f"Calibration file not found: {path}")


def create_detector(tag_family):
    if tag_family not in APRILTAG_DICTS:
        raise RuntimeError(f"Unsupported AprilTag family: {tag_family}")

    apriltag_dict = cv2.aruco.getPredefinedDictionary(APRILTAG_DICTS[tag_family])
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(apriltag_dict, detector_params)

    return detector


def create_marker_3d_points(marker_size):
    half_size = marker_size / 2.0

    return np.array([
        [-half_size,  half_size, 0],
        [ half_size,  half_size, 0],
        [ half_size, -half_size, 0],
        [-half_size, -half_size, 0],
    ], dtype=np.float32)


def open_camera(camera_index, width=None, height=None):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)

    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)

    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    return cap


def estimate_pose(marker_2d_corners, marker_3d_points, cam_matrix, dist_coeffs):
    success, rvec, tvec = cv2.solvePnP(
        marker_3d_points,
        marker_2d_corners,
        cam_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    R, _ = cv2.Rodrigues(rvec)

    # Marker position relative to camera
    tag_x_m = float(tvec[0][0])
    tag_y_m = float(tvec[1][0])
    tag_z_m = float(tvec[2][0])

    distance_m = float(np.linalg.norm(tvec))

    # Camera position relative to marker
    camera_pos = -R.T @ tvec

    cam_x_m = float(camera_pos[0][0])
    cam_y_m = float(camera_pos[1][0])
    cam_z_m = float(camera_pos[2][0])

    R_cam = R.T

    yaw_rad = np.arctan2(
        R_cam[0, 2],
        R_cam[2, 2]
    )

    yaw_deg = float(np.degrees(yaw_rad))

    if yaw_deg > 180:
        yaw_deg -= 360

    return {
        "tag_x_m": tag_x_m,
        "tag_y_m": tag_y_m,
        "tag_z_m": tag_z_m,
        "distance_m": distance_m,
        "cam_x_m": cam_x_m,
        "cam_y_m": cam_y_m,
        "cam_z_m": cam_z_m,
        "yaw_deg": yaw_deg,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--calibration", default="calibration_data.npz")
    parser.add_argument("--tag-family", default="16h5")
    parser.add_argument("--marker-size", type=float, default=MARKER_SIZE_METERS)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--display", action="store_true")

    args = parser.parse_args()

    cam_matrix, dist_coeffs = load_calibration(args.calibration)
    detector = create_detector(args.tag_family)
    marker_3d_points = create_marker_3d_points(args.marker_size)

    cap = open_camera(args.camera, args.width, args.height)

    prev_time = time.time()

    print("AprilTag localization started", flush=True)
    print(f"Tag family: {args.tag_family}", flush=True)
    print(f"Marker size: {args.marker_size} m", flush=True)

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Camera frame read failed", flush=True)
                time.sleep(0.1)
                continue

            corners, ids, _ = detector.detectMarkers(frame)

            now = time.time()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            if ids is not None:
                for i in range(len(ids)):
                    marker_id = int(ids[i][0])
                    marker_2d_corners = corners[i][0].astype(np.float32)

                    pose = estimate_pose(
                        marker_2d_corners,
                        marker_3d_points,
                        cam_matrix,
                        dist_coeffs
                    )

                    if pose is None:
                        continue

                    output = {
                        "detected": True,
                        "id": marker_id,
                        "fps": round(fps, 2),
                        "tag_x_m": round(pose["tag_x_m"], 4),
                        "tag_y_m": round(pose["tag_y_m"], 4),
                        "tag_z_m": round(pose["tag_z_m"], 4),
                        "distance_m": round(pose["distance_m"], 4),
                        "cam_x_m": round(pose["cam_x_m"], 4),
                        "cam_y_m": round(pose["cam_y_m"], 4),
                        "cam_z_m": round(pose["cam_z_m"], 4),
                        "yaw_deg": round(pose["yaw_deg"], 2),
                    }

                    print(json.dumps(output), flush=True)

                    if args.display:
                        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                        cv2.drawFrameAxes(
                            frame,
                            cam_matrix,
                            dist_coeffs,
                            np.zeros((3, 1), dtype=np.float32),
                            np.array([
                                [pose["tag_x_m"]],
                                [pose["tag_y_m"]],
                                [pose["tag_z_m"]]
                            ], dtype=np.float32),
                            args.marker_size * 0.5
                        )

            else:
                output = {
                    "detected": False,
                    "fps": round(fps, 2)
                }

                print(json.dumps(output), flush=True)

            if args.display:
                cv2.imshow("AprilTag Localization", frame)

                if cv2.waitKey(1) == 27:
                    break

    except KeyboardInterrupt:
        print("Stopped by user", flush=True)

    finally:
        cap.release()

        if args.display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
