import cv2
import numpy as np
import csv
import os
import time
from datetime import datetime
import stag


TEST_NAME = "stag_dist_2.5m_angle_0deg"

REAL_DISTANCE_METERS = 2.5
MARKER_SIZE_METERS = 0.097
EXPECTED_ID = None
MAX_FRAMES = 500

CAMERA_ID = 0

SAVE_DIR = "metrics_results"
CALIBRATION_FILE = "calibration_data.npz"

STAG_LIBRARY_HD = 21


try:
    with np.load(CALIBRATION_FILE) as data:
        cam_matrix = data["mtx"]
        dist_coeffs = data["dist"]
except FileNotFoundError:
    raise RuntimeError("Please run camera calibration first and save calibration_data.npz")


# Marker corners in 3D
s = MARKER_SIZE_METERS / 2

marker_3d_edges = np.array([
    [-s,  s, 0],
    [ s,  s, 0],
    [ s, -s, 0],
    [-s, -s, 0]
], dtype=np.float32)


os.makedirs(SAVE_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

frames_csv_path = os.path.join(
    SAVE_DIR,
    f"{TEST_NAME}_{timestamp}_frames.csv"
)

summary_csv_path = os.path.join(
    SAVE_DIR,
    f"{TEST_NAME}_{timestamp}_summary.csv"
)


def compute_reprojection_error(image_corners, rvec, tvec):
    projected_points, _ = cv2.projectPoints(
        marker_3d_edges,
        rvec,
        tvec,
        cam_matrix,
        dist_coeffs
    )

    projected_points = projected_points.reshape(-1, 2)
    image_corners = image_corners.reshape(-1, 2)

    errors = np.linalg.norm(image_corners - projected_points, axis=1)
    return float(np.mean(errors))


def safe_mean(values):
    if len(values) == 0:
        return None

    return float(np.mean(values))


def safe_std(values):
    if len(values) == 0:
        return None

    return float(np.std(values))


def normalize_stag_corners(corner_data):
    arr = np.array(corner_data, dtype=np.float32)
    return arr.reshape(4, 2)


def normalize_stag_id(id_data):
    return int(np.array(id_data).flatten()[0])


cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)

if not cap.isOpened():
    raise RuntimeError("Could not open camera.")


actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Camera resolution: {actual_width} x {actual_height}")


total_frames = 0
detected_frames = 0
pose_success_frames = 0
correct_id_frames = 0
wrong_id_frames = 0

z_errors_m = []
distance_errors_m = []
estimated_z_values_m = []
estimated_distance_values_m = []
reprojection_errors_px = []

detection_times_ms = []
pose_times_ms = []
total_times_ms = []

per_frame_rows = []


print("====================================")
print("STAG DISTANCE + ACCURACY TEST")
print("====================================")
print(f"Test name: {TEST_NAME}")
print(f"Real distance: {REAL_DISTANCE_METERS} m")
print(f"Marker size: {MARKER_SIZE_METERS} m")
print(f"STag library HD: {STAG_LIBRARY_HD}")
print(f"Expected ID: {EXPECTED_ID}")
print(f"Max frames: {MAX_FRAMES}")
print("Press ESC to stop early.")
print("====================================")


while total_frames < MAX_FRAMES:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame.")
        break

    total_frames += 1
    frame_start_time = time.perf_counter()

    detection_start = time.perf_counter()

    try:
        corners, ids, rejected_corners = stag.detectMarkers(
            frame,
            STAG_LIBRARY_HD
        )
    except Exception as e:
        print(f"STag detection error on frame {total_frames}: {e}")
        corners, ids = None, None

    detection_end = time.perf_counter()

    detection_time_ms = (detection_end - detection_start) * 1000
    detection_times_ms.append(detection_time_ms)

    detected = False
    pose_success = False

    marker_id_value = None
    tvec_x = None
    tvec_y = None
    tvec_z = None
    estimated_distance_m = None
    z_error_m = None
    distance_error_m = None
    reprojection_error_px = None
    pose_time_ms = 0.0

    if ids is not None and len(ids) > 0:
        chosen_index = 0
        parsed_ids = [normalize_stag_id(x) for x in ids]

        if EXPECTED_ID is not None:
            if EXPECTED_ID in parsed_ids:
                chosen_index = parsed_ids.index(EXPECTED_ID)
            else:
                wrong_id_frames += 1
                chosen_index = 0

        marker_id_value = parsed_ids[chosen_index]

        if EXPECTED_ID is None or marker_id_value == EXPECTED_ID:
            correct_id_frames += 1

        detected = True
        detected_frames += 1

        marker_2d_corners = normalize_stag_corners(corners[chosen_index])

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
            f"STag ID: {marker_id_value}",
            tuple(pts[0]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        pose_start = time.perf_counter()

        success, rvec, tvec = cv2.solvePnP(
            marker_3d_edges,
            marker_2d_corners,
            cam_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        pose_end = time.perf_counter()
        pose_time_ms = (pose_end - pose_start) * 1000

        if success:
            pose_success = True
            pose_success_frames += 1
            pose_times_ms.append(pose_time_ms)

            tvec_x = float(tvec[0][0])
            tvec_y = float(tvec[1][0])
            tvec_z = float(tvec[2][0])

            estimated_distance_m = float(np.linalg.norm(tvec))

            z_error_m = abs(tvec_z - REAL_DISTANCE_METERS)
            distance_error_m = abs(estimated_distance_m - REAL_DISTANCE_METERS)

            reprojection_error_px = compute_reprojection_error(
                marker_2d_corners,
                rvec,
                tvec
            )

            z_errors_m.append(z_error_m)
            distance_errors_m.append(distance_error_m)
            estimated_z_values_m.append(tvec_z)
            estimated_distance_values_m.append(estimated_distance_m)
            reprojection_errors_px.append(reprojection_error_px)

            cv2.drawFrameAxes(
                frame,
                cam_matrix,
                dist_coeffs,
                rvec,
                tvec,
                MARKER_SIZE_METERS * 0.5
            )

            x_text = int(marker_2d_corners[0][0])
            y_text = int(marker_2d_corners[0][1]) - 10

            cv2.putText(
                frame,
                f"Z: {tvec_z:.3f} m",
                (x_text, y_text - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Z error: {z_error_m * 100:.2f} cm",
                (x_text, y_text - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Reproj: {reprojection_error_px:.2f} px",
                (x_text, y_text - 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

    frame_end_time = time.perf_counter()
    total_time_ms = (frame_end_time - frame_start_time) * 1000
    total_times_ms.append(total_time_ms)

    per_frame_rows.append({
        "frame": total_frames,
        "detected": int(detected),
        "pose_success": int(pose_success),
        "marker_id": marker_id_value,
        "real_distance_m": REAL_DISTANCE_METERS,
        "tvec_x_m": tvec_x,
        "tvec_y_m": tvec_y,
        "tvec_z_m": tvec_z,
        "estimated_distance_m": estimated_distance_m,
        "z_error_m": z_error_m,
        "distance_error_m": distance_error_m,
        "reprojection_error_px": reprojection_error_px,
        "detection_time_ms": detection_time_ms,
        "pose_time_ms": pose_time_ms,
        "total_time_ms": total_time_ms
    })

    detection_rate_live = detected_frames / total_frames * 100
    pose_rate_live = pose_success_frames / total_frames * 100

    cv2.putText(
        frame,
        f"Frame: {total_frames}/{MAX_FRAMES}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Detection: {detection_rate_live:.1f}% | Pose: {pose_rate_live:.1f}%",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("STag Distance Accuracy Metrics", frame)

    if cv2.waitKey(1) == 27:
        print("Stopped early by user.")
        break


cap.release()
cv2.destroyAllWindows()


detection_rate_percent = detected_frames / total_frames * 100 if total_frames > 0 else 0
pose_success_rate_percent = pose_success_frames / total_frames * 100 if total_frames > 0 else 0

mean_z_error_m = safe_mean(z_errors_m)
mean_distance_error_m = safe_mean(distance_errors_m)
mean_estimated_z_m = safe_mean(estimated_z_values_m)
mean_estimated_distance_m = safe_mean(estimated_distance_values_m)
mean_reprojection_error_px = safe_mean(reprojection_errors_px)

std_z_m = safe_std(estimated_z_values_m)
std_distance_m = safe_std(estimated_distance_values_m)

avg_detection_time_ms = safe_mean(detection_times_ms)
avg_pose_time_ms = safe_mean(pose_times_ms)
avg_total_time_ms = safe_mean(total_times_ms)

fps = 1000 / avg_total_time_ms if avg_total_time_ms and avg_total_time_ms > 0 else None


summary = {
    "test_name": TEST_NAME,
    "marker_type": "stag",
    "stag_library_hd": STAG_LIBRARY_HD,
    "real_distance_m": REAL_DISTANCE_METERS,
    "marker_size_m": MARKER_SIZE_METERS,
    "expected_id": EXPECTED_ID,
    "total_frames": total_frames,
    "detected_frames": detected_frames,
    "detection_rate_percent": detection_rate_percent,
    "pose_success_frames": pose_success_frames,
    "pose_success_rate_percent": pose_success_rate_percent,
    "correct_id_frames": correct_id_frames,
    "wrong_id_frames": wrong_id_frames,
    "mean_estimated_z_m": mean_estimated_z_m,
    "mean_estimated_distance_m": mean_estimated_distance_m,
    "mean_z_error_cm": mean_z_error_m * 100 if mean_z_error_m is not None else None,
    "mean_distance_error_cm": (
        mean_distance_error_m * 100 if mean_distance_error_m is not None else None
    ),
    "z_jitter_cm": std_z_m * 100 if std_z_m is not None else None,
    "distance_jitter_cm": (
        std_distance_m * 100 if std_distance_m is not None else None
    ),
    "mean_reprojection_error_px": mean_reprojection_error_px,
    "avg_detection_time_ms": avg_detection_time_ms,
    "avg_pose_time_ms": avg_pose_time_ms,
    "avg_total_time_ms": avg_total_time_ms,
    "fps": fps
}


if len(per_frame_rows) > 0:
    with open(frames_csv_path, "w", newline="") as f:
        fieldnames = list(per_frame_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_frame_rows)


with open(summary_csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
    writer.writeheader()
    writer.writerow(summary)


print("\n====================================")
print("TEST COMPLETE")
print("====================================")
print(f"Test name: {TEST_NAME}")
print(f"Total frames: {total_frames}")
print(f"Detected frames: {detected_frames}")
print(f"Detection rate: {detection_rate_percent:.2f}%")
print(f"Pose success frames: {pose_success_frames}")
print(f"Pose success rate: {pose_success_rate_percent:.2f}%")

if mean_estimated_z_m is not None:
    print(f"Mean estimated Z: {mean_estimated_z_m:.4f} m")

if mean_z_error_m is not None:
    print(f"Mean Z error: {mean_z_error_m * 100:.2f} cm")

if mean_distance_error_m is not None:
    print(f"Mean straight-line distance error: {mean_distance_error_m * 100:.2f} cm")

if mean_reprojection_error_px is not None:
    print(f"Mean reprojection error: {mean_reprojection_error_px:.3f} px")

if std_z_m is not None:
    print(f"Z jitter: {std_z_m * 100:.2f} cm")

if fps is not None:
    print(f"Average FPS: {fps:.2f}")

print("\nSaved files:")
print(frames_csv_path)
print(summary_csv_path)
print("====================================")