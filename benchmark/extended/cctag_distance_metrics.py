import csv
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np


TEST_NAME = "cctag_dist_2.5m_angle_0deg"

REAL_DISTANCE_METERS = 2.5
REAL_CCTAG_DIAMETER_METERS = 0.096
EXPECTED_ID = 0
MAX_FRAMES = 500

CAMERA_ID = 0

SAVE_DIR = "metrics_results"
CALIBRATION_FILE = "calibration_data.npz"


PROJECT_ROOT = Path(__file__).resolve().parent
CCTAG_ROOT = PROJECT_ROOT / "external" / "CCTag"
CCTAG_EXE = CCTAG_ROOT / "build-cpu-isolated" / "Linux-x86_64" / "detection"
CCTAG_LIB_DIR = CCTAG_ROOT / "build-cpu-isolated" / "Linux-x86_64"


if not CCTAG_EXE.exists():
    raise RuntimeError(f"CCTag detector not found: {CCTAG_EXE}")

try:
    with np.load(CALIBRATION_FILE) as data:
        cam_matrix = data["mtx"]
except FileNotFoundError:
    raise RuntimeError("Please run camera calibration first and save calibration_data.npz")

fx = float(cam_matrix[0, 0])
fy = float(cam_matrix[1, 1])
cx = float(cam_matrix[0, 2])
cy = float(cam_matrix[1, 2])


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


def safe_mean(values):
    if len(values) == 0:
        return None
    return float(np.mean(values))


def safe_std(values):
    if len(values) == 0:
        return None
    return float(np.std(values))


frame_re = re.compile(r"Id\s*:\s*#frame\s+(\d+)")
candidate_re = re.compile(r"Detected\s+(\d+)\s+candidates")
time_re = re.compile(r"Total time:\s+([0-9.]+)s wall")


cmd = [
    "stdbuf", "-oL", "-eL",
    str(CCTAG_EXE),
    "-n", "3",
    "-i", str(CAMERA_ID)
]

env = os.environ.copy()
env["LD_LIBRARY_PATH"] = f"{CCTAG_LIB_DIR}:{env.get('LD_LIBRARY_PATH', '')}"


print("====================================")
print("CCTAG DISTANCE ACCURACY TEST")
print("====================================")
print(f"Test name: {TEST_NAME}")
print(f"Real distance: {REAL_DISTANCE_METERS} m")
print(f"Real CCTag diameter: {REAL_CCTAG_DIAMETER_METERS} m")
print(f"Expected ID: {EXPECTED_ID}")
print(f"Max frames: {MAX_FRAMES}")
print("Press Ctrl+C to stop early.")
print("====================================")


process = subprocess.Popen(
    cmd,
    cwd=str(CCTAG_ROOT),
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    env=env
)


per_frame_rows = []
current_row = None

z_errors_m = []
distance_errors_m = []
estimated_z_values_m = []
estimated_distance_values_m = []
diameter_values_px = []
ellipse_a_values_px = []
ellipse_b_values_px = []
total_times_ms = []

script_start = time.perf_counter()


try:
    for line in process.stdout:
        line = line.strip()

        m = frame_re.search(line)
        if m:
            if len(per_frame_rows) >= MAX_FRAMES:
                break

            frame_no = int(m.group(1))

            current_row = {
                "frame": frame_no,
                "detected": 0,
                "pose_success": 0,
                "marker_id": None,
                "status": None,
                "real_distance_m": REAL_DISTANCE_METERS,
                "real_cctag_diameter_m": REAL_CCTAG_DIAMETER_METERS,
                "center_x_px": None,
                "center_y_px": None,
                "ellipse_a_px": None,
                "ellipse_b_px": None,
                "ellipse_angle_rad": None,
                "apparent_diameter_px": None,
                "estimated_x_m": None,
                "estimated_y_m": None,
                "estimated_z_m": None,
                "estimated_distance_m": None,
                "z_error_m": None,
                "distance_error_m": None,
                "reprojection_error_px": None,
                "candidate_count": None,
                "total_time_ms": None
            }

            per_frame_rows.append(current_row)
            continue

        if current_row is None:
            continue

        cm = candidate_re.search(line)
        if cm:
            current_row["candidate_count"] = int(cm.group(1))
            continue

        tm = time_re.search(line)
        if tm:
            total_time_ms = float(tm.group(1)) * 1000.0
            current_row["total_time_ms"] = total_time_ms
            total_times_ms.append(total_time_ms)
            continue

        parts = line.split()

        # Expected patched CCTag row:
        # x y id status ellipse_a ellipse_b ellipse_angle apparent_diameter_px
        if len(parts) != 8:
            continue

        try:
            x_px = float(parts[0])
            y_px = float(parts[1])
            marker_id = int(parts[2])
            status = int(parts[3])
            ellipse_a_px = float(parts[4])
            ellipse_b_px = float(parts[5])
            ellipse_angle_rad = float(parts[6])
            apparent_diameter_px = float(parts[7])
        except ValueError:
            continue

        if marker_id != EXPECTED_ID or status != 1:
            continue

        estimated_z_m = fx * REAL_CCTAG_DIAMETER_METERS / apparent_diameter_px

        estimated_x_m = (x_px - cx) * estimated_z_m / fx
        estimated_y_m = (y_px - cy) * estimated_z_m / fy

        estimated_distance_m = float(
            np.sqrt(
                estimated_x_m ** 2 +
                estimated_y_m ** 2 +
                estimated_z_m ** 2
            )
        )

        z_error_m = abs(estimated_z_m - REAL_DISTANCE_METERS)
        distance_error_m = abs(estimated_distance_m - REAL_DISTANCE_METERS)

        current_row.update({
            "detected": 1,
            "pose_success": 1,
            "marker_id": marker_id,
            "status": status,
            "center_x_px": x_px,
            "center_y_px": y_px,
            "ellipse_a_px": ellipse_a_px,
            "ellipse_b_px": ellipse_b_px,
            "ellipse_angle_rad": ellipse_angle_rad,
            "apparent_diameter_px": apparent_diameter_px,
            "estimated_x_m": estimated_x_m,
            "estimated_y_m": estimated_y_m,
            "estimated_z_m": estimated_z_m,
            "estimated_distance_m": estimated_distance_m,
            "z_error_m": z_error_m,
            "distance_error_m": distance_error_m,
            "reprojection_error_px": None
        })

        estimated_z_values_m.append(estimated_z_m)
        estimated_distance_values_m.append(estimated_distance_m)
        z_errors_m.append(z_error_m)
        distance_errors_m.append(distance_error_m)
        diameter_values_px.append(apparent_diameter_px)
        ellipse_a_values_px.append(ellipse_a_px)
        ellipse_b_values_px.append(ellipse_b_px)

        detection_rate_live = len(estimated_z_values_m) / len(per_frame_rows) * 100

        print(
            f"Frame {len(per_frame_rows)}/{MAX_FRAMES} | "
            f"Z {estimated_z_m:.3f} m | "
            f"Err {z_error_m * 100:.2f} cm | "
            f"Detection {detection_rate_live:.1f}%"
        )

except KeyboardInterrupt:
    print("\nStopped early by user.")

finally:
    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()


total_frames = len(per_frame_rows)
detected_frames = sum(row["detected"] for row in per_frame_rows)
pose_success_frames = sum(row["pose_success"] for row in per_frame_rows)

correct_id_frames = detected_frames
wrong_id_frames = total_frames - detected_frames

detection_rate_percent = detected_frames / total_frames * 100 if total_frames > 0 else 0
pose_success_rate_percent = pose_success_frames / total_frames * 100 if total_frames > 0 else 0

mean_z_error_m = safe_mean(z_errors_m)
mean_distance_error_m = safe_mean(distance_errors_m)
mean_estimated_z_m = safe_mean(estimated_z_values_m)
mean_estimated_distance_m = safe_mean(estimated_distance_values_m)

std_z_m = safe_std(estimated_z_values_m)
std_distance_m = safe_std(estimated_distance_values_m)

avg_total_time_ms = safe_mean(total_times_ms)
fps = 1000 / avg_total_time_ms if avg_total_time_ms and avg_total_time_ms > 0 else None

elapsed_script_time_s = time.perf_counter() - script_start


summary = {
    "test_name": TEST_NAME,
    "marker_type": "cctag_3rings_distance_only",
    "real_distance_m": REAL_DISTANCE_METERS,
    "marker_size_m": REAL_CCTAG_DIAMETER_METERS,
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
    "mean_distance_error_cm": mean_distance_error_m * 100 if mean_distance_error_m is not None else None,
    "z_jitter_cm": std_z_m * 100 if std_z_m is not None else None,
    "distance_jitter_cm": std_distance_m * 100 if std_distance_m is not None else None,
    "mean_reprojection_error_px": None,
    "avg_detection_time_ms": avg_total_time_ms,
    "avg_pose_time_ms": None,
    "avg_total_time_ms": avg_total_time_ms,
    "fps": fps,
    "mean_apparent_diameter_px": safe_mean(diameter_values_px),
    "mean_ellipse_a_px": safe_mean(ellipse_a_values_px),
    "mean_ellipse_b_px": safe_mean(ellipse_b_values_px),
    "fx_px": fx,
    "fy_px": fy,
    "cx_px": cx,
    "cy_px": cy,
    "elapsed_script_time_s": elapsed_script_time_s
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
print("CCTAG TEST COMPLETE")
print("====================================")
print(f"Test name: {TEST_NAME}")
print(f"Total frames: {total_frames}")
print(f"Detected frames: {detected_frames}")
print(f"Detection rate: {detection_rate_percent:.2f}%")
print(f"Distance success frames: {pose_success_frames}")
print(f"Distance success rate: {pose_success_rate_percent:.2f}%")

if mean_estimated_z_m is not None:
    print(f"Mean estimated Z: {mean_estimated_z_m:.4f} m")

if mean_z_error_m is not None:
    print(f"Mean Z error: {mean_z_error_m * 100:.2f} cm")

if mean_distance_error_m is not None:
    print(f"Mean straight-line distance error: {mean_distance_error_m * 100:.2f} cm")

if std_z_m is not None:
    print(f"Z jitter: {std_z_m * 100:.2f} cm")

if fps is not None:
    print(f"Average FPS: {fps:.2f}")

print("\nSaved files:")
print(frames_csv_path)
print(summary_csv_path)
print("====================================")
