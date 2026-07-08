import csv
import os
import re
import signal
import subprocess
import time
from datetime import datetime

import numpy as np


TEST_NAME = "jumarker_uco_dist_2.5m_angle_0deg"

REAL_DISTANCE_METERS = 2.5
MARKER_SIZE_METERS = 0.096
EXPECTED_ID = 1
MAX_FRAMES = 500

CAMERA_ID = 0

SAVE_DIR = "metrics_results"

JUMARKER_DIR = os.path.expanduser("~/cv-project/external/JuMarker")

JUMARKER_EXE = os.path.join(
    JUMARKER_DIR,
    "build-isolated",
    "utils",
    "jumarker_test"
)

SVG_FILE = os.path.join(
    JUMARKER_DIR,
    "marker_designs",
    "UCOMarker.svg"
)

CALIBRATION_FILE = os.path.join(
    JUMARKER_DIR,
    "utils",
    "my_camera_calibration.yml"
)

MARKER_BITS = 3
MARKER_TYPE = "UCO"


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

raw_log_path = os.path.join(
    SAVE_DIR,
    f"{TEST_NAME}_{timestamp}_raw_log.txt"
)


POSE_RE = re.compile(
    r"P?OSE id:(?P<id>-?\d+)\s+"
    r"tx_m:(?P<tx>[-+0-9.eE]+)\s+"
    r"ty_m:(?P<ty>[-+0-9.eE]+)\s+"
    r"tz_m:(?P<tz>[-+0-9.eE]+)\s+"
    r"dist_m:(?P<dist>[-+0-9.eE]+)"
)

FRAME_RE = re.compile(
    r"Time detection:\s*(?P<detection_time>[-+0-9.eE]+)\s*milliseconds\s+"
    r"nmarkers:\s*(?P<nmarkers>\d+)"
)


def safe_mean(values):
    if len(values) == 0:
        return None

    return float(np.mean(values))


def safe_std(values):
    if len(values) == 0:
        return None

    return float(np.std(values))


def check_file(path, name):
    if not os.path.exists(path):
        raise RuntimeError(f"{name} not found: {path}")


def stop_process(proc):
    if proc.poll() is not None:
        return

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=2)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


check_file(JUMARKER_EXE, "JuMarker executable")
check_file(SVG_FILE, "UCO marker SVG")
check_file(CALIBRATION_FILE, "JuMarker camera calibration file")


command = [
    "stdbuf",
    "-oL",
    JUMARKER_EXE,
    SVG_FILE,
    str(MARKER_BITS),
    "-c",
    CALIBRATION_FILE,
    "-v",
    f"live:{CAMERA_ID}",
    "-t",
    MARKER_TYPE,
]


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
print("JUMARKER UCO DISTANCE + ACCURACY TEST")
print("====================================")
print(f"Test name: {TEST_NAME}")
print(f"Real distance: {REAL_DISTANCE_METERS} m")
print(f"Marker size: {MARKER_SIZE_METERS} m")
print(f"Expected ID: {EXPECTED_ID}")
print(f"Max frames: {MAX_FRAMES}")
print(f"JuMarker executable: {JUMARKER_EXE}")
print(f"SVG file: {SVG_FILE}")
print(f"Calibration file: {CALIBRATION_FILE}")
print("Press ESC in the JuMarker window to stop early.")
print("====================================")


proc = subprocess.Popen(
    command,
    cwd=JUMARKER_DIR,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    preexec_fn=os.setsid
)

pending_poses = []

try:
    with open(raw_log_path, "w") as raw_log:
        while total_frames < MAX_FRAMES:
            line = proc.stdout.readline()

            if line == "" and proc.poll() is not None:
                break

            if line == "":
                continue

            print(line, end="")
            raw_log.write(line)
            raw_log.flush()

            pose_match = POSE_RE.search(line)

            if pose_match:
                pending_poses.append({
                    "marker_id": int(pose_match.group("id")),
                    "tvec_x": float(pose_match.group("tx")),
                    "tvec_y": float(pose_match.group("ty")),
                    "tvec_z": float(pose_match.group("tz")),
                    "estimated_distance": float(pose_match.group("dist"))
                })
                continue

            frame_match = FRAME_RE.search(line)

            if not frame_match:
                continue

            total_frames += 1

            detection_time_ms = float(frame_match.group("detection_time"))
            nmarkers = int(frame_match.group("nmarkers"))

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

            # JuMarker does pose inside the C++ executable.
            # Separate pose timing is not available from the current output.
            pose_time_ms = 0.0

            if nmarkers > 0:
                detected = True
                detected_frames += 1

                chosen_pose = None

                if EXPECTED_ID is not None:
                    for pose in pending_poses:
                        if pose["marker_id"] == EXPECTED_ID:
                            chosen_pose = pose
                            break

                    if chosen_pose is None and len(pending_poses) > 0:
                        wrong_id_frames += 1
                        chosen_pose = pending_poses[0]
                else:
                    if len(pending_poses) > 0:
                        chosen_pose = pending_poses[0]

                if chosen_pose is not None:
                    marker_id_value = chosen_pose["marker_id"]

                    if EXPECTED_ID is None or marker_id_value == EXPECTED_ID:
                        correct_id_frames += 1

                    pose_success = True
                    pose_success_frames += 1
                    pose_times_ms.append(pose_time_ms)

                    tvec_x = chosen_pose["tvec_x"]
                    tvec_y = chosen_pose["tvec_y"]
                    tvec_z = chosen_pose["tvec_z"]

                    estimated_distance_m = chosen_pose["estimated_distance"]

                    z_error_m = abs(tvec_z - REAL_DISTANCE_METERS)
                    distance_error_m = abs(estimated_distance_m - REAL_DISTANCE_METERS)

                    # Not available unless C++ also prints detected/projected corners.
                    reprojection_error_px = None

                    z_errors_m.append(z_error_m)
                    distance_errors_m.append(distance_error_m)
                    estimated_z_values_m.append(tvec_z)
                    estimated_distance_values_m.append(estimated_distance_m)

            # For this wrapper, C++ "Time detection" is the useful timing value.
            total_time_ms = detection_time_ms
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

            print(
                f"PY_FRAME {total_frames}/{MAX_FRAMES} | "
                f"Detection: {detection_rate_live:.1f}% | "
                f"Pose: {pose_rate_live:.1f}%"
            )

            pending_poses = []

finally:
    stop_process(proc)


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
    "marker_type": "jumarker_uco",
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
    "mean_distance_error_cm": mean_distance_error_m * 100 if mean_distance_error_m is not None else None,
    "z_jitter_cm": std_z_m * 100 if std_z_m is not None else None,
    "distance_jitter_cm": std_distance_m * 100 if std_distance_m is not None else None,
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
else:
    print("Mean reprojection error: not available for current JuMarker wrapper")

if std_z_m is not None:
    print(f"Z jitter: {std_z_m * 100:.2f} cm")

if fps is not None:
    print(f"Average FPS: {fps:.2f}")

print("\nSaved files:")
print(frames_csv_path)
print(summary_csv_path)
print(raw_log_path)
print("====================================")
