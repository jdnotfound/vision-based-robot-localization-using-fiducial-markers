import os
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np


CALIBRATION_FILE = "calibration_data.npz"

REAL_CCTAG_DIAMETER_METERS = 0.096
TARGET_ID = 0
CAMERA_ID = 0

PROJECT_ROOT = Path(__file__).resolve().parent
CCTAG_ROOT = PROJECT_ROOT / "external" / "CCTag"
CCTAG_EXE = CCTAG_ROOT / "build-cpu-isolated" / "Linux-x86_64" / "detection"
CCTAG_LIB_DIR = CCTAG_ROOT / "build-cpu-isolated" / "Linux-x86_64"

MAP_SIZE = 600
MAP_CENTER = MAP_SIZE // 2
PIXELS_PER_METER = 120

CAMERA_MAP_X = MAP_CENTER
CAMERA_MAP_Y = int(MAP_SIZE * 0.82)


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


frame_re = re.compile(r"Id\s*:\s*#frame\s+(\d+)")

cmd = [
    "stdbuf", "-oL", "-eL",
    str(CCTAG_EXE),
    "-n", "3",
    "-i", str(CAMERA_ID)
]

env = os.environ.copy()
env["LD_LIBRARY_PATH"] = f"{CCTAG_LIB_DIR}:{env.get('LD_LIBRARY_PATH', '')}"

print("====================================")
print("CCTAG LIVE DISTANCE ESTIMATION")
print("====================================")
print(f"Target ID: {TARGET_ID}")
print(f"Real CCTag diameter: {REAL_CCTAG_DIAMETER_METERS} m")
print(f"fx: {fx:.2f}, fy: {fy:.2f}, cx: {cx:.2f}, cy: {cy:.2f}")
print("Camera is drawn below the marker on the minimap.")
print("Press ESC in the map window or Ctrl+C in terminal to stop.")
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

current_frame = None

try:
    for line in process.stdout:
        line = line.strip()

        m = frame_re.search(line)
        if m:
            current_frame = int(m.group(1))
            continue

        parts = line.split()

        # Patched CCTag row format:
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

        if marker_id != TARGET_ID or status != 1:
            continue

        estimated_z_m = fx * REAL_CCTAG_DIAMETER_METERS / apparent_diameter_px

        estimated_x_m = (x_px - cx) * estimated_z_m / fx
        estimated_y_m = (y_px - cy) * estimated_z_m / fy

        distance_m = float(
            np.sqrt(
                estimated_x_m ** 2 +
                estimated_y_m ** 2 +
                estimated_z_m ** 2
            )
        )

        print(
            f"Frame {current_frame} | "
            f"ID {marker_id} | "
            f"X {estimated_x_m:.3f} m | "
            f"Y {estimated_y_m:.3f} m | "
            f"Z {estimated_z_m:.3f} m | "
            f"Dist {distance_m:.3f} m | "
            f"diam {apparent_diameter_px:.1f}px"
        )

        map_img = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)

        # Draw forward direction line from camera upward
        cv2.line(
            map_img,
            (CAMERA_MAP_X, CAMERA_MAP_Y),
            (CAMERA_MAP_X, 0),
            (80, 80, 80),
            1
        )

        # Draw lateral reference line through camera
        cv2.line(
            map_img,
            (0, CAMERA_MAP_Y),
            (MAP_SIZE, CAMERA_MAP_Y),
            (80, 80, 80),
            1
        )

        # Draw camera near bottom
        cv2.circle(
            map_img,
            (CAMERA_MAP_X, CAMERA_MAP_Y),
            9,
            (0, 255, 0),
            -1
        )

        cv2.putText(
            map_img,
            "Camera",
            (CAMERA_MAP_X + 12, CAMERA_MAP_Y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2
        )

        cv2.putText(
            map_img,
            "Forward +Z",
            (CAMERA_MAP_X + 12, CAMERA_MAP_Y - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 180, 180),
            1
        )

        # Marker appears above camera because it is in front of the camera
        marker_map_x = int(CAMERA_MAP_X + estimated_x_m * PIXELS_PER_METER)
        marker_map_y = int(CAMERA_MAP_Y - estimated_z_m * PIXELS_PER_METER)

        if 0 <= marker_map_x < MAP_SIZE and 0 <= marker_map_y < MAP_SIZE:
            cv2.circle(
                map_img,
                (marker_map_x, marker_map_y),
                9,
                (0, 255, 255),
                -1
            )

            cv2.line(
                map_img,
                (CAMERA_MAP_X, CAMERA_MAP_Y),
                (marker_map_x, marker_map_y),
                (255, 255, 0),
                2
            )

            cv2.putText(
                map_img,
                f"CCTag ID {marker_id}",
                (marker_map_x + 12, marker_map_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1
            )
        else:
            cv2.putText(
                map_img,
                "Marker outside map scale",
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        cv2.putText(
            map_img,
            f"Frame: {current_frame}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            map_img,
            f"Z: {estimated_z_m:.3f} m",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        cv2.putText(
            map_img,
            f"Dist: {distance_m:.3f} m",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2
        )

        cv2.putText(
            map_img,
            f"Diameter: {apparent_diameter_px:.1f} px",
            (20, MAP_SIZE - 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

        cv2.putText(
            map_img,
            f"Ellipse a:{ellipse_a_px:.1f} b:{ellipse_b_px:.1f}",
            (20, MAP_SIZE - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

        cv2.imshow("CCTag Top Down Distance Map", map_img)

        if cv2.waitKey(1) == 27:
            break

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    process.terminate()
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()

    cv2.destroyAllWindows()
    print("CCTag live script stopped.")
