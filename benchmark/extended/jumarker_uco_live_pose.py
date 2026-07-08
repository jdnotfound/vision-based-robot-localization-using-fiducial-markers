import os
import re
import signal
import subprocess


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

CAMERA_ID = 0
MARKER_BITS = 3
MARKER_TYPE = "UCO"
EXPECTED_ID = 1


POSE_RE = re.compile(
    r"P?OSE id:(?P<id>-?\d+)\s+"
    r"tx_m:(?P<tx>[-+0-9.eE]+)\s+"
    r"ty_m:(?P<ty>[-+0-9.eE]+)\s+"
    r"tz_m:(?P<tz>[-+0-9.eE]+)\s+"
    r"dist_m:(?P<dist>[-+0-9.eE]+)"
)

FRAME_RE = re.compile(
    r"Time detection:\s*(?P<time_ms>[-+0-9.eE]+)\s*milliseconds\s+"
    r"nmarkers:\s*(?P<nmarkers>\d+)"
)


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
check_file(SVG_FILE, "UCO SVG file")
check_file(CALIBRATION_FILE, "camera calibration file")


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


print("====================================")
print("JUMARKER UCO LIVE POSE")
print("====================================")
print(f"Expected ID: {EXPECTED_ID}")
print(f"Camera ID: {CAMERA_ID}")
print(f"Marker type: {MARKER_TYPE}")
print(f"Calibration: {CALIBRATION_FILE}")
print("Press ESC in JuMarker window or Ctrl+C here to stop.")
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


last_pose = None
frame_count = 0

try:
    while True:
        line = proc.stdout.readline()

        if line == "" and proc.poll() is not None:
            break

        if line == "":
            continue

        pose_match = POSE_RE.search(line)

        if pose_match:
            marker_id = int(pose_match.group("id"))

            last_pose = {
                "id": marker_id,
                "tx": float(pose_match.group("tx")),
                "ty": float(pose_match.group("ty")),
                "tz": float(pose_match.group("tz")),
                "dist": float(pose_match.group("dist")),
            }

            continue

        frame_match = FRAME_RE.search(line)

        if frame_match:
            frame_count += 1

            detection_time_ms = float(frame_match.group("time_ms"))
            nmarkers = int(frame_match.group("nmarkers"))

            if last_pose is not None:
                id_ok = last_pose["id"] == EXPECTED_ID

                print(
                    f"Frame {frame_count:04d} | "
                    f"ID: {last_pose['id']} "
                    f"{'OK' if id_ok else 'WRONG'} | "
                    f"X: {last_pose['tx']:.3f} m | "
                    f"Y: {last_pose['ty']:.3f} m | "
                    f"Z: {last_pose['tz']:.3f} m | "
                    f"Distance: {last_pose['dist']:.3f} m | "
                    f"nmarkers: {nmarkers} | "
                    f"time: {detection_time_ms:.2f} ms"
                )
            else:
                print(
                    f"Frame {frame_count:04d} | "
                    f"No pose | "
                    f"nmarkers: {nmarkers} | "
                    f"time: {detection_time_ms:.2f} ms"
                )

            last_pose = None

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    stop_process(proc)
    print("JuMarker live pose stopped.")
