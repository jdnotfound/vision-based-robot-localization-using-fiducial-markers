import cv2
import time
import numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer

CAMERA_INDEX = 7
WIDTH = 640
HEIGHT = 480
PORT = 8081

MARKER_SIZE_METERS = 0.097
CALIBRATION_FILE = "calibration_data.npz"

STOP_DISTANCE_M = 0.35
CENTER_TOLERANCE_M = 0.04
YAW_TOLERANCE_DEG = 10.0

data = np.load(CALIBRATION_FILE)
cam_matrix = data["mtx"]
dist_coeffs = data["dist"]

s = MARKER_SIZE_METERS / 2

marker_3d_points = np.array([
    [-s,  s, 0],
    [ s,  s, 0],
    [ s, -s, 0],
    [-s, -s, 0]
], dtype=np.float32)

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5)
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, params)

cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

def get_yaw_deg(rvec):
    R, _ = cv2.Rodrigues(rvec)
    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return yaw

def get_camera_position(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    camera_pos = -R.T @ tvec
    return camera_pos

def decide_command(cam_x, distance, yaw_deg):
    if distance <= STOP_DISTANCE_M:
        return "STOP"

    if cam_x > CENTER_TOLERANCE_M:
        return "TURN_LEFT"

    if cam_x < -CENTER_TOLERANCE_M:
        return "TURN_RIGHT"

    if yaw_deg < -YAW_TOLERANCE_DEG:
        return "ROTATE_LEFT"

    if yaw_deg > YAW_TOLERANCE_DEG:
        return "ROTATE_RIGHT"

    return "FORWARD"

def put(frame, text, y, color=(255, 255, 255)):
    cv2.putText(
        frame,
        text,
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2
    )

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()

            html = """
            <html>
            <head>
                <title>STM32 AprilTag Pose Stream</title>
                <style>
                    body {
                        background: #111;
                        color: white;
                        text-align: center;
                        font-family: Arial;
                    }
                    img {
                        border: 3px solid #444;
                        width: 90%;
                        max-width: 800px;
                    }
                </style>
            </head>
            <body>
                <h2>STM32MP257F-DK AprilTag Pose Stream</h2>
                <img src="/stream.mjpg">
                <p>Live webcam + AprilTag pose output from ST board</p>
            </body>
            </html>
            """

            self.wfile.write(html.encode("utf-8"))

        elif self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header(
                "Content-Type",
                "multipart/x-mixed-replace; boundary=FRAME"
            )
            self.end_headers()

            prev_time = time.time()
            fps = 0.0

            while True:
                ret, frame = cap.read()

                if not ret:
                    continue

                now = time.time()
                dt = now - prev_time
                prev_time = now

                if dt > 0:
                    fps = 1.0 / dt

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, rejected = detector.detectMarkers(gray)

                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                    corner_points = corners[0].reshape((4, 2)).astype(np.float32)

                    success, rvec, tvec = cv2.solvePnP(
                        marker_3d_points,
                        corner_points,
                        cam_matrix,
                        dist_coeffs,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE
                    )

                    if success:
                        camera_pos = get_camera_position(rvec, tvec)

                        cam_x = camera_pos[0][0]
                        cam_z = camera_pos[2][0]

                        distance = float(np.linalg.norm(tvec))
                        yaw_deg = get_yaw_deg(rvec)

                        cmd = decide_command(cam_x, distance, yaw_deg)

                        put(frame, "TAG DETECTED", 35, (0, 255, 0))
                        put(frame, f"ID: {ids.flatten()[0]}", 65, (0, 255, 0))
                        put(frame, f"CamX: {cam_x:.3f} m", 95)
                        put(frame, f"CamZ: {cam_z:.3f} m", 125)
                        put(frame, f"Dist: {distance:.3f} m", 155)
                        put(frame, f"Yaw: {yaw_deg:.1f} deg", 185)
                        put(frame, f"CMD: {cmd}", 220, (0, 255, 255))
                        put(frame, f"FPS: {fps:.1f}", 250)

                    else:
                        put(frame, "TAG DETECTED, POSE FAILED", 35, (0, 0, 255))

                else:
                    put(frame, "NO TAG - SEARCH", 35, (0, 0, 255))
                    put(frame, f"FPS: {fps:.1f}", 65)

                put(frame, "STM32MP257F-DK | AprilTag 16h5 | OpenCV", HEIGHT - 20)

                ret, jpg = cv2.imencode(".jpg", frame)

                if not ret:
                    continue

                try:
                    self.wfile.write(b"--FRAME\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpg)))
                    self.end_headers()
                    self.wfile.write(jpg.tobytes())
                    self.wfile.write(b"\r\n")

                    time.sleep(0.03)

                except (BrokenPipeError, ConnectionResetError):
                    break

        else:
            self.send_error(404)

def main():
    if not cap.isOpened():
        print("ERROR: Camera not opened")
        return

    print("Camera opened")
    print("Open on laptop/phone/tablet:")
    print(f"http://10.133.71.10:{PORT}")
    print("Press Ctrl+C to stop")

    server = HTTPServer(("0.0.0.0", PORT), Handler)

    try:
        server.serve_forever()

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        cap.release()
        server.server_close()
        print("Stopped")

if __name__ == "__main__":
    main()