import cv2
import numpy as np
import stag



CALIBRATION_FILE = "calibration_data.npz"


MARKER_SIZE_METERS = 0.097


STAG_LIBRARY_HD = 21

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




detector_params = cv2.aruco.DetectorParameters()

opencv_detectors = [
    {
        "name": "ArUco 4x4_50",
        "dict": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        "color": (0, 255, 0)
    },
    {
        "name": "AprilTag 16h5",
        "dict": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16h5),
        "color": (255, 0, 0)
    },
    {
        "name": "AprilTag 36h11",
        "dict": cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11),
        "color": (0, 165, 255)
    }
]

for item in opencv_detectors:
    item["detector"] = cv2.aruco.ArucoDetector(item["dict"], detector_params)

def draw_3d_cube(frame, rvec, tvec, color):
    # Draws a 3D cube on top of the detected marker.

    s = MARKER_SIZE_METERS / 2

    
    cube_3d_points = np.array([
        [-s,  s, 0],
        [ s,  s, 0],
        [ s, -s, 0],
        [-s, -s, 0],

        [-s,  s, MARKER_SIZE_METERS],
        [ s,  s, MARKER_SIZE_METERS],
        [ s, -s, MARKER_SIZE_METERS],
        [-s, -s, MARKER_SIZE_METERS],
    ], dtype=np.float32)

    # Convert 3D cube points into 2D image points
    cube_2d_points, _ = cv2.projectPoints(
        cube_3d_points,
        rvec,
        tvec,
        cam_matrix,
        dist_coeffs
    )

    cube_2d_points = cube_2d_points.reshape(-1, 2).astype(int)

    # Bottom square
    for i in range(4):
        cv2.line(
            frame,
            tuple(cube_2d_points[i]),
            tuple(cube_2d_points[(i + 1) % 4]),
            color,
            2
        )

    # Top square
    for i in range(4, 8):
        cv2.line(
            frame,
            tuple(cube_2d_points[i]),
            tuple(cube_2d_points[4 + (i + 1 - 4) % 4]),
            color,
            2
        )

    # Vertical lines
    for i in range(4):
        cv2.line(
            frame,
            tuple(cube_2d_points[i]),
            tuple(cube_2d_points[i + 4]),
            color,
            2
        )

def draw_marker_pose(frame, marker_2d_edges, marker_name, marker_id, color):


    marker_2d_edges = np.array(marker_2d_edges, dtype=np.float32).reshape(4, 2)

    success, rvec, tvec = cv2.solvePnP(
        marker_3d_edges,
        marker_2d_edges,
        cam_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return False


    corners_int = marker_2d_edges.astype(int)
    cv2.polylines(frame, [corners_int], True, color, 2)

    draw_3d_cube(frame, rvec, tvec, color)


 


    z_distance_m = float(tvec[2][0])
    straight_distance_m = float(np.linalg.norm(tvec))


    x, y = corners_int[0]
    label = f"{marker_name} ID:{marker_id} Z:{z_distance_m:.2f}m D:{straight_distance_m:.2f}m"

    cv2.putText(
        frame,
        label,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        2
    )

    return True



cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check VirtualBox webcam passthrough and /dev/video0.")


print("Combined marker detector started.")
print("Detecting: ArUco 4x4_50, AprilTag 16h5, AprilTag 36h11, STag HD21")
print("Press 'q' to quit.")


while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    counts = {
        "ArUco 4x4_50": 0,
        "AprilTag 16h5": 0,
        
        "STag HD21": 0
    }

    

    for item in opencv_detectors:
        marker_name = item["name"]
        detector = item["detector"]
        color = item["color"]

        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None:
            for i in range(len(ids)):
                marker_id = int(ids[i][0])
                marker_2d_edges = corners[i].reshape(4, 2)

                pose_ok = draw_marker_pose(
                    frame,
                    marker_2d_edges,
                    marker_name,
                    marker_id,
                    color
                )

                if pose_ok:
                    counts[marker_name] += 1

    
    try:
        stag_corners, stag_ids, rejected_corners = stag.detectMarkers(frame, STAG_LIBRARY_HD)

        if stag_ids is not None and len(stag_ids) > 0:
            for i in range(len(stag_ids)):
                # STag IDs may come as nested arrays, so flatten safely
                marker_id = int(np.array(stag_ids[i]).flatten()[0])
                marker_2d_edges = np.array(stag_corners[i], dtype=np.float32).reshape(4, 2)

                pose_ok = draw_marker_pose(
                    frame,
                    marker_2d_edges,
                    "STag HD21",
                    marker_id,
                    (255, 0, 255)
                )

                if pose_ok:
                    counts["STag HD21"] += 1

    except Exception as e:
        cv2.putText(
            frame,
            f"STag error: {str(e)[:60]}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )

    
    
    y0 = 25
    for idx, (name, count) in enumerate(counts.items()):
        cv2.putText(
            frame,
            f"{name}: {count}",
            (20, y0 + idx * 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    cv2.imshow("All Marker Detector - ArUco / AprilTag / STag", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
