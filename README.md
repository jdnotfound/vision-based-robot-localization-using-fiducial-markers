# Vision-Based Robot Localization using Fiducial Markers and OpenCV

This project implements and benchmarks fiducial marker based pose estimation for vision-based robot localization using OpenCV.

A camera detects printed markers, extracts image points, and estimates the marker pose relative to the camera using camera calibration and `solvePnP()`. The estimated pose can be used for robot localization, alignment, docking, and navigation correction.

## Project Goal

The aim is to compare different fiducial marker systems based on:

- Detection reliability
- Pose estimation success rate
- Z-distance accuracy
- Straight-line distance accuracy
- Reprojection error
- Z-distance jitter
- FPS / real-time performance
- Practical Linux and embedded deployment suitability

## Development Environment

The project was developed in an Ubuntu Linux virtual machine using Oracle VirtualBox.

Main environment details:

```text
Project directory: ~/cv-project
Python environment: ~/cv-project/venv
Camera device: /dev/video0
Camera backend: V4L2
IDE/tools: VS Code, nano, Linux terminal
```

The webcam is accessed using OpenCV:

```python
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
```

The Ubuntu VM was used for camera calibration, marker detection, pose estimation, CSV logging, and plot generation.

## Marker Systems

The project currently includes benchmark pipelines for:

| Marker | Implementation |
|---|---|
| ArUco | OpenCV ArUco dictionary |
| AprilTag | OpenCV AprilTag dictionary |
| STag | `stag-python` |
| Chilitags | External C++ library exposed through a Python wrapper |

Other marker systems such as TopoTag and WhyCode/WhyCon were explored, but were not included in the final clean benchmark due to integration complexity and workflow mismatch.

## Pose Estimation Pipeline

```text
Camera frame
→ Marker detection
→ 2D corner extraction
→ Camera calibration + marker size
→ solvePnP()
→ rvec / tvec
→ distance, stability, reprojection, FPS metrics
```

This project uses classical computer vision and geometric pose estimation, not deep learning.

## Calibration and Marker Size

All marker systems use the same calibration file:

```text
calibration_data.npz
```

All printed markers were tested using the same physical marker size:

```text
0.097 m
```

Used in code as:

```python
MARKER_SIZE_METERS = 0.097
```

Important note:

```text
Camera calibration resolution and capture resolution must match.
```

Using a different capture resolution from the calibration resolution can cause incorrect distance estimation.

## Benchmark Methodology

Each accepted run uses:

- Same webcam
- Same calibration file
- Same marker size
- Same indoor setup
- Same approximate lighting
- 0-degree front-facing marker alignment
- 500 frames per test
- Test distances: 0.5 m, 1.0 m, 1.5 m, 2.0 m, 2.5 m

CSV files were checked before plotting to avoid incorrect ground-truth distance labels or duplicate runs.

## Main Scripts

| File | Purpose |
|---|---|
| `aruco_pose.py` | Live ArUco pose estimation |
| `apriltag_pose.py` | Live AprilTag pose estimation |
| `stag_pose.py` | Live STag pose estimation |
| `chilitags_pose_live_py.py` | Live Chilitags pose estimation |
| `aruco_pose_metrics.py` | ArUco benchmark metrics |
| `apriltag_pose_metrics.py` | AprilTag benchmark metrics |
| `stag_pose_metrics.py` | STag benchmark metrics |
| `chilitags_pose_metrics.py` | Chilitags benchmark metrics |
| `chilitags_py.cpp` | Minimal Python wrapper for Chilitags detection |
| `camera_alignment_lines.py` | Camera alignment helper |
| `plot_results.py` | Generates main comparison plots |
| `plot_apriltag_datadensity_effect.py` | Generates AprilTag data-density plots |

## Results

Benchmark CSV files are stored in:

```text
metrics_results/
```

Generated plots are stored in:

```text
plots/
```

The generated plots compare marker performance across distance for:

- Detection rate
- Pose success rate
- Mean Z error
- Reprojection error
- Z jitter

The current comparison includes ArUco, AprilTag, STag, and Chilitags.

## AprilTag Data-Density Study

The repository also contains AprilTag 16h5 and 36h11 material to observe how marker data density affects pose estimation when the physical marker size is fixed.

Relevant folders:

```text
apriltag_datadensities/
plots_apriltag_datadensity_effect/
```

## Repository Structure

```text
.
├── aruco_pose.py
├── apriltag_pose.py
├── stag_pose.py
├── chilitags_pose_live_py.py
├── aruco_pose_metrics.py
├── apriltag_pose_metrics.py
├── stag_pose_metrics.py
├── chilitags_pose_metrics.py
├── chilitags_py.cpp
├── camera_alignment_lines.py
├── calibration_data.npz
├── chilitag_markers/
├── metrics_results/
├── plots/
├── apriltag_datadensities/
├── plots_apriltag_datadensity_effect/
├── plot_results.py
├── plot_apriltag_datadensity_effect.py
└── requirements.txt
```

Local folders such as `venv/`, `external/`, compiled binaries, and `.so` files are ignored and should be rebuilt locally if needed.

## Future Work

- Test longer distances if space allows
- Test yaw angles such as 15°, 30°, 45°, and 60°
- Test occlusion, blur, and lighting variation
- Use recorded videos for repeatable testing
- Deploy the selected pipeline on an ST Linux MPU board
- Optimize FPS for embedded use
- Test marker-based alignment or docking on a small robot
