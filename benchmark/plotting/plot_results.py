import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = "metrics_results"
OUTPUT_DIR = "plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)


summary_files = glob.glob(os.path.join(RESULTS_DIR, "*summary.csv"))

if not summary_files:
    raise RuntimeError("No summary CSV files found in metrics_results/")


rows = []

for file in summary_files:
    df = pd.read_csv(file)

    if len(df) == 0:
        continue

    row = df.iloc[0].to_dict()
    row["source_file"] = os.path.basename(file)
    rows.append(row)


data = pd.DataFrame(rows)

data["marker_type"] = data["marker_type"].astype(str)

def normalize_marker_type(row):
    text = (
        str(row.get("marker_type", "")) + " " +
        str(row.get("test_name", "")) + " " +
        str(row.get("source_file", ""))
    ).lower()

    if "cctag" in text or "cc_tag" in text or "cc tag" in text:
        return "CCTag"

    # IMPORTANT: check aruco before uco
    if "aruco" in text:
        return "ArUco"

    if "apriltag" in text or "april" in text:
        return "AprilTag"

    if "stag" in text and "apriltag" not in text:
        return "STag"

    if "chilitag" in text or "chilitags" in text:
        return "Chilitags"

    # More specific JuMarker UCO check
    if "jumarker" in text or "jumarker_uco" in text or "uco_dist" in text:
        return "JuMarker UCO"

    return str(row.get("marker_type", ""))
data["marker_type"] = data.apply(normalize_marker_type, axis=1)

print("\nMarker types after normalization:")
print(data["marker_type"].unique())

numeric_columns = [
    "detection_rate_percent",
    "pose_success_rate_percent",
    "mean_z_error_cm",
    "mean_distance_error_cm",
    "mean_reprojection_error_px",
    "z_jitter_cm",
    "distance_jitter_cm",
    "fps"
]

for col in numeric_columns:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

data = data.sort_values(["marker_type", "real_distance_m"])


print("\nLoaded summary data:")
print_columns = [
    "marker_type",
    "real_distance_m",
    "detection_rate_percent",
    "pose_success_rate_percent",
    "mean_z_error_cm",
    "mean_distance_error_cm",
    "mean_reprojection_error_px",
    "z_jitter_cm",
    "distance_jitter_cm",
    "fps",
    "source_file"
]

print_columns = [col for col in print_columns if col in data.columns]

print(data[print_columns])


MARKER_ORDER = ["ArUco", "AprilTag", "STag", "Chilitags", "CCTag", "JuMarker UCO"]


def plot_metric(metric_column, y_label, title, output_name):
    plt.figure(figsize=(9, 6))

    for marker in MARKER_ORDER:
        marker_data = data[data["marker_type"] == marker].copy()

        if marker_data.empty or metric_column not in marker_data.columns:
            continue

        marker_data[metric_column] = pd.to_numeric(
            marker_data[metric_column],
            errors="coerce"
        )

        marker_data = marker_data.dropna(subset=[metric_column])

        if marker_data.empty:
            continue

        plt.plot(
            marker_data["real_distance_m"],
            marker_data[metric_column],
            marker="o",
            label=marker
        )

    plt.xlabel("Distance (m)")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, output_name)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


plot_metric(
    "mean_z_error_cm",
    "Mean Z Error (cm)",
    "Mean Z Error vs Distance",
    "mean_z_error_vs_distance.png"
)
plot_metric(
    "mean_distance_error_cm",
    "Mean Straight-Line Distance Error (cm)",
    "Mean Straight-Line Distance Error vs Distance",
    "mean_distance_error_vs_distance.png"
)

plot_metric(
    "detection_rate_percent",
    "Detection Rate (%)",
    "Detection Rate vs Distance",
    "detection_rate_vs_distance.png"
)

plot_metric(
    "pose_success_rate_percent",
    "Pose Success Rate (%)",
    "Pose Success Rate vs Distance",
    "pose_success_rate_vs_distance.png"
)

plot_metric(
    "z_jitter_cm",
    "Z Jitter (cm)",
    "Z Jitter vs Distance",
    "z_jitter_vs_distance.png"
)
plot_metric(
    "distance_jitter_cm",
    "Distance Jitter (cm)",
    "Distance Jitter vs Distance",
    "distance_jitter_vs_distance.png"
)
plot_metric(
    "fps",
    "FPS",
    "Average FPS vs Distance",
    "fps_vs_distance.png"
)

plot_metric(
    "mean_reprojection_error_px",
    "Mean Reprojection Error (px)",
    "Reprojection Error vs Distance",
    "reprojection_error_vs_distance.png"
)

print("\nAll plots created successfully.")
print(f"Check the folder: {OUTPUT_DIR}/")