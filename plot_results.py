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

data["marker_type"] = data["marker_type"].replace({
    "apriltag_16h5": "AprilTag",
    "apriltag": "AprilTag",
    "aruco": "ArUco",
    "stag": "STag",
    "chilitag": "Chilitags",
    "chilitags": "Chilitags"
})

data["real_distance_m"] = pd.to_numeric(
    data["real_distance_m"],
    errors="coerce"
)

data = data.sort_values(["marker_type", "real_distance_m"])


print("\nLoaded summary data:")
print(data[[
    "marker_type",
    "real_distance_m",
    "detection_rate_percent",
    "pose_success_rate_percent",
    "mean_z_error_cm",
    "mean_reprojection_error_px",
    "z_jitter_cm",
    "source_file"
]])


MARKER_ORDER = ["ArUco", "AprilTag", "STag", "Chilitags"]


def plot_metric(metric_column, y_label, title, output_name):
    plt.figure(figsize=(9, 6))

    for marker in MARKER_ORDER:
        marker_data = data[data["marker_type"] == marker]

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
    "mean_reprojection_error_px",
    "Mean Reprojection Error (px)",
    "Reprojection Error vs Distance",
    "reprojection_error_vs_distance.png"
)

print("\nAll plots created successfully.")
print(f"Check the folder: {OUTPUT_DIR}/")