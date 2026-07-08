import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


# Folder containing both 36h11 and 16h5 AprilTag CSV files
RESULTS_DIR = "apriltag_datadensities"

# New folder for data-density comparison plots
OUTPUT_DIR = "plots_apriltag_datadensity_effect"

os.makedirs(OUTPUT_DIR, exist_ok=True)


summary_files = glob.glob(os.path.join(RESULTS_DIR, "*summary.csv"))

if not summary_files:
    raise RuntimeError(f"No summary CSV files found in {RESULTS_DIR}/")


rows = []

for file in summary_files:
    df = pd.read_csv(file)

    if len(df) == 0:
        continue

    row = df.iloc[0].to_dict()
    source_file = os.path.basename(file)
    row["source_file"] = source_file

    # Identify AprilTag family using file date
    # 20260618 = old 36h11 runs
    # 20260620 = new 16h5 runs
    if "20260618" in source_file:
        row["apriltag_family"] = "AprilTag 36h11"
        row["data_density"] = "Higher data density"
    elif "20260620" in source_file:
        row["apriltag_family"] = "AprilTag 16h5"
        row["data_density"] = "Lower data density"
    else:
        print(f"Skipping unknown file date: {source_file}")
        continue

    rows.append(row)


data = pd.DataFrame(rows)

# Convert important numeric columns
numeric_columns = [
    "real_distance_m",
    "detection_rate_percent",
    "pose_success_rate_percent",
    "mean_z_error_cm",
    "mean_reprojection_error_px",
    "z_jitter_cm"
]

for col in numeric_columns:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")


data = data.sort_values(["apriltag_family", "real_distance_m"])


print("\nLoaded AprilTag data-density comparison data:")
print(data[[
    "apriltag_family",
    "data_density",
    "real_distance_m",
    "detection_rate_percent",
    "pose_success_rate_percent",
    "mean_z_error_cm",
    "mean_reprojection_error_px",
    "z_jitter_cm",
    "source_file"
]])


# Save combined summary CSV for reference/report
combined_csv_path = os.path.join(OUTPUT_DIR, "combined_apriltag_datadensity_summary.csv")
data.to_csv(combined_csv_path, index=False)
print(f"\nSaved combined CSV: {combined_csv_path}")


def plot_metric(metric_column, y_label, title, output_name):
    plt.figure(figsize=(9, 6))

    for family in ["AprilTag 36h11", "AprilTag 16h5"]:
        family_data = data[data["apriltag_family"] == family]

        if family_data.empty:
            continue

        plt.plot(
            family_data["real_distance_m"],
            family_data[metric_column],
            marker="o",
            label=family
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
    "AprilTag Data Density Effect: Mean Z Error vs Distance",
    "apriltag_datadensity_mean_z_error_vs_distance.png"
)

plot_metric(
    "detection_rate_percent",
    "Detection Rate (%)",
    "AprilTag Data Density Effect: Detection Rate vs Distance",
    "apriltag_datadensity_detection_rate_vs_distance.png"
)

plot_metric(
    "pose_success_rate_percent",
    "Pose Success Rate (%)",
    "AprilTag Data Density Effect: Pose Success Rate vs Distance",
    "apriltag_datadensity_pose_success_rate_vs_distance.png"
)

plot_metric(
    "z_jitter_cm",
    "Z Jitter (cm)",
    "AprilTag Data Density Effect: Z Jitter vs Distance",
    "apriltag_datadensity_z_jitter_vs_distance.png"
)

plot_metric(
    "mean_reprojection_error_px",
    "Mean Reprojection Error (px)",
    "AprilTag Data Density Effect: Reprojection Error vs Distance",
    "apriltag_datadensity_reprojection_error_vs_distance.png"
)

print("\nAll AprilTag data-density plots created successfully.")
print(f"Check the folder: {OUTPUT_DIR}/")
