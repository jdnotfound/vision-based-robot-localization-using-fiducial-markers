#include <chilitags.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

int main()
{
    const double MARKER_SIZE_METERS = 0.097;
    const double HALF = MARKER_SIZE_METERS / 2.0;
    const double AXIS_LENGTH = MARKER_SIZE_METERS * 0.5;

    cv::Mat cameraMatrix, distCoeffs;

    cv::FileStorage fs("calibration_chilitags.yml", cv::FileStorage::READ);
    if (!fs.isOpened()) {
        cerr << "Could not open calibration_chilitags.yml" << endl;
        return 1;
    }

    fs["camera_matrix"] >> cameraMatrix;
    fs["dist_coeffs"] >> distCoeffs;
    fs.release();

    cout << "Camera matrix:\n" << cameraMatrix << endl;
    cout << "Dist coeffs:\n" << distCoeffs << endl;

    vector<cv::Point3f> objectPoints = {
        cv::Point3f(-HALF,  HALF, 0),  // top-left
        cv::Point3f( HALF,  HALF, 0),  // top-right
        cv::Point3f( HALF, -HALF, 0),  // bottom-right
        cv::Point3f(-HALF, -HALF, 0)   // bottom-left
    };

    cv::VideoCapture cap;
    cap.open(0, cv::CAP_V4L2);

    if (!cap.isOpened()) {
        cerr << "Could not open camera with V4L2" << endl;
        return 1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    chilitags::Chilitags detector;
    detector.setFilter(0, 0.0f); // raw detection, no smoothing

    cv::Mat frame;

    while (true) {
        cap >> frame;

        if (frame.empty()) {
            cerr << "Empty frame" << endl;
            break;
        }

        chilitags::TagCornerMap tags = detector.find(frame);

        for (const auto& tag : tags) {
            int id = tag.first;
            const cv::Mat_<cv::Point2f> cornersMat(tag.second);

            vector<cv::Point2f> imagePoints = {
                cornersMat(0),
                cornersMat(1),
                cornersMat(2),
                cornersMat(3)
            };

            cv::Mat rvec, tvec;
            bool ok = cv::solvePnP(
                objectPoints,
                imagePoints,
                cameraMatrix,
                distCoeffs,
                rvec,
                tvec,
                false,
                cv::SOLVEPNP_ITERATIVE
            );

            if (ok) {
                double x = tvec.at<double>(0);
                double y = tvec.at<double>(1);
                double z = tvec.at<double>(2);
                double distance = sqrt(x * x + y * y + z * z);

                for (int i = 0; i < 4; i++) {
                    cv::line(
                        frame,
                        imagePoints[i],
                        imagePoints[(i + 1) % 4],
                        cv::Scalar(0, 255, 0),
                        2
                    );
                }

                cv::drawFrameAxes(
                    frame,
                    cameraMatrix,
                    distCoeffs,
                    rvec,
                    tvec,
                    AXIS_LENGTH
                );

                cv::putText(
                    frame,
                    "ID: " + to_string(id),
                    imagePoints[0] + cv::Point2f(0, -10),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.7,
                    cv::Scalar(0, 255, 0),
                    2
                );

                cv::putText(
                    frame,
                    "Z: " + to_string(z).substr(0, 5) + " m",
                    cv::Point(30, 40),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.8,
                    cv::Scalar(0, 255, 0),
                    2
                );

                cv::putText(
                    frame,
                    "Distance: " + to_string(distance).substr(0, 5) + " m",
                    cv::Point(30, 75),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.8,
                    cv::Scalar(0, 255, 0),
                    2
                );
            }
        }

        cv::imshow("Chilitags Pose Live", frame);

        char key = (char)cv::waitKey(1);
        if (key == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}