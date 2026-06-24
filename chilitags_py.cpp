#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <chilitags.hpp>
#include <opencv2/opencv.hpp>

#include <vector>
#include <stdexcept>

namespace py = pybind11;
using namespace std;

class Detector {
private:
    chilitags::Chilitags detector;

public:
    Detector() {
        //detector.setFilter(0, 0.0f); // raw detection, no smoothing
    }

    py::list detect(py::array_t<unsigned char, py::array::c_style | py::array::forcecast> image) {
        py::buffer_info buf = image.request();

        if (buf.ndim != 2 && buf.ndim != 3) {
            throw runtime_error("Input image must be grayscale HxW or color HxWx3 / HxWx4");
        }

        int rows = static_cast<int>(buf.shape[0]);
        int cols = static_cast<int>(buf.shape[1]);

        cv::Mat frame;

        if (buf.ndim == 2) {
            frame = cv::Mat(rows, cols, CV_8UC1, buf.ptr);
        }
        else {
            int channels = static_cast<int>(buf.shape[2]);

            if (channels == 3) {
                frame = cv::Mat(rows, cols, CV_8UC3, buf.ptr);
            }
            else if (channels == 4) {
                cv::Mat rgba(rows, cols, CV_8UC4, buf.ptr);
                cv::cvtColor(rgba, frame, cv::COLOR_RGBA2BGR);
            }
            else {
                throw runtime_error("Color image must have 3 or 4 channels");
            }
        }

        chilitags::TagCornerMap tags = detector.find(frame);

        py::list results;

        for (const auto& tag : tags) {
            int id = tag.first;
            const cv::Mat_<cv::Point2f> cornersMat(tag.second);

            py::list corners;
            for (int i = 0; i < 4; i++) {
                py::list point;
                point.append(cornersMat(i).x);
                point.append(cornersMat(i).y);
                corners.append(point);
            }

            py::dict detection;
            detection["id"] = id;
            detection["corners"] = corners;

            results.append(detection);
        }

        return results;
    }
};

PYBIND11_MODULE(chilitags_py, m) {
    m.doc() = "Minimal Python wrapper for Chilitags detection";

    py::class_<Detector>(m, "Detector")
        .def(py::init<>())
        .def("detect", &Detector::detect);
}
