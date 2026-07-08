/*
 * This file is part of the JuMarker library
 * Copyright (c) 2022 David Jurado Rodríguez, Rafael Muñoz Salinas,
 *                    Sergio Garrido Jurado, Rafael Medina Carnicer.
 *
 * JuMarker is published and distributed under the CC BY-NC-SA license.
 * jumarker is distributed in the hope that it will be useful for
 * non-commercial academic research, but WITHOUT ANY WARRANTY.
 *
 * You should have received a copy of the CC BY-NC-SA license along with
 * this program; if not, write to rmsalinas@uco.es.
 */

#include "cvdrawingutils.h"
#include "markerdetector.h"

class CmdLineParser
{
    int argc;
    char **argv;

  public:
    CmdLineParser(int _argc, char **_argv) : argc(_argc), argv(_argv)
    {
    }
    bool operator[](string param)
    {
        int idx = -1;
        for (int i = 0; i < argc && idx == -1; i++)
            if (string(argv[i]) == param)
                idx = i;
        return (idx != -1);
    }
    string operator()(string param, string defvalue = "-1")
    {
        int idx = -1;
        for (int i = 0; i < argc && idx == -1; i++)
            if (string(argv[i]) == param)
                idx = i;
        if (idx == -1)
            return defvalue;
        else
            return (argv[idx + 1]);
    }
};

TimerAvrg Fps;
//  Threshold Trackbar setting
int threshold_type = 1, Adaptive_Threshold_Value = 17, threshold_value = 200, Adaptive_Block_Size = 34, erosion_elem = 1,
    erosion_size = 1, dilation_elem = 1, dilation_size = 1;
int const max_elem = 2, max_kernel_size = 21, max_threshold_value = 255;
float resizeFactor, sizeMarker = 0.096f;

// Video setting
const cv::Size imageResolution(640, 480);
const bool saveVideo = false;
const double fpsOutVideo = 25, frameStart = 0;

// General variables
cv::VideoCapture TheVideoCapturer;
cv::VideoWriter outputVideo;
CameraParameters TheCameraParameters;
MarkerDetector MDetector("homography");
SVGData SVGFileData;
cv::Mat TheInputImage, TheInputCopyImage;
string TheInputVideo, cameraParametersFile, markerType = "building";

// Control variables
int idBitMarker, waitTime = 0, showAllContours = 0;
char key = 0;
bool isVideo = false, thresHoldingDebug = false, vumarkEnable = false;

// Resize image
inline cv::Mat resize(const cv::Mat &in, int width)
{
    if (in.size().width <= width)
        return in;
    float yf = float(width) / float(in.size().width);
    cv::Mat im2;
    cv::resize(in, im2, cv::Size(width, static_cast<int>(in.size().height * yf)));
    return im2;
}

inline cv::Mat resizeImageFactor(cv::Mat &in, float resizeFactor)
{
    if (fabs(1 - resizeFactor) < 1e-3)
        return in;
    float nc = float(in.cols) * resizeFactor;
    float nr = float(in.rows) * resizeFactor;
    cv::Mat imres;
    cv::resize(in, imres, cv::Size(nc, nr));
    return imres;
}

// Init params from marker detector
inline void initMarkerDetector(MarkerDetector &md)
{
    md.erosion_elem = erosion_elem;
    md.erosion_size = erosion_size;
    md.dilation_elem = dilation_elem;
    md.dilation_size = dilation_size;

    md.threshold_value = threshold_value;
    md.threshold_type = threshold_type;
    md.Adaptive_Threshold_Value = Adaptive_Threshold_Value;
    md.Adaptive_Block_Size = Adaptive_Block_Size;

    // Initialite SVG value
    md.SVGFileData = SVGFileData;
    md.markerbitsData = idBitMarker;
    md.enableTracking = false;
    md.vumarkerName = markerType;
    md.only_detect = 0;
    md.thresHoldingDebug = thresHoldingDebug;
    md.VuMark = vumarkEnable;
}

// Draw drawAllContourFiltered
inline void drawAllContourFiltered(cv::Mat &TheInputImage, const vector<vector<cv::Point>> &filterContour)
{
    int lineWidth = 3;
    // cv::Mat atom_image = cv::Mat::zeros( TheInputImage.rows, TheInputImage.cols, CV_8UC3 );
    for (int i = 0; i < filterContour.size(); i++)
    {
        for (int j = 0; j < filterContour[i].size(); j++)
        {
            if (j < filterContour[i].size() - 1)
                cv::line(TheInputImage, filterContour[i][j], filterContour[i][j + 1], cv::Scalar(0, 255, 0), lineWidth,
                         cv::LINE_8, 0);
            else
                cv::line(TheInputImage, filterContour[i][j], filterContour[i][0], cv::Scalar(0, 255, 0), lineWidth, cv::LINE_8,
                         0);

            cv::putText(TheInputImage, to_string(j), filterContour[i][j], cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255, 0, 255), 2,
                        cv::LINE_8, 0);
            cv::circle(TheInputImage, filterContour[i][j], lineWidth + 4, cv::Scalar(0, 0, 255), -1, cv::LINE_8, 0);
        }
    }

    // cv::imshow("argv[1]", resize(atom_image,1080));
}

void cvTackBarEvents(int pos, void *)
{
    (void)(pos);

    initMarkerDetector(MDetector);

    TheVideoCapturer >> TheInputImage;

    TheInputImage.copyTo(TheInputCopyImage);

    /// Check resize factor
    TheInputCopyImage = resizeImageFactor(TheInputCopyImage, resizeFactor);

    /// Detect and track marker
    Fps.start();
    MDetector.detectAndTrack(TheInputImage, TheCameraParameters, sizeMarker);
    Fps.stop();

    /// Draw informarker
    for (auto marker : MDetector.markersDetected)
        marker.model.pose->draw(TheInputCopyImage, SVGFileData, marker.id, TheCameraParameters);

    if (showAllContours)
        drawAllContourFiltered(TheInputCopyImage, MDetector.filterContours);

    if (MDetector.thresHoldingDebug)
        cv::imshow("Thresholding", resize(MDetector.TheInputImageThresholding, 1080));
    cv::imshow("InputImage", resize(TheInputCopyImage, 1080));
}

// Create trackbar in Thresholding Image
inline void createTrackbar()
{
    const string nameWindow = "Thresholding";
    /// Thresholding Trackbar
    cv::createTrackbar("Thresholding Type", nameWindow, &threshold_type, 1, cvTackBarEvents);
    cv::createTrackbar("Threshold Value", nameWindow, &threshold_value, max_threshold_value, cvTackBarEvents);
    cv::createTrackbar("Adaptative Threshold Value", nameWindow, &Adaptive_Threshold_Value, 40, cvTackBarEvents);
    cv::createTrackbar("Adaptative BlockSice", nameWindow, &Adaptive_Block_Size, 40, cvTackBarEvents);
    cv::createTrackbar("Show all contours", nameWindow, &showAllContours, 1, cvTackBarEvents);

    /// Erosion Trackbar
    cv::createTrackbar("Element: erotion \n 0: Rect \n 1: Cross \n 2: Ellipse", nameWindow, &erosion_elem, max_elem,
                       cvTackBarEvents);
    cv::createTrackbar("Kernel erotion size:\n 2n +1", nameWindow, &erosion_size, max_kernel_size, cvTackBarEvents);

    /// Dilation Trackbar
    cv::createTrackbar("Element: dilation \n 0: Rect \n 1: Cross \n 2: Ellipse", "Thresholding", &dilation_elem, max_elem,
                       cvTackBarEvents);
    cv::createTrackbar("Kernel dilation size:\n 2n +1", nameWindow, &dilation_size, max_kernel_size, cvTackBarEvents);
}

// Show information in the inputImage
void showInformation()
{
    // cv::putText(TheInputCopyImage, "RESOLUTION: [ " + to_string(TheInputCopyImage.cols) + " x "
    // +to_string(TheInputCopyImage.rows) + " ]" ,cv::Point(TheInputCopyImage.cols - 300,50),cv::FONT_HERSHEY_SIMPLEX,
    // 0.6f,cv::Scalar(125,255,255),2);
    cv::putText(TheInputCopyImage, "FPS: " + to_string(1. / Fps.getAvrg()), cv::Point(20, 50), cv::FONT_HERSHEY_SIMPLEX, 0.6f,
                cv::Scalar(125, 255, 255), 2);
    cv::putText(TheInputCopyImage, "MODE: " + MDetector.modeInfo, cv::Point(20, 90), cv::FONT_HERSHEY_SIMPLEX, 0.6f,
                cv::Scalar(125, 255, 255), 2);
    // cv::putText(TheInputCopyImage, "FRAME: " +
    // to_string((int)TheVideoCapturer.get(cv::CAP_PROP_POS_FRAMES)),cv::Point(20,90),cv::FONT_HERSHEY_SIMPLEX,
    // 0.6f,cv::Scalar(125,255,255),2); cv::putText(TheInputCopyImage, "TIMES CALLS DETECTED: " +
    // to_string(MDetector.timesCallsDetect),cv::Point(20,130),cv::FONT_HERSHEY_SIMPLEX, 0.6f,cv::Scalar(125,255,255),2);
    // cv::putText(TheInputCopyImage, "Path video file: " + TheInputVideo,cv::Point(20,TheInputCopyImage.rows -
    // 50),cv::FONT_HERSHEY_SIMPLEX, 0.6f,cv::Scalar(255,255,255),2); cv::putText(TheInputCopyImage, "Path calibration file: " +
    // cameraParametersFile,cv::Point(20,TheInputCopyImage.rows - 80),cv::FONT_HERSHEY_SIMPLEX, 0.6f,cv::Scalar(255,255,255),2);
}

// ----------------------------------------------------------------------------
//   Main
// ----------------------------------------------------------------------------

int main(int argc, char **argv)
{
    try
    {
        CmdLineParser cml(argc, argv);
        if (argc < 5 || cml["-h"])
        {
            cerr << "Usage: [path svg file] [bits used for check id marker] "
                    "[-c Camera parameters] [-v [in_image|video] ]  [-t VumarkerType] [-rf 0.X]"
                 << endl;
            cerr << endl;
        }

        /// Read file SVG
        SVGParse::read(argv[1], SVGFileData);

        /// Calcule bits used for identificate id in marker
        idBitMarker = std::stoi(argv[2]);

        /// Read input informations
        if (cml["-v"])
            TheInputVideo = cml("-v");
        if (cml["-c"])
        {
            TheCameraParameters.readFromXMLFile(cml("-c"));
            cameraParametersFile = cml("-c");
        }
        if (cml["-t"])
            markerType = cml("-t");

        /// Open video capture
        if (TheInputVideo.find("live") != string::npos)
        {
            int vIdx = 0;
            char cad[100];
            if (TheInputVideo.find(":") != string::npos)
            {
                std::replace(TheInputVideo.begin(), TheInputVideo.end(), ':', ' ');
                sscanf(TheInputVideo.c_str(), "%s %d", cad, &vIdx);
            }
            cout << "Opening camera index " << vIdx << endl;

            TheVideoCapturer.open(vIdx);
            TheVideoCapturer.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
            TheVideoCapturer.set(cv::CAP_PROP_AUTOFOCUS, 0);
            TheVideoCapturer.set(cv::CAP_PROP_FRAME_WIDTH, imageResolution.width);
            TheVideoCapturer.set(cv::CAP_PROP_FRAME_HEIGHT, imageResolution.height);
            waitTime = 10;
            isVideo = true;
        }
        else
        {
            TheVideoCapturer.open(TheInputVideo);
            if (TheVideoCapturer.get(cv::CAP_PROP_FRAME_COUNT) >= 2)
                isVideo = true;
            if (cml["-skip"])
                TheVideoCapturer.set(cv::CAP_PROP_POS_FRAMES, stoi(cml("-skip")));
        }

        /// Instanciate TheInputImage
        TheVideoCapturer >> TheInputImage;

        /// Init value for MDetector
        initMarkerDetector(MDetector);

        /// Check resize factor
        resizeFactor = stof(cml("-rf", "1"));
        if (resizeFactor != 1)
            TheInputImage = resizeImageFactor(TheInputImage, resizeFactor);
        TheCameraParameters.resize(TheInputImage.size());

        /// Setting Video Saved
        if (saveVideo)
        {
            TheVideoCapturer.set(cv::CAP_PROP_AUTOFOCUS, 0);
            TheVideoCapturer.set(cv::CAP_PROP_FRAME_WIDTH, imageResolution.width);   // 1280
            TheVideoCapturer.set(cv::CAP_PROP_FRAME_HEIGHT, imageResolution.height); // 720

            /// Start video in X frame
            if (frameStart != 0)
                TheVideoCapturer.set(cv::CAP_PROP_POS_FRAMES, frameStart);
            outputVideo = cv::VideoWriter("video/outputVideo.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fpsOutVideo,
                                          cv::Size(TheInputImage.cols, TheInputImage.rows));
        }

        // ----------------------------------------------------------------------------
        //   Method starts
        // ----------------------------------------------------------------------------
        char key = 0;
        do
        {
            /// Start capture video
            TheVideoCapturer >> TheInputImage;

            /// Create a new image to draw
            TheInputImage.copyTo(TheInputCopyImage);

            /// Check resize factor
            TheInputCopyImage = resizeImageFactor(TheInputCopyImage, resizeFactor);

            /// Detect and track marker
            Fps.start();
            MDetector.detectAndTrack(TheInputCopyImage, TheCameraParameters, sizeMarker);
            Fps.stop();

            /// Draw marker information
            for (auto marker : MDetector.markersDetected)
                marker.model.pose->draw(TheInputCopyImage, SVGFileData, marker.id, TheCameraParameters);

            if (showAllContours)
                drawAllContourFiltered(TheInputCopyImage, MDetector.filterContours);

            std::cout << "Frame:" << TheVideoCapturer.get(cv::CAP_PROP_POS_FRAMES) << "/"
                      << TheVideoCapturer.get(cv::CAP_PROP_FRAME_COUNT);
            std::cout << " Time detection: " << Fps.getAvrg() * 1000
                      << " milliseconds nmarkers: " << MDetector.markersDetected.size()
                      << " image resolution=" << TheInputCopyImage.size() << endl;

            /// Show information
            showInformation();

            if (saveVideo)
                outputVideo.write(TheInputCopyImage);

            /// Debug thresHoldImage
            if (thresHoldingDebug)
            {
                cv::namedWindow("Thresholding", 1);
                createTrackbar();
                cv::imshow("Thresholding", resize(MDetector.TheInputImageThresholding, 1080));
            }

            /// Show image
            cv::imshow("InputImage", resize(TheInputCopyImage, 1080));

            key = cv::waitKey(waitTime);

            if (key == 's')
                waitTime = waitTime == 0 ? 10 : 0;

            if (key == 'd')
                MDetector.only_detect = !MDetector.only_detect;

        } while (key != 27); //&& TheVideoCapturer.get(cv::CAP_PROP_POS_FRAMES) < TheVideoCapturer.get(cv::CAP_PROP_FRAME_COUNT));

        outputVideo.release();
    }
    catch (std::exception &ex)
    {
        cout << "Exception :" << ex.what() << endl;
    }
}
