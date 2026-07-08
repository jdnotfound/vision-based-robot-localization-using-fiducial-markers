/*
 * This file is part of the JuMarker library
 * Copyright (c) 2022 David Jurado Rodríguez, Rafael Muñoz Salinas,
 *                    Sergio Garrido Jurado, Rafael Medina Carnicer.
 *
 * JuMarker is published and distributed under the CC BY-NC-SA license.
 * JuMarker is distributed in the hope that it will be useful for
 * non-commercial academic research, but WITHOUT ANY WARRANTY.
 *
 * You should have received a copy of the CC BY-NC-SA license along with
 * this program; if not, write to rmsalinas@uco.es.
 */

#include <cmath>
#include "model.h"
#include <algorithm>


///*************** Model2ImageTransform *********************///
namespace model2image
{
std::shared_ptr<Model2ImageTransform> Model2ImageTransform::create(string type)
{
    if (type == "homography")
        return std::shared_ptr<Homo>(new Homo());
    else if (type == "rt")
        return std::shared_ptr<RT>(new RT());
    throw std::runtime_error("Invalid type");
}

///******* HOMO *******///

void Homo::setRvect(cv::Mat &Rvect)
{
    this->Rvec = Rvect;
}
void Homo::setTvect(cv::Mat &Tvect)
{
    this->Tvec = Tvect;
}

cv::Mat Homo::getHomo()
{
    return this->homography;
}

cv::Mat Homo::getRvect()
{
    return this->Rvec;
}
cv::Mat Homo::getTvect()
{
    Tvec.convertTo(Tvec, CV_64FC1);
    return this->Tvec;
}

/// Set params
void Homo::setParameters(cv::Mat &cameraMat, cv::Mat &distorsion, cv::Mat &_Rvec, cv::Mat &_Tvec, cv::Mat &_homography)
{
    _homography.convertTo(_homography, CV_64FC1, 1, 0);
    _Rvec.convertTo(_Rvec, CV_64FC1, 1, 0);
    _Tvec.convertTo(_Tvec, CV_64FC1, 1, 0);
    homography = _homography;
    Rvec = _Rvec;
    Tvec = _Tvec;
}

/// Proyect point inputImage2Model
cv::Point2f Homo::Image2Model(const cv::Point2f &p)
{
    assert(homography.type() == CV_64FC1);
    return homography.inv() * p;
}

/// Proyect point model2inputImage
cv::Point2f Homo::project(const cv::Point3f &p)
{
    assert(homography.type() == CV_64FC1);
    return homography * cv::Point2f(p.x, p.y);
}

void calculeWeights(const float &matchesPercent, const float &cornersPercent, std::vector<double> &weights,
                    std::vector<double> &weightsOcclusion)
{
    std::vector<double> _weights;

    // Total sum weights keypoint octave
    const double sumWeights = std::accumulate(weights.begin(), weights.end(), 0.0);
    const double sumWeightsOcclusion = std::accumulate(weightsOcclusion.begin(), weightsOcclusion.end(), 0.0);
    const double totalSum = sumWeights * matchesPercent + cornersPercent * sumWeightsOcclusion;

    for (auto &weight : weights)
        _weights.push_back(weight * totalSum);

    for (auto &weightOcclusion : weightsOcclusion)
        _weights.push_back(weightOcclusion / sumWeightsOcclusion * totalSum);

    weights = _weights;
}

double laplacianPixel(const cv::Mat &src, const cv::Point2f &corner)
{
    const int kernel[3][3] = {{1, 1, 1}, {1, -8, 1}, {1, 1, 1}};

    // cambiar aplicar al corner, 7x7 tamaño de ventana
    cv::Mat src_g = src;
    // cv::GaussianBlur(src, src_g,cv::Size(1,1),2.0,2.0);

    int i = corner.x;
    int j = corner.y;

    int pixval_x = (kernel[0][0] * (int)src_g.at<uchar>(j, i)) + (kernel[0][1] * (int)src_g.at<uchar>(j + 1, i)) +
                   (kernel[0][2] * (int)src_g.at<uchar>(j + 2, i)) + (kernel[1][0] * (int)src_g.at<uchar>(j, i + 1)) +
                   (kernel[1][1] * (int)src_g.at<uchar>(j + 1, i + 1)) + (kernel[1][2] * (int)src_g.at<uchar>(j + 2, i + 1)) +
                   (kernel[2][0] * (int)src_g.at<uchar>(j, i + 2)) + (kernel[2][1] * (int)src_g.at<uchar>(j + 1, i + 2)) +
                   (kernel[2][2] * (int)src_g.at<uchar>(j + 2, i + 2));

    int sum = abs(pixval_x);

    if (sum > 255)
        sum = 255;

    return sum;
}

double checkCornerGradient(const cv::Point2f &p, const cv::Mat &inputImage)
{
    cv::Rect rect(cv::Point(), inputImage.size());

    if (rect.contains(p))
        return laplacianPixel(inputImage, p);

    return -1;
}

/// Refine homography Subpixel
cv::Mat Homo::refineHomography(const std::vector<cv::DMatch> &matches, const std::vector<cv::KeyPoint> &keypoints_InputImage,
                               const std::vector<ModelElement> &elements, const std::vector<cv::Point2f> &markerCorners,
                               const cv::Mat &inputImage)
{
    std::vector<cv::Point2f> input, output, cornersBeforeSubPix;
    std::vector<double> weights, weightsOcclusion;
    const bool debugSubpix = false, homographySubPix = true;
    const float matchesPercent = 0.5, scaleFactor = 1.2, cornersPercent = 1 - matchesPercent;

    // Init input, output and weights vector
    for (auto &match : matches)
    {
        input.push_back(elements[match.queryIdx].SVGPoint);
        output.push_back(keypoints_InputImage[match.trainIdx].pt);
        weights.push_back(1. / pow(scaleFactor, keypoints_InputImage[match.trainIdx].octave));
    }

    // Proyect markerCorners-> InputImage and check if corner appear in image
    for (auto &vertex : markerCorners)
    {
        const cv::Point2f cornerMarker = homography * vertex;
        double gradient = checkCornerGradient(cornerMarker, inputImage);

        if (gradient != -1)
        {
            weightsOcclusion.push_back(gradient);
            cornersBeforeSubPix.push_back(cornerMarker);
        }
    }

    if (homographySubPix)
    {
        std::vector<cv::Point2f> objectPoints, imgPoints;
        for (size_t i = 0; i < cornersBeforeSubPix.size(); i++)
        {
            objectPoints.push_back(markerCorners[i]);
            imgPoints.push_back(cornersBeforeSubPix[i]);
        }
        // Calcule new Homography
        homography = findHomography(objectPoints, imgPoints);
        homography.convertTo(homography, CV_64FC1, 1, 0);
    }

    // Debug corner subPixel
    if (debugSubpix)
    {
        cv::Mat inputColorImage;
        cv::cvtColor(inputImage, inputColorImage, cv::COLOR_GRAY2BGR);
        for (size_t i = 0; i < cornersBeforeSubPix.size(); i++)
        {
            cv::circle(inputColorImage, cornersBeforeSubPix[i], 5, cv::Scalar(255, 0, 0), 1);
            cv::circle(inputColorImage, cornersBeforeSubPix[i], 5, cv::Scalar(0, 0, 255), 1);
        }
        cv::imshow("test 1", resize1(inputColorImage, 1080));
    }

    // Recalcule vector weights
    calculeWeights(matchesPercent, cornersPercent, weights, weightsOcclusion);

    // Add point after corner subPixels
    for (size_t i = 0; i < cornersBeforeSubPix.size(); i++)
    {
        input.push_back(markerCorners[i]);
        output.push_back(cornersBeforeSubPix[i]);
    }

    auto error = [&](const aruco::LevMarq<double>::eVector &sol, aruco::LevMarq<double>::eVector &err) {
        cv::Mat HomoAux = cv::Mat::eye(3, 3, CV_64F);

        for (int i = 0; i < 8; i++)
            HomoAux.ptr<double>(0)[i] = sol(i);

        err.resize(input.size() * 2);
        int ei = 0;
        for (size_t i = 0; i < input.size(); i++)
        {
            cv::Point2f ptres = HomoAux * input[i];
            err(ei) = weights[i] * (ptres.x - output[i].x);
            err(ei + 1) = weights[i] * (ptres.y - output[i].y);
            ei += 2;
        }
    };

    aruco::LevMarq<double> Solver;
    aruco::LevMarq<double>::eVector sol(8);
    Solver.setParams(200, 1e-9);

    sol(0) = homography.at<double>(0, 0);
    sol(1) = homography.at<double>(0, 1);
    sol(2) = homography.at<double>(0, 2);
    sol(3) = homography.at<double>(1, 0);
    sol(4) = homography.at<double>(1, 1);
    sol(5) = homography.at<double>(1, 2);
    sol(6) = homography.at<double>(2, 0);
    sol(7) = homography.at<double>(2, 1);

    Solver.solve(sol, std::bind(error, std::placeholders::_1, std::placeholders::_2));

    cv::Mat homo = cv::Mat::eye(3, 3, CV_64FC1);

    for (int i = 0; i < 8; i++)
        homo.ptr<double>(0)[i] = sol(i);

    return homo;
}

void Homo::optimize(std::vector<cv::DMatch> &matches, CameraParameters &cameraParameters,
                    const std::vector<cv::KeyPoint> &keypoints_InputImage, const std::vector<ModelElement> &elements,
                    const cv::Mat &inputImage, const std::vector<cv::Point2f> &markerCorners)
{
    if (matches.size() < 10)
        return;

    vector<cv::Point2f> imgPoints, objectPoints;
    const float proyectionError = 0.9;

    for (auto &match : matches)
    {
        objectPoints.push_back(cv::Point2f(elements[match.queryIdx].SVGPoint.x, elements[match.queryIdx].SVGPoint.y));
        imgPoints.push_back(keypoints_InputImage[match.trainIdx].pt);
    }

    // Calcule new Homography
    cv::Mat newHomography = findHomography(objectPoints, imgPoints, cv::RANSAC);

    // Assert new homography and vector inliers not empty
    if (newHomography.empty())
        return;

    newHomography.convertTo(newHomography, CV_64FC1, 1, 0);

    // Update matrix homography
    homography = newHomography;

    std::vector<cv::DMatch> inliersMatches;

    // Calcule inliers matches
    for (auto &match : matches)
    {
        cv::Point2f p1 = homography * elements[match.queryIdx].SVGPoint;
        cv::Point2f p2 = keypoints_InputImage[match.trainIdx].pt;

        // Check error proyection < proyectionError
        if (abs(p1.x - p2.x) < proyectionError && abs(p1.y - p2.y) < proyectionError)
            inliersMatches.push_back(match);
    }

    // Refine homography with corner subpixel
    if (inliersMatches.size() < 4)
        return;

    // Update input matches
    matches = inliersMatches;

    homography = refineHomography(inliersMatches, keypoints_InputImage, elements, markerCorners, inputImage);

    // Recalcule objectsPoint with homography refined and inliers matches
    vector<cv::Point3f> objPoints; // Its necessary 3D points to solvePnP
    imgPoints.clear();

    for (auto &match : inliersMatches)
    {
        cv::Point2f p1 = keypoints_InputImage[match.trainIdx].pt;
        cv::Point2f p2 = homography.inv() * p1;

        objPoints.push_back(cv::Point3f(p2.x, p2.y, 0));
        imgPoints.push_back(p1);
    }

    // Use solvePnP to calcule Rvec and Tvec
    cv::solvePnP(objPoints, imgPoints, cameraParameters.CameraMatrix, cameraParameters.Distorsion, Rvec, Tvec, true);

    Rvec.convertTo(Rvec, CV_64FC1);
    Tvec.convertTo(Tvec, CV_64FC1);
}

inline cv::Point2f getCenter(vector<cv::Point2f> &corners)
{
    if (corners.size() > 2)
    {
        float doubleArea = 0;
        cv::Point2f p(0, 0);
        cv::Point2f p0 = corners.back();
        for (const cv::Point2f &p1 : corners)
        {                                        // C++11
            float a = p0.x * p1.y - p0.y * p1.x; // cross product, (signed) double area of triangle of vertices (origin,p0,p1)
            p += (p0 + p1) * a;
            doubleArea += a;
            p0 = p1;
        }
        if (doubleArea != 0)
            return p * (1 / (3 * doubleArea)); // Operator / does not exist for cv::Point
    }
    return cv::Point2f();
}

cv::Point2f Homo::getCenterMarker(const SVGData &SVGFileData)
{
    std::vector<cv::Point2f> imagePoints;
    for (auto &p : SVGFileData.border)
        imagePoints.push_back(p);

    return getCenter(imagePoints);
}

/// Draw information marker
/// Draw information marker
/// Draw information marker
void Homo::draw(cv::Mat &inpuImage, const SVGData &SVGFileData, const int id, const CameraParameters &cameraParameters)
{
    assert(homography.type() == CV_64FC1 && Rvec.type() == CV_64FC1);

    const int lineSize = 2;
    const int height = 500;

    // ---------------------------------------------------------
    // Get marker border points and center in SVG/model coordinates
    // This was missing in your current pasted code.
    // ---------------------------------------------------------
    std::vector<cv::Point2f> modelBorderPoints;
    for (auto &p : SVGFileData.border)
        modelBorderPoints.push_back(p);

    cv::Point2f markerCenter = getCenter(modelBorderPoints);

    // ---------------------------------------------------------
    // Print numeric pose values
    // Raw Tvec is in JuMarker/SVG model units, not meters.
    // Convert it to meters using real physical marker height.
    // ---------------------------------------------------------
    if (!Tvec.empty() && Tvec.total() >= 3 && !SVGFileData.border.empty())
    {
        cv::Mat T;
        Tvec.convertTo(T, CV_64FC1);

        double tx_units = 0.0, ty_units = 0.0, tz_units = 0.0;

        if (T.rows == 1)
        {
            tx_units = T.at<double>(0, 0);
            ty_units = T.at<double>(0, 1);
            tz_units = T.at<double>(0, 2);
        }
        else
        {
            tx_units = T.at<double>(0, 0);
            ty_units = T.at<double>(1, 0);
            tz_units = T.at<double>(2, 0);
        }

        double minY = SVGFileData.border[0].y;
        double maxY = SVGFileData.border[0].y;

        for (auto &p : SVGFileData.border)
        {
            minY = std::min(minY, (double)p.y);
            maxY = std::max(maxY, (double)p.y);
        }

        double model_height_units = maxY - minY;

        // Current real marker height = 6 cm.
        // Later change this to 0.096 if using 9.6 cm marker.
        const double physical_marker_height_m = 0.096;

        if (model_height_units > 1e-9)
        {
            double units_to_m = physical_marker_height_m / model_height_units;

            double tx_m = tx_units * units_to_m;
            double ty_m = ty_units * units_to_m;
            double tz_m = tz_units * units_to_m;

            double dist_m = std::sqrt(tx_m * tx_m + ty_m * ty_m + tz_m * tz_m);

            std::cout << "POSE id:" << id
                      << " tx_m:" << tx_m
                      << " ty_m:" << ty_m
                      << " tz_m:" << tz_m
                      << " dist_m:" << dist_m
                      << " raw_tz_units:" << tz_units
                      << " model_height_units:" << model_height_units
                      << std::endl;
        }
    }

    // ---------------------------------------------------------
    // Draw pose axis using original Rvec/Tvec
    // ---------------------------------------------------------
    CvDrawingUtils::draw3dAxis(inpuImage, Rvec, Tvec,
                               cameraParameters.CameraMatrix,
                               cameraParameters.Distorsion,
                               markerCenter,
                               lineSize,
                               height);

    // ---------------------------------------------------------
    // Homography-locked visual 3D polygon
    // This is only for visual demo.
    // It starts exactly from the correct detected 2D border.
    // ---------------------------------------------------------
    std::vector<cv::Point2f> basePts;
    std::vector<cv::Point2f> topPts;

    for (auto &p : SVGFileData.border)
    {
        basePts.push_back(homography * p);
    }

    // Tune these if you want the fake depth direction to change.
    // +x = right, +y = down.
    cv::Point2f fakeDepthShift(90, 55);

    for (auto &p : basePts)
    {
        topPts.push_back(p + fakeDepthShift);
    }

    const cv::Scalar blueColor(255, 0, 0);

    for (int i = 0; i < (int)basePts.size(); i++)
    {
        int j = (i + 1) % (int)basePts.size();

        // Line from detected marker corner to fake 3D corner
        cv::line(inpuImage, basePts[i], topPts[i], blueColor, lineSize, cv::LINE_8);

        // Top/back polygon edge
        cv::line(inpuImage, topPts[i], topPts[j], blueColor, lineSize, cv::LINE_8);
    }

    // Draw red marker border last so it remains clearly visible
    CvDrawingUtils::drawMarkerBorderH(inpuImage, homography, SVGFileData, markerCenter, lineSize, id);
}

///******* RT *******///

cv::Point2f RT::getCenterMarker(const SVGData &SVGFileData)
{
    std::vector<cv::Point2f> imagePoints;
    for (auto &p : SVGFileData.border)
        imagePoints.push_back(p);

    return getCenter(imagePoints);
}

void RT::setRvect(cv::Mat &Rvect)
{
    this->Rvec = Rvect;
}
void RT::setTvect(cv::Mat &Tvect)
{
    this->Tvec = Tvect;
}
cv::Mat RT::getHomo()
{
    return this->homography;
}
cv::Mat RT::getRvect()
{
    return this->Rvec;
}
cv::Mat RT::getTvect()
{
    return this->Tvec;
}

/// Set params
void RT::setParameters(cv::Mat &cameraMat, cv::Mat &distorsion, cv::Mat &Rvec, cv::Mat &Tvec, cv::Mat &homography)
{
    cameraMat.convertTo(cameraMat, CV_64FC1);
    distorsion.convertTo(distorsion, CV_64FC1);
    Rvec.convertTo(Rvec, CV_64FC1);
    Tvec.convertTo(Tvec, CV_64FC1);

    this->cameraMat = cameraMat;
    this->distorsion = distorsion;
    this->Rvec = Rvec;
    this->Tvec = Tvec;
}

/// Proyect point inputImage2Model
cv::Point2f RT::Image2Model(const cv::Point2f &p)
{
    assert(homography.type() == CV_64FC1);
    return homography.inv() * p;
}

/// Proyect point model2inputImage
cv::Point2f RT::project(const cv::Point3f &p)
{
    assert(Rvec.type() == CV_64FC1 && Tvec.type() == CV_64FC1);

    std::vector<cv::Point3f> objectPoints;
    std::vector<cv::Point2f> imagePoints;
    objectPoints.push_back(p);

    // Proyect fileData border using new RT vector
    cv::projectPoints(objectPoints, Rvec, Tvec, cameraMat, distorsion, imagePoints);

    return imagePoints[0];
}

/// Refine homography Subpixel
cv::Mat RT::refineHomography(const std::vector<cv::DMatch> &matches, const std::vector<cv::KeyPoint> &keypoints_InputImage,
                             const std::vector<ModelElement> &elements, const std::vector<cv::Point2f> &markerCorners,
                             const cv::Mat &inputImage)
{
    return cv::Mat(); // not implemented
}

void RT::optimize(std::vector<cv::DMatch> &matches, CameraParameters &cameraParameters,
                  const std::vector<cv::KeyPoint> &keypoints_InputImage, const std::vector<ModelElement> &elements,
                  const cv::Mat &inputImage, const std::vector<cv::Point2f> &markerCorners)
{
    vector<cv::Point3f> objPoints;
    vector<cv::Point2f> imgPoints;
    // Calcule vector for solvepnp
    for (int i = 0; i < matches.size(); i++)
    {
        objPoints.push_back(cv::Point3f(elements[matches[i].queryIdx].SVGPoint.x, elements[matches[i].queryIdx].SVGPoint.y, 0));
        imgPoints.push_back(keypoints_InputImage[matches[i].trainIdx].pt);
    }

    assert(Rvec.type() == CV_64FC1 && Tvec.type() == CV_64FC1);
    if (objPoints.size() < 8)
        return;

    // Transfor input 3D points in 2D points
    std::vector<cv::Point2f> objectPoints;
    for (auto &p : objPoints)
        objectPoints.push_back(cv::Point2f(p.x, p.y));

    // Calcule new Homography
    cv::Mat newHomography = findHomography(objPoints, imgPoints, cv::RANSAC);

    // inliers

    // Assert new homography and vector inliers not empty
    if (newHomography.empty())
        return;

    // Update matrix homography
    homography = newHomography;
    homography.convertTo(homography, CV_64FC1, 1, 0);

    // solvepnp

    vector<int> inliers;
    // Use solvePnPRansac
    cv::solvePnPRansac(objPoints, imgPoints, cameraMat, distorsion, Rvec, Tvec, false, 100, 2.0, 0.99, inliers,
                       cv::SOLVEPNP_ITERATIVE);
    Rvec.convertTo(Rvec, CV_64FC1);
    Tvec.convertTo(Tvec, CV_64FC1);

    if (inliers.empty())
        return;

    // Update vector matches
    std::vector<cv::DMatch> matchesInliers(inliers.size());

    for (int i = 0; i < inliers.size(); i++)
        matchesInliers[i] = matches[inliers[i]];

    matches = matchesInliers;
}

/// Draw information marker
void RT::draw(cv::Mat &Image, const SVGData &SVGFileData, const int id, const CameraParameters &cameraParameters)
{
    assert(Rvec.type() == CV_64FC1 && Tvec.type() == CV_64FC1);
    const int lineSize = 4;
    const int height = 400;

    std::vector<cv::Point2f> imagePoints;
    for (auto &p : SVGFileData.border)
        imagePoints.push_back(p);

    cv::Point2f markerCenter = getCenter(imagePoints);

    // Draw 3DAxis
    CvDrawingUtils::draw3dAxis(Image, Rvec, Tvec, cameraMat, distorsion, markerCenter, lineSize, height);

    // Draw 3D polygon
    CvDrawingUtils::draw3dPolygon(Image, Rvec, Tvec, cameraMat, distorsion, SVGFileData, lineSize, height);

    // Draw marker border
    CvDrawingUtils::drawMarkerBorder(Image, Rvec, Tvec, cameraMat, distorsion, SVGFileData, lineSize);
}

} // namespace model2image

///*************** Model *********************///

Model::Model(std::shared_ptr<model2image::Model2ImageTransform> _pose) : pose(_pose)
{
}

///*************** ModelElement *********************///

ModelElement::ModelElement(cv::Point2f &SVGPoint, cv::Mat descriptor, cv::KeyPoint &inputKeyPoint)
{
    this->SVGPoint = SVGPoint;
    this->descriptor = descriptor;
    this->inputKeyPoint = inputKeyPoint;
    this->_isStrong = false;
}

void ModelElement::setAsMatched(bool val)
{
    // if(_isStrong) return;
    if (val)
        _nmatched++;
}

void ModelElement::increaseAliveCounter()
{
    nframes_alive++;
}

bool ModelElement::mustBeremoved()
{
    if (nframes_alive < 8)
        return false;
    else
    {
        // if(_isStrong) return false;
        // else
        //{
        if (float(_nmatched) / float(nframes_alive) > 0.5f)
            _isStrong = true;
        else
            _isStrong = false;
        //}
        return !_isStrong;
    }
}
