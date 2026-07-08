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

#include "markerdetector.h"

const int CRCBits = 16;
const int idMarkerFilter = 0;
const vector<int> radius = {30, 5};
const bool printInfo = false;
const int minMatches = 15;
const double percent2UpdateModel = 0.6;
int idMatchesDebug = 0;

MarkerDetector::MarkerDetector(const string &type)
{
    modelType = type;
}

inline cv::Mat resize(const cv::Mat &in, int width)
{
    if (in.size().width <= width)
        return in;
    float yf = float(width) / float(in.size().width);
    cv::Mat im2;
    cv::resize(in, im2, cv::Size(width, static_cast<int>(in.size().height * yf)));
    return im2;
}

inline float getSubpixelValue(const cv::Mat &im_grey, const cv::Point2f &p)
{
    float intpartX;
    float decameraParametersartX = std::modf(p.x, &intpartX);
    float intpartY;
    float decameraParametersartY = std::modf(p.y, &intpartY);
    cv::Point tl;

    if (decameraParametersartX > 0.5)
    {
        if (decameraParametersartY > 0.5)
            tl = cv::Point(intpartX, intpartY);
        else
            tl = cv::Point(intpartX, intpartY - 1);
    }
    else
    {
        if (decameraParametersartY > 0.5)
            tl = cv::Point(intpartX - 1, intpartY);
        else
            tl = cv::Point(intpartX - 1, intpartY - 1);
    }
    return (1.f - decameraParametersartY) * (1. - decameraParametersartX) * float(im_grey.at<uchar>(tl.y, tl.x)) +
           decameraParametersartX * (1 - decameraParametersartY) * float(im_grey.at<uchar>(tl.y, tl.x + 1)) +
           (1 - decameraParametersartX) * decameraParametersartY * float(im_grey.at<uchar>(tl.y + 1, tl.x)) +
           decameraParametersartX * decameraParametersartY * float(im_grey.at<uchar>(tl.y + 1, tl.x + 1));
}

inline cv::Vec3b getSubpixelSimpleValue(const cv::Mat &imageBGR, const cv::Point2f &p)
{
    int x = p.x + 0.5;
    int y = p.y + 0.5;
    if (x >= imageBGR.cols)
        x = imageBGR.cols - 1;
    if (y >= imageBGR.cols)
        y = imageBGR.rows - 1;
    return imageBGR.at<cv::Vec3b>(y, x);
}

inline uint16_t crc16(uint8_t const *data, size_t size)
{
    uint16_t crc = 0;
    while (size--)
    {
        crc ^= *data++;
        for (unsigned k = 0; k < 8; k++)
            crc = crc & 1 ? (crc >> 1) ^ 0xa001 : crc >> 1;
    }
    return crc;
}

inline string uint16ToBinary(uint16_t n)
{
    char buffer[17]; /* 16 bits, plus room for a \0 */
    buffer[16] = '\0';
    for (int i = 15; i >= 0; --i)
    {                              /* convert bits from the end */
        buffer[i] = '0' + (n & 1); /* '0' + bit => '0' or '1' */
        n >>= 1;                   /* make the next bit the 'low bit' */
    }
    return buffer;
}

inline bool checkPoint(const cv::Point2f &point, const cv::Mat &Image)
{
    if (point.x < Image.cols && point.y < Image.rows && point.y > 0 && point.x > 0)
        return true;
    else
        return false;
}

std::vector<std::vector<cv::Point>> contoursFilter(vector<vector<cv::Point>> &allContours, int vertexNumber, bool convexPolygon)
{
    vector<vector<cv::Point>> filterContour;

    for (auto &contour : allContours)
    {
        vector<cv::Point> approxcontour;

        if (cv::contourArea(contour) > 2000)
        {

            // Approximate Polygon
            double epsilon = 0.01 * contour.size();
            cv::approxPolyDP(contour, approxcontour, epsilon, true);

            if (approxcontour.size() == vertexNumber)
            {
                // If the marker contour  designed is convex, only are interesting the convex image contours
                if (convexPolygon)
                {
                    if (isContourConvex(approxcontour))
                        filterContour.push_back(approxcontour);
                }
                // If marker is not a convex polygon, add the contour
                else
                    filterContour.push_back(approxcontour);
            }
        }
    }
    return filterContour;
}

inline void normalize(std::vector<float> &vec)
{
    float sum = 0;
    for (auto v : vec)
        sum += v;
    float invsum = 1. / sum;
    for (auto &v : vec)
        v *= invsum;
}

inline vector<vector<cv::Point2f>> circularShift(const vector<cv::Point> &a)
{
    vector<vector<cv::Point2f>> b;
    for (auto it = a.begin(); it != a.end(); ++it)
    {
        vector<cv::Point2f> dest(a.size());
        std::rotate_copy(a.begin(), it, a.end(), dest.begin());
        if (b.size() != 0 && dest == b[0])
        {
            return b;
        } // checks to see if you have repetition in cycles.
        b.push_back(dest);
    }
    return b;
}

inline float average(std::vector<float> &vec, int start, int end, int initFrequency)
{
    float avrg = 0;
    float sum = 0;
    do
    {
        avrg += float(initFrequency) * vec[start % 180];
        sum += vec[start % 180];
        initFrequency++;
        start++;

    } while (initFrequency < end);
    return avrg / sum;
}

inline void ucoslam_FM___computeThreeMaxima(const vector<vector<int>> &histo, int &ind1, int &ind2, int &ind3)
{
    int max1 = 0;
    int max2 = 0;
    int max3 = 0;

    for (size_t i = 0; i < histo.size(); i++)
    {
        const int s = histo[i].size();
        if (s > max1)
        {
            max3 = max2;
            max2 = max1;
            max1 = s;
            ind3 = ind2;
            ind2 = ind1;
            ind1 = i;
        }
        else if (s > max2)
        {
            max3 = max2;
            max2 = s;
            ind3 = ind2;
            ind2 = i;
        }
        else if (s > max3)
        {
            max3 = s;
            ind3 = i;
        }
    }

    if (max2 < 0.1f * (float)max1)
    {
        ind2 = -1;
        ind3 = -1;
    }
    else if (max3 < 0.1f * (float)max1)
    {
        ind3 = -1;
    }
}

inline cv::Mat calculeHomography(vector<cv::Point2f> &initPoints, vector<cv::Point> &destinationContours)
{

    cv::Mat finalHomography;
    float finalError = std::numeric_limits<float>::max();

    // Calcule all possible changes in vector
    vector<vector<cv::Point2f>> allCombination = circularShift(destinationContours);

    // Check all possible rotation
    for (auto combination : allCombination)
    {
        // Calcule homography for each change in vector
        cv::Mat parcialHomography = cv::findHomography(initPoints, combination);

        parcialHomography.convertTo(parcialHomography, CV_64FC1, 1, 0);

        float errorPoint = 0;

        // Calcule error
        for (int j = 0; j < combination.size(); j++)
        {
            cv::Point2f p = parcialHomography * initPoints[j];
            errorPoint += cv::norm(p - combination[j]);
        }

        // Update general error
        if (errorPoint < finalError)
        {
            finalHomography = parcialHomography;
            finalError = errorPoint / destinationContours.size();
        }
    }

    return finalHomography;
}

inline float variance(std::vector<float> &vec, float mean, int start, int end, int initFrequency)
{
    float res = 0;
    float sum = 0;
    do
    {
        res += vec[start % 180] * (float(initFrequency) - mean) * (float(initFrequency) - mean);
        sum += vec[start % 180];
        start++;
        initFrequency++;

    } while (initFrequency < end);
    return res / sum;
}

float circularAverage(std::vector<float> vect)
{
    std::vector<float> hist(180, 0);
    for (auto e : vect)
        hist[int(e)]++;
    normalize(hist);

    int rot = 0;
    int rotation = 0;
    std::pair<int, float> best(-1, std::numeric_limits<float>::min());

    float GAvrg = 90;
    float SigmaExtra = variance(hist, GAvrg, rot, 180, 0);
    do
    {
        float Avrg1 = average(hist, rot, 90, 0);
        float Sigma1Intra = variance(hist, Avrg1, rot, 90, 0);

        float Avrg2 = average(hist, rot + 90, 180, 90);
        float Sigma2Intra = variance(hist, Avrg2, rot + 90, 180, 90);

        float Goodness = SigmaExtra / (Sigma1Intra + Sigma2Intra);
        if (std::isnan(Goodness))
            Goodness = 0;

        if (Goodness > best.second)
            best = {rotation, Goodness};

        rotation++;
        rot = (180 - rotation);
    } while (rotation < 180);
    return best.first;
}

string getBinaryCode(const int strat, int end, const std::vector<float> &pixelsColors, const int numberBitsData)
{
    string allColorsBits;
    if (pixelsColors.empty())
    {
        for (int j = 0; j < numberBitsData + CRCBits; j++)
            allColorsBits.push_back('0');
    }
    else
    {
        for (int j = 0; j < numberBitsData + CRCBits; j++)
        {
            // Calcule color subpixel
            float pixelValue = pixelsColors[j];

            if (pixelValue < end && pixelValue > strat)
                allColorsBits.push_back('1');
            else
                allColorsBits.push_back('0');
        }
    }

    return allColorsBits;
}

std::vector<float> MarkerDetector::extractElementsColors(cv::Mat &homography)
{
    std::vector<float> pixelsColorsBis;

    cv::Scalar color = cv::Scalar(-1);
    for (auto &element : SVGFileData.elements)
    {
        cv::Point2f p = homography * element.center;

        // cv::circle(TheInputImage,p,5,color,-1,cv::LINE_8,0);

        if (checkPoint(p, TheInputImage))
        {
            if (SVGFileData.HSV)
            {
                cv::Mat HSV;
                cv::Mat BGR = TheInputImage(cv::Rect(p.x, p.y, 1, 1));
                cvtColor(BGR, HSV, cv::COLOR_BGR2HSV);
                cv::Vec3b hsv = HSV.at<cv::Vec3b>(0, 0);
                pixelsColorsBis.push_back(hsv.val[0]);
            }
            else
                pixelsColorsBis.push_back(getSubpixelValue(TheInputImageGrey, p));
        }
    }

    return pixelsColorsBis;
}

void MarkerDetector::getCRCandId(cv::Mat &homo, string &CRC, string &ID)
{
    std::vector<float> pixelsColors = extractElementsColors(homo);
    // Calcule idcode and CRC for each contour
    string elementsColors;
    if (SVGFileData.HSV)
    {
        float maxAverageHSV = circularAverage(pixelsColors);
        if (maxAverageHSV == 0)
            maxAverageHSV = 1;
        float minAverageHSV = (180 - maxAverageHSV);
        if (maxAverageHSV < minAverageHSV)
            swap(maxAverageHSV, minAverageHSV);

        elementsColors = getBinaryCode(minAverageHSV, maxAverageHSV, pixelsColors, markerbitsData);

        // Subtract from
        ID = elementsColors.substr(0, markerbitsData);
        CRC = elementsColors.substr(markerbitsData, markerbitsData + CRCBits);
    }
    else
    {
        float colorAverageGray = std::accumulate(pixelsColors.begin(), pixelsColors.end(), 0.0) / pixelsColors.size();
        ;
        elementsColors = getBinaryCode(0, colorAverageGray, pixelsColors, markerbitsData);
        ID = elementsColors.substr(0, markerbitsData);
        CRC = elementsColors.substr(markerbitsData, markerbitsData + CRCBits);
    }
}

inline bool PointInPolygon(const cv::Point2f &point, const std::vector<cv::Point2f> &points)
{
    int i, j, nvert = points.size();
    bool c = false;
    for (i = 0, j = nvert - 1; i < nvert; j = i++)
    {
        if (((points[i].y >= point.y) != (points[j].y >= point.y)) &&
            (point.x <= (points[j].x - points[i].x) * (point.y - points[i].y) / (points[j].y - points[i].y) + points[i].x))
            c = !c;
    }

    return c;
}

bool CornersInsideMarker(Marker &m1, Marker &m2)
{
    for (auto corner : m2.corners)
    {
        if (PointInPolygon(corner, m1.corners))
            return true;
    }

    return false;
}

bool subMarker(Marker &m, std::vector<Marker> &markersDetected)
{
    for (auto marker : markersDetected)
        if (CornersInsideMarker(m, marker))
            return true;
    return false;
}

void MarkerDetector::addMarkerDetected(CameraParameters &cameraParameters, float markerSizeMeters, int idMarker,
                                       const cv::Mat &homography)
{
    std::vector<cv::Point2f> corners;

    for (cv::Point2f p : SVGFileData.border)
        corners.push_back(homography * p);

    // Corner SubPixel
    const int _winSize = 12;
    const cv::Size winSize = cv::Size(_winSize, _winSize);
    const cv::Size zeroZone = cv::Size(-1, -1);
    const cv::TermCriteria criteria(cv::TermCriteria::MAX_ITER | cv::TermCriteria::EPS, 12, 0.005);
    std::vector<cv::Point2f> objectPoints;

    cv::cornerSubPix(TheInputImageGrey, corners, winSize, zeroZone, criteria);

    for (size_t i = 0; i < corners.size(); i++)
        objectPoints.push_back(SVGFileData.border[i]);

    // Create marker object
    Marker marker(modelType);
    marker.id = idMarker;
    marker.corners = corners;
    marker.homography = findHomography(objectPoints, corners);

    marker.calculateExtrinsics(markerSizeMeters, cameraParameters.CameraMatrix, SVGFileData, cameraParameters.Distorsion);
    // Check if the marker detected is a submarker
    if (!subMarker(marker, markersDetected))
        markersDetected.push_back(marker);
}

std::vector<cv::Mat> calculeHomographySymmetry(const vector<cv::Point2f> &initPoints,
                                               const vector<cv::Point> &destinationContours)
{
    std::vector<cv::Mat> homographies;

    // Calcule all possible changes in vector
    vector<vector<cv::Point2f>> allCombination = circularShift(destinationContours);

    for (auto combination : allCombination)
    {
        cv::Mat parcialHomography = cv::findHomography(initPoints, combination);

        parcialHomography.convertTo(parcialHomography, CV_64FC1, 1, 0);

        homographies.push_back(parcialHomography);
    }

    return homographies;
}

float getHammDescDistance_2(const uint64_t *ptr1, const uint64_t *ptr2, uint64_t descSize)
{
    int n8 = descSize / 8;
    int hamm = 0;
    for (int i = 0; i < n8; i++)
        hamm += std::bitset<64>(ptr1[i] ^ ptr2[i]).count();

    int extra = descSize - n8 * 4;
    if (extra == 0)
        return hamm;

    const uint8_t *uptr1 = (uint8_t *)ptr1 + n8;
    const uint8_t *uptr2 = (uint8_t *)ptr2 + n8;
    for (int i = 0; i < extra; i++)
        hamm += std::bitset<8>(uptr1[i] ^ uptr2[i]).count();

    // finally, the rest
    return hamm;
}

float getHammDescDistance(const cv::Mat &dsc1, const cv::Mat &dsc2)
{
    assert(dsc1.type() == dsc2.type());
    assert(dsc1.rows == dsc2.rows && dsc2.rows == 1);
    assert(dsc1.cols == dsc2.cols);
    assert(dsc1.type() == CV_8UC1);
    return getHammDescDistance_2(dsc1.ptr<uint64_t>(0), dsc2.ptr<uint64_t>(0), dsc1.total());
}

bool insideBoundingBox(cv::Point2f p, cv::Rect box)
{
    if (p.y > box.y + box.height || p.y < box.y)
        return false;
    else if (p.x > box.x + box.width || p.x < box.x)
        return false;
    else
        return true;
}

void filter_ambiguous_train(std::vector<cv::DMatch> &matches)
{
    if (matches.size() == 0)
        return;
    // determine maximum values of queryIdx
    int maxT = -1;
    for (auto m : matches)
        maxT = std::max(maxT, m.trainIdx);

    // now, create the vector with the elements
    vector<int> used(maxT + 1, -1);
    vector<cv::DMatch> best_matches(maxT);
    int idx = 0;
    bool needRemove = false;

    for (auto &match : matches)
    {
        if (used[match.trainIdx] == -1)
        {
            used[match.trainIdx] = idx;
        }
        else
        {
            if (matches[used[match.trainIdx]].distance > match.distance)
            {
                matches[used[match.trainIdx]].trainIdx = -1; // annulate the other match
                used[match.trainIdx] = idx;
            }
            else
            {
                match.trainIdx = -1;
            } // annulate this match
        }
        needRemove = true;
        idx++;
    }
    if (needRemove)
        matches.erase(std::remove_if(matches.begin(), matches.end(),
                                     [](const cv::DMatch &m) { return m.trainIdx == -1 || m.queryIdx == -1; }),
                      matches.end());
}

void MarkerDetector::createMarker(vector<cv::Point> &filterContour, CameraParameters &cameraParameters, cv::Mat &homography,
                                  string idCode, string crcImage, float markerSizeMeters)
{
    Fps.start();
    bool traceBits_Value = false;
    if (VuMark)
    {
        string codeVumark;
        string crcVumark;

        if (vumarkerName == "rubik")
        {
            codeVumark = "1";
            crcVumark = "1100100001010010";
        }
        else if (vumarkerName == "camera")
        {
            codeVumark = "1";
            crcVumark = "0101000110010000";
        }
        else if (vumarkerName == "cordoba")
        {
            codeVumark = "0";
            crcVumark = "1110010111100111";
        }
        else if (vumarkerName == "seabery")
        {
            codeVumark = "1";
            crcVumark = "1111111001101001";
        }
        else if (vumarkerName == "UCO")
        {
            codeVumark = "1";
            crcVumark = "1110101101110110";
        }
        else if (vumarkerName == "building")
        {
            codeVumark = "1";
            crcVumark = "0110111010010101";
        }
        else if (vumarkerName == "JR")
        {
            codeVumark = "1";
            crcVumark = "0110010110110101";
        }
        else if (vumarkerName == "JRBis")
        {
            codeVumark = "1";
            crcVumark = "0110000000011010";
        }

        string codeVumarkInv = codeVumark;
        boost::replace_all(codeVumarkInv, "1", "3");
        boost::replace_all(codeVumarkInv, "0", "1");
        boost::replace_all(codeVumarkInv, "3", "0");

        if (traceBits_Value)
            std::cout << crcImage << endl;

        string crcVumarkInv = crcVumark;
        boost::replace_all(crcVumarkInv, "1", "3");
        boost::replace_all(crcVumarkInv, "0", "1");
        boost::replace_all(crcVumarkInv, "3", "0");

        if ((idCode == codeVumark && crcImage == crcVumark) || (idCode == codeVumarkInv && crcImage == crcVumark))
        {

            // Calcule corner point with homography
            std::vector<cv::Point2f> corners;
            for (int j = 0; j < filterContour.size(); j++)
            {
                cv::Point2f corner = homography * SVGFileData.border[j];
                corners.push_back(corner);
            }

            const int idVumark = 32;
            addMarkerDetected(cameraParameters, markerSizeMeters, idVumark, homography);
        }
    }
    else
    {
        // Calcule Id marker
        int idMarker = std::stoi(idCode, nullptr, 2);
        const uint8_t *p = reinterpret_cast<const uint8_t *>(idCode.c_str());
        // Calcule its CRC
        string crcID = uint16ToBinary(crc16(p, markerbitsData));

        string crcIDD = crcImage;
        boost::replace_all(crcIDD, "1", "3");
        boost::replace_all(crcIDD, "0", "1");
        boost::replace_all(crcIDD, "3", "0");

        string idInv = idCode;
        boost::replace_all(idInv, "1", "3");
        boost::replace_all(idInv, "0", "1");
        boost::replace_all(idInv, "3", "0");

        const uint8_t *pInv = reinterpret_cast<const uint8_t *>(idInv.c_str());
        string crcImageInv = uint16ToBinary(crc16(pInv, markerbitsData));

        // Check valid Marker
        if ((crcID == crcImage || crcIDD == crcImageInv))
        {
            if (crcIDD == crcImageInv)
                idMarker = std::stoi(idInv, nullptr, 2);

            // Calcule corner point with homography
            std::vector<cv::Point2f> corners, cornersContour;
            for (int j = 0; j < filterContour.size(); j++)
            {
                cv::Point2f corner = homography * SVGFileData.border[j];
                corners.push_back(corner);
                cornersContour.push_back(cv::Point2f(filterContour[j].x, filterContour[j].y));
            }

            addMarkerDetected(cameraParameters, markerSizeMeters, idMarker, homography);
        }
        // Revisar
        /*
        if(IndexMarkerInVMarker == -1)
        {
            Marker marker = addMarkerDetected(filterContour,cameraParameters,markerSizeMeters,idMarker,homography);
            // Add marker in markersDetected vector
            markersDetected.push_back(marker);
        }

        // This id marker has been detected, choose marker with biggest area
        else
        {
            // Choose marker with biggest area
            bool pointInsideMarker = checkRepeatMarker(corners,IndexMarkerInVMarker);

            if(pointInsideMarker  && cv::contourArea(corners) > cv::contourArea(markersDetected[IndexMarkerInVMarker].corners))
            {
                Marker marker = addMarkerDetected(filterContour,cameraParameters,markerSizeMeters,idMarker,homography);
                // Add marker in markersDetected vector
                markersDetected[IndexMarkerInVMarker] = marker;
            }
            else
            {
                Marker marker = addMarkerDetected(filterContour,cameraParameters,markerSizeMeters,idMarker,homography);
                // Add marker in markersDetected vector
                markersDetected.push_back(marker);
            }
            return;
        }
*/
    }
}

void MarkerDetector::markerCenter()
{
    for (int i = 0; i < markersDetected.size(); i++)
    {
        cv::Point2f center = markersDetected[i].model.pose->getCenterMarker(SVGFileData);

        // RT origin axis to camera
        cv::Mat R1 = markersDetected[i].model.pose->getRvect().clone();
        cv::Mat T1 = markersDetected[i].model.pose->getTvect().clone();

        // RT marker axis to origin axis
        cv::Mat R2 = R1.clone();
        cv::Mat T2 = T1.clone();

        R2.setTo(0);

        T2.at<double>(0, 0) = center.x;
        T2.at<double>(1, 0) = center.y;
        T2.at<double>(2, 0) = 0;

        cv::Mat R3, T3;

        cv::composeRT(R2, T2, R1, T1, R3, T3);
        markersDetected[i].model.pose->setRvect(R3);
        markersDetected[i].model.pose->setTvect(T3);
    }
}

void MarkerDetector::checkOrientation(std::vector<cv::DMatch> &matches, Model &_model)
{
    vector<vector<int>> rotHist(30);
    for (auto &v : rotHist)
        v.reserve(500);
    const float factor = 1.0f / float(rotHist.size());
    for (size_t midx = 0; midx < matches.size(); midx++)
    {
        const auto &match = matches[midx];
        float rot = _model.elements[match.queryIdx].inputKeyPoint.angle - keypoints_InputImage[match.trainIdx].angle;
        if (rot < 0.0)
            rot += 360.0f;
        size_t bin = round(rot * factor);
        if (bin == rotHist.size())
            bin = 0;
        assert(bin >= 0 && bin < rotHist.size());
        rotHist[bin].push_back(midx);
    }

    int ind1 = -1, ind2 = -1, ind3 = -1;
    ucoslam_FM___computeThreeMaxima(rotHist, ind1, ind2, ind3);
    for (int i = 0; i < int(rotHist.size()); i++)
    {
        if (i == ind1 || i == ind2 || i == ind3)
            continue;
        for (auto midx : rotHist[i])
            matches[midx].queryIdx = matches[midx].trainIdx = -1; // mark as unused
    }
    matches.erase(
        std::remove_if(matches.begin(), matches.end(), [](const cv::DMatch &m) { return m.trainIdx == -1 || m.queryIdx == -1; }),
        matches.end());
}

void MarkerDetector::Erosion()
{
    int erosion_type;
    if (erosion_elem == 0)
    {
        erosion_type = cv::MORPH_RECT;
    }
    else if (erosion_elem == 1)
    {
        erosion_type = cv::MORPH_CROSS;
    }
    else
    {
        erosion_type = cv::MORPH_ELLIPSE;
    }

    cv::Mat element = getStructuringElement(erosion_type, cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                                            cv::Point(erosion_size, erosion_size));

    /// Apply the erosion operation
    erode(TheInputImageThresholding, TheInputImageThresholding, element);
}

void MarkerDetector::Dilation()
{
    int dilation_type;
    if (dilation_elem == 0)
    {
        dilation_type = cv::MORPH_RECT;
    }
    else if (dilation_elem == 1)
    {
        dilation_type = cv::MORPH_CROSS;
    }
    else
    {
        dilation_type = cv::MORPH_ELLIPSE;
    }

    cv::Mat element = getStructuringElement(dilation_type, cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
                                            cv::Point(dilation_size, dilation_size));
    /// Apply the dilation operation
    dilate(TheInputImageThresholding, TheInputImageThresholding, element);
}

void MarkerDetector::getThresholdedImage()
{
    if (threshold_type == 0)
    {
        threshold(TheInputImageGrey, TheInputImageThresholding, threshold_value, max_threshold_value, cv::THRESH_BINARY);
    }
    else
    {
        if (Adaptive_Block_Size % 2 == 0)
            Adaptive_Block_Size++;
        adaptiveThreshold(TheInputImageGrey, TheInputImageThresholding, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY,
                          Adaptive_Block_Size, Adaptive_Threshold_Value);
    }

    // cv::imshow("Thresholding", resize(TheInputImageThresholding ,720));
}

/// DETECT AND TRACK
bool MarkerDetector::detectAndTrack(const cv::Mat &InputImage, CameraParameters &cameraParameters, float markerSizeMeters)
{
    TheInputImage = InputImage;

    if (VuMark)
        markerbitsData = 1;

    // Grayscale Image:
    cvtColor(TheInputImage, TheInputImageGrey, cv::COLOR_BGR2GRAY);

    if (!enableTracking)
    {
        modeInfo = "DETECT";
        if (detect(cameraParameters, markerSizeMeters))
        {
            // Enable tracking
            if (!only_detect)
                enableTracking = true;
            timesCallsDetect++;

            // Init all models markers
            pDerived.detectAndCompute(TheInputImageGrey, cv::Mat(), keypoints_InputImage, descriptors_InputImage, params);

            for (auto &marker : markersDetected)
                marker.initModel(keypoints_InputImage, descriptors_InputImage, cameraParameters, SVGFileData);
        }
    }

    else
    {
        modeInfo = "TRACK";
        if (!track(cameraParameters))
        {
            enableTracking = false;
            // Recall detect
            modeInfo = "DETECT";

            if (detect(cameraParameters, markerSizeMeters))
            {
                // Enable tracking
                if (!only_detect)
                    enableTracking = true;

                timesCallsDetect++;

                // Init all models markers
                pDerived.detectAndCompute(TheInputImageGrey, cv::Mat(), keypoints_InputImage, descriptors_InputImage, params);

                for (auto &marker : markersDetected)
                    marker.initModel(keypoints_InputImage, descriptors_InputImage, cameraParameters, SVGFileData);
            }
        }
        else
        {
            if (only_detect)
                enableTracking = false;
        }
    }
    if (!markersDetected.empty())
        markerCenter();

    return true;
}

/// DETECT
bool MarkerDetector::detect(CameraParameters &cameraParameters, float markerSizeMeters)
{
    filterContours.clear();
    markersDetected.clear();

    std::vector<cv::Point2f> corners;

    // Thresholding
    getThresholdedImage();

    // Erosion and Dilation
    Erosion();
    Dilation();

    // Find all contours
    cv::findContours(TheInputImageThresholding, allContour, cv::noArray(), cv::RETR_LIST, cv::CHAIN_APPROX_NONE);

    // Filter contours
    filterContours = contoursFilter(allContour, SVGFileData.border.size(), SVGFileData.convexPolygon);

    // For each contour checked
    for (auto &contour : filterContours)
    {
        cv::Mat homography;
        if (SVGFileData.symmetry)
        {
            std::vector<cv::Mat> Homographies = calculeHomographySymmetry(SVGFileData.border, contour);

            for (auto &homo : Homographies)
            {
                string idCode, crcImage;
                getCRCandId(homo, crcImage, idCode);

                // Check idCode is not empy
                if (!idCode.empty())
                {
                    createMarker(contour, cameraParameters, homo, idCode, crcImage, markerSizeMeters);
                }
            }
        }
        else
        {
            // Calcule homography for each filtercontours
            homography = calculeHomography(SVGFileData.border, contour);

            // Calcule id and crc in image
            string idCode, crcImage;
            getCRCandId(homography, crcImage, idCode);
            // Check idCode is not empy
            if (!idCode.empty())
                createMarker(contour, cameraParameters, homography, idCode, crcImage, markerSizeMeters);
        }
    }

    cout << "\rTime in check element value for all contour = " << Fps.getAvrg() * 1000 << " milliseconds" << endl;

    if (idMarkerFilter != 0)
    {
        markersDetected.erase(
            std::remove_if(markersDetected.begin(), markersDetected.end(), [](Marker x) { return x.id != idMarkerFilter; }),
            markersDetected.end());
    }

    if (!markersDetected.empty())
    {
        /// cv::destroyWindow("Thresholding");
        return true;
    }

    return false;
}

/// TRACK
bool MarkerDetector::track(CameraParameters &cameraParameters)
{

    vector<cv::DMatch> matches;

    if (thresHoldingDebug)
    {
        getThresholdedImage();
        Erosion();
        Dilation();
    }

    // Debug draw keypoint
    if (printInfo)
    {
        cv::drawKeypoints(TheInputImage, keypoints_InputImage, TheInputImage, cv::Scalar::all(-1),
                          cv::DrawMatchesFlags::DRAW_OVER_OUTIMG);
        cout << "Detected Keypoints " << keypoints_InputImage.size() << endl;
    }

    /// Detect Keypoints in input image
    pDerived.detectAndCompute(TheInputImageGrey, cv::Mat(), keypoints_InputImage, descriptors_InputImage, params);
    /// Create and build Kdtree
    picoflann::KdTreeIndex<2, PicoFlann_KeyPointAdapter> kdtree;
    kdtree.build(keypoints_InputImage);

    for (auto &marker : markersDetected)
    {
        std::vector<cv::Point3f> pointModel;
        std::vector<cv::Point2f> proyectedPoints;

        for (auto &p : marker.model.elements)
            pointModel.push_back(cv::Point3f(p.SVGPoint.x, p.SVGPoint.y, 0.0f));

        if (printInfo)
            cout << "Point proyected: " << pointModel.size() << endl;

        string nameWindow = "Matches " + to_string(marker.id);

        for (auto &r : radius)
        {
            /// Proyect 3D point SVGModel2InputImage
            proyectedPoints = marker.model.pose->project(pointModel);

            /// Update bounding rectangles
            marker.updateBoundingBox(proyectedPoints);

            /// Find matches SVGModel2InputImage
            matches = findMatches(kdtree, proyectedPoints, r, marker);

            if (printInfo)
                cout << "Matches found: " << matches.size() << endl;

            /// Filter match
            filter_ambiguous_train(matches);
            checkOrientation(matches, marker.model);

            /// Refine pose Rvec, Tvec, homography and calcule inliers matches
            marker.model.pose->optimize(matches, cameraParameters, keypoints_InputImage, marker.model.elements, TheInputImageGrey,
                                        SVGFileData.border);

            if (printInfo)
                cout << "Matches after refinePose: " << matches.size() << endl;

            /// Update model element which are been matched
            for (auto &match : matches)
                marker.model.elements.at(match.queryIdx).setAsMatched(true);
        }
        /// Set minimun matches
        if (matches.size() < minMatches)
            return false;

        /// Check if is necessary to update model
        const double percentPointMatches = matches.size() * 1.0f / proyectedPoints.size() * 1.0f;

        if (percentPointMatches > percent2UpdateModel)
            modelupdate = false;
        else
            modelupdate = true;

        /// Update model
        if (modelupdate)
            marker.updateModel(matches, descriptors_InputImage, keypoints_InputImage, SVGFileData);
    }

    return true;
}

/// FIND MATCHES
std::vector<cv::DMatch> MarkerDetector::findMatches(picoflann::KdTreeIndex<2, PicoFlann_KeyPointAdapter> &kdtree,
                                                    const vector<cv::Point2f> &proyectedPoint, const int _radius,
                                                    const Marker &marker)
{
    std::vector<cv::DMatch> matches;
    const int numNeighbors = 10;
    const int radius = _radius * _radius;
    const float hammingDistance = 0.7; /*hamming distance between two nearest keypoint*/

    for (int j = 0; j < proyectedPoint.size(); j++)
    {
        cv::KeyPoint idMatchesDebugKeypoint;
        idMatchesDebugKeypoint.pt = proyectedPoint[j];
        // Search knn neighbor of proyectedPoint[j] in vector keypoints2f_InputImage
        // .First [uint32_t] is index in keypoints_InputImage and .Second [double] is euclidean distance
        const std::vector<std::pair<uint32_t, double>> keyPairs =
            kdtree.searchKnn(keypoints_InputImage, idMatchesDebugKeypoint, numNeighbors);

        // .First [uint32_t] is index in keypoints_InputImage and .Second [double] is hamming distance
        std::pair<uint32_t, double> firstNearestNeighbor(-1, std::numeric_limits<double>::max());
        std::pair<uint32_t, double> secondNearestNeighbor(-1, std::numeric_limits<double>::max());

        // Assert find nearest point in inputImage
        bool pointFound = false;
        // Search two minimal keypoint
        for (auto &KeyPair : keyPairs)
        {
            // Check if keypoint is inside bounding marker
            if (!insideBoundingBox(keypoints_InputImage[KeyPair.first].pt, marker.boundingBox))
                continue;

            // Check euclidean distance
            if (KeyPair.second < radius)
            {
                // Check Hamming distance
                const float hammDistance =
                    getHammDescDistance(marker.model.elements[j].descriptor, descriptors_InputImage.row(KeyPair.first));

                // Calcule two minimal hamming distance
                if (hammDistance < firstNearestNeighbor.second)
                {
                    // Update secondNearestNeighbor
                    secondNearestNeighbor.second = firstNearestNeighbor.second;
                    secondNearestNeighbor.first = firstNearestNeighbor.first;
                    // Update firstNearestNeighbor
                    firstNearestNeighbor.second = hammDistance;
                    firstNearestNeighbor.first = KeyPair.first;
                }

                else if (hammDistance < secondNearestNeighbor.second && hammDistance != firstNearestNeighbor.second)
                {
                    // Update secondNearestNeighbor
                    secondNearestNeighbor.second = hammDistance;
                    secondNearestNeighbor.first = KeyPair.first; // keyPairs[i].second = Euclidean distance;
                }
            }

            pointFound = true;
        }

        // Compare hamming distance between two nearest keypoint
        if ((firstNearestNeighbor.second + 0.1) / (secondNearestNeighbor.second + 0.1) > hammingDistance || !pointFound)
            continue;

        // Match SVG2InputImage
        matches.push_back(cv::DMatch(j, firstNearestNeighbor.first,
                                     firstNearestNeighbor.second)); // DMatch(int _queryIdx, int _trainIdx, float _distance)
    }
    return matches;
}
