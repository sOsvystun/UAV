#include "enhanced_image_processor.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <fstream>

EnhancedImageProcessor::EnhancedImageProcessor() 
    : camera_calibrated_(false), photogrammetric_initialized_(false) {
    // Initialize default camera parameters
    camera_params_.focal_length_mm = 24.0;  // Default focal length
    camera_params_.pixel_size_um = 5.5;     // Default pixel size
    camera_params_.sensor_size = cv::Size(1920, 1080);  // Default sensor size
}

EnhancedImageProcessor::~EnhancedImageProcessor() {
    // Cleanup if needed
}

void EnhancedImageProcessor::setCameraParameters(const CameraParameters& params) {
    camera_params_ = params;
    camera_calibrated_ = true;
}

void EnhancedImageProcessor::setPhotogrammetricData(const PhotogrammetricData& data) {
    photogrammetric_data_ = data;
    photogrammetric_initialized_ = true;
}

cv::Mat EnhancedImageProcessor::preprocessImage(const cv::Mat& image) {
    if (image.empty()) {
        std::cerr << "Error: Empty input image" << std::endl;
        return cv::Mat();
    }
    
    cv::Mat processed = image.clone();
    
    // Step 1: Undistort image if camera is calibrated
    if (camera_calibrated_ && !camera_params_.camera_matrix.empty()) {
        processed = undistortImage(processed);
    }
    
    // Step 2: Apply bilateral filter for noise reduction while preserving edges
    processed = applyBilateralFilter(processed);
    
    // Step 3: Apply CLAHE for contrast enhancement
    processed = applyCLAHE(processed);
    
    return processed;
}

cv::Mat EnhancedImageProcessor::combineRgbAndThermal(const cv::Mat& rgbImage, const cv::Mat& thermalImage) {
    if (rgbImage.empty() || thermalImage.empty()) {
        std::cerr << "Error: Empty input images for fusion" << std::endl;
        return cv::Mat();
    }
    
    // Align thermal image to RGB image
    cv::Mat alignedThermal = alignThermalToRGB(rgbImage, thermalImage);
    
    // Resize thermal to match RGB if needed
    if (alignedThermal.size() != rgbImage.size()) {
        cv::resize(alignedThermal, alignedThermal, rgbImage.size());
    }
    
    // Convert thermal to 3-channel if it's single channel
    cv::Mat thermalColor;
    if (alignedThermal.channels() == 1) {
        cv::applyColorMap(alignedThermal, thermalColor, cv::COLORMAP_JET);
    } else {
        thermalColor = alignedThermal;
    }
    
    // Perform weighted fusion
    cv::Mat fusedImage;
    double alpha = 0.7;  // Weight for RGB
    double beta = 0.3;   // Weight for thermal
    cv::addWeighted(rgbImage, alpha, thermalColor, beta, 0, fusedImage);
    
    return fusedImage;
}

GeometricFeatures EnhancedImageProcessor::extractGeometricFeatures(const cv::Mat& image, const DefectROI& roi) {
    GeometricFeatures features;
    
    if (!validateROI(roi.bounding_box, image.size())) {
        std::cerr << "Error: Invalid ROI for geometric feature extraction" << std::endl;
        return features;
    }
    
    // Extract ROI
    cv::Mat roiImage = cropROI(image, roi.bounding_box);
    
    // Preprocess ROI
    cv::Mat processed = preprocessImage(roiImage);
    
    // Convert to grayscale if needed
    cv::Mat gray;
    if (processed.channels() == 3) {
        cv::cvtColor(processed, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = processed;
    }
    
    // Apply adaptive thresholding
    cv::Mat binary = adaptiveThresholding(gray);
    
    // Apply morphological filtering
    cv::Mat filtered = morphologicalFiltering(binary);
    
    // Find largest contour (assumed to be the defect)
    std::vector<cv::Point> contour = findLargestContour(filtered);
    
    if (contour.empty()) {
        std::cerr << "Warning: No contour found in ROI" << std::endl;
        return features;
    }
    
    // Calculate geometric features
    features.area_pixels = computeContourArea(contour);
    features.perimeter_real_mm = computeContourPerimeter(contour);
    features.centroid = computeContourCentroid(contour);
    features.contour = contour;
    
    // Calculate bounding rectangle features
    cv::Rect boundingRect = cv::boundingRect(contour);
    features.aspect_ratio = computeAspectRatio(boundingRect);
    features.solidity = computeSolidity(contour);
    
    // Calculate curvature
    features.curvature_avg = calculateCurvature(contour);
    
    // Convert to real-world measurements if photogrammetric data is available
    if (photogrammetric_initialized_) {
        double scalingFactor = photogrammetric_data_.scaling_factor;
        features.area_real_mm2 = calculateRealWorldArea(features.area_pixels, scalingFactor);
        features.length_real_mm = calculateRealWorldLength(boundingRect.height, scalingFactor);
        features.width_avg_mm = calculateRealWorldLength(boundingRect.width, scalingFactor);
        features.perimeter_real_mm = features.perimeter_real_mm * scalingFactor;
    } else {
        // Use default scaling if photogrammetric data not available
        double defaultScaling = 0.1;  // Assume 0.1 mm/pixel
        features.area_real_mm2 = features.area_pixels * defaultScaling * defaultScaling;
        features.length_real_mm = boundingRect.height * defaultScaling;
        features.width_avg_mm = boundingRect.width * defaultScaling;
        features.perimeter_real_mm = features.perimeter_real_mm * defaultScaling;
    }
    
    return features;
}

ThermalFeatures EnhancedImageProcessor::extractThermalFeatures(const cv::Mat& thermalImage, const DefectROI& roi) {
    ThermalFeatures features;
    
    if (!validateThermalImage(thermalImage) || !validateROI(roi.bounding_box, thermalImage.size())) {
        std::cerr << "Error: Invalid thermal image or ROI" << std::endl;
        return features;
    }
    
    // Extract thermal ROI
    cv::Mat thermalROI = cropROI(thermalImage, roi.bounding_box);
    
    // Calculate temperature differences
    features.delta_t_max = calculateTemperatureDifference(thermalImage, roi.bounding_box);
    
    // Calculate average temperature difference
    cv::Scalar meanTemp = cv::mean(thermalROI);
    cv::Scalar backgroundMean = cv::mean(thermalImage);  // Simplified background estimation
    features.delta_t_avg = std::abs(meanTemp[0] - backgroundMean[0]);
    
    // Calculate thermal gradient (Laplacian)
    features.thermal_gradient = calculateThermalGradient(thermalImage, roi.bounding_box);
    
    // Calculate thermal contrast
    features.thermal_contrast = calculateThermalContrast(thermalImage, roi.bounding_box);
    
    // Find hotspot center
    cv::Point maxLoc;
    cv::minMaxLoc(thermalROI, nullptr, nullptr, nullptr, &maxLoc);
    features.hotspot_center = cv::Point2f(maxLoc.x + roi.bounding_box.x, 
                                         maxLoc.y + roi.bounding_box.y);
    
    // Calculate thermal area ratio
    cv::Mat thermalMask;
    double threshold = backgroundMean[0] + 2.0;  // 2°C above background
    cv::threshold(thermalROI, thermalMask, threshold, 255, cv::THRESH_BINARY);
    features.thermal_area_ratio = cv::countNonZero(thermalMask) / static_cast<double>(thermalROI.total());
    
    return features;
}

cv::Mat EnhancedImageProcessor::applyBilateralFilter(const cv::Mat& image, int d, 
                                                    double sigmaColor, double sigmaSpace) {
    cv::Mat filtered;
    cv::bilateralFilter(image, filtered, d, sigmaColor, sigmaSpace);
    return filtered;
}

cv::Mat EnhancedImageProcessor::applyCLAHE(const cv::Mat& image, double clipLimit, 
                                          cv::Size tileGridSize) {
    cv::Mat enhanced;
    
    if (image.channels() == 1) {
        // Grayscale image
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clipLimit, tileGridSize);
        clahe->apply(image, enhanced);
    } else {
        // Color image - apply CLAHE to L channel in LAB color space
        cv::Mat lab;
        cv::cvtColor(image, lab, cv::COLOR_BGR2Lab);
        
        std::vector<cv::Mat> labChannels;
        cv::split(lab, labChannels);
        
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clipLimit, tileGridSize);
        clahe->apply(labChannels[0], labChannels[0]);
        
        cv::merge(labChannels, lab);
        cv::cvtColor(lab, enhanced, cv::COLOR_Lab2BGR);
    }
    
    return enhanced;
}

cv::Mat EnhancedImageProcessor::adaptiveThresholding(const cv::Mat& image, int blockSize, double C) {
    cv::Mat binary;
    cv::adaptiveThreshold(image, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, 
                         cv::THRESH_BINARY, blockSize, C);
    return binary;
}

cv::Mat EnhancedImageProcessor::morphologicalFiltering(const cv::Mat& binaryImage, 
                                                      cv::MorphShapes shape, cv::Size kernelSize) {
    cv::Mat kernel = cv::getStructuringElement(shape, kernelSize);
    cv::Mat eroded, dilated;
    
    // Erosion followed by dilation (opening operation)
    cv::erode(binaryImage, eroded, kernel);
    cv::dilate(eroded, dilated, kernel);
    
    return dilated;
}

std::vector<cv::Point> EnhancedImageProcessor::findLargestContour(const cv::Mat& binaryImage) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binaryImage, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    if (contours.empty()) {
        return std::vector<cv::Point>();
    }
    
    // Find the largest contour by area
    double maxArea = 0;
    int maxAreaIdx = 0;
    
    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area > maxArea) {
            maxArea = area;
            maxAreaIdx = i;
        }
    }
    
    return contours[maxAreaIdx];
}

double EnhancedImageProcessor::calculateCurvature(const std::vector<cv::Point>& contour) {
    if (contour.size() < 3) {
        return 0.0;
    }
    
    double totalCurvature = 0.0;
    int validPoints = 0;
    
    for (size_t i = 1; i < contour.size() - 1; i++) {
        cv::Point p1 = contour[i - 1];
        cv::Point p2 = contour[i];
        cv::Point p3 = contour[i + 1];
        
        // Calculate vectors
        cv::Point v1 = p2 - p1;
        cv::Point v2 = p3 - p2;
        
        // Calculate cross product and dot product
        double cross = v1.x * v2.y - v1.y * v2.x;
        double dot = v1.x * v2.x + v1.y * v2.y;
        
        // Calculate lengths
        double len1 = cv::norm(v1);
        double len2 = cv::norm(v2);
        
        if (len1 > 0 && len2 > 0) {
            // Calculate curvature using the formula: κ = |v1 × v2| / |v1|³
            double curvature = std::abs(cross) / (len1 * len1 * len1);
            totalCurvature += curvature;
            validPoints++;
        }
    }
    
    return validPoints > 0 ? totalCurvature / validPoints : 0.0;
}

double EnhancedImageProcessor::calculateRealWorldArea(double pixelArea, double scalingFactor) {
    return pixelArea * scalingFactor * scalingFactor;
}

double EnhancedImageProcessor::calculateRealWorldLength(double pixelLength, double scalingFactor) {
    return pixelLength * scalingFactor;
}

double EnhancedImageProcessor::calculateTemperatureDifference(const cv::Mat& thermalImage, const cv::Rect& roi) {
    cv::Mat thermalROI = thermalImage(roi);
    
    // Find min and max temperatures in ROI
    double minTemp, maxTemp;
    cv::minMaxLoc(thermalROI, &minTemp, &maxTemp);
    
    // Calculate background temperature (simplified as image mean excluding ROI)
    cv::Mat mask = cv::Mat::zeros(thermalImage.size(), CV_8UC1);
    cv::rectangle(mask, roi, cv::Scalar(255), -1);
    
    cv::Mat backgroundMask = ~mask;
    cv::Scalar backgroundMean = cv::mean(thermalImage, backgroundMask);
    
    // Return maximum temperature difference
    return std::max(std::abs(maxTemp - backgroundMean[0]), 
                   std::abs(minTemp - backgroundMean[0]));
}

double EnhancedImageProcessor::calculateThermalGradient(const cv::Mat& thermalImage, const cv::Rect& roi) {
    cv::Mat thermalROI = thermalImage(roi);
    
    // Convert to floating point for gradient calculation
    cv::Mat floatROI;
    thermalROI.convertTo(floatROI, CV_32F);
    
    // Calculate Laplacian (second derivative)
    cv::Mat laplacian;
    cv::Laplacian(floatROI, laplacian, CV_32F);
    
    // Calculate mean absolute Laplacian
    cv::Scalar meanLaplacian = cv::mean(cv::abs(laplacian));
    
    return meanLaplacian[0];
}

double EnhancedImageProcessor::calculateThermalContrast(const cv::Mat& thermalImage, const cv::Rect& roi) {
    cv::Mat thermalROI = thermalImage(roi);
    
    // Calculate standard deviation as a measure of contrast
    cv::Scalar mean, stddev;
    cv::meanStdDev(thermalROI, mean, stddev);
    
    return stddev[0];
}

double EnhancedImageProcessor::calculateScalingFactor(double distanceToTarget, double focalLength, double pixelSize) {
    // Formula: scaling_factor = (distance * pixel_size) / focal_length
    // This gives mm per pixel
    return (distanceToTarget * 1000.0 * pixelSize / 1000.0) / focalLength;
}

cv::Point3f EnhancedImageProcessor::convertToWorldCoordinates(const cv::Point2f& imagePoint, double depth) {
    if (!camera_calibrated_) {
        std::cerr << "Warning: Camera not calibrated, returning approximate coordinates" << std::endl;
        return cv::Point3f(imagePoint.x, imagePoint.y, depth);
    }
    
    // Convert image coordinates to world coordinates using camera parameters
    double fx = camera_params_.camera_matrix.at<double>(0, 0);
    double fy = camera_params_.camera_matrix.at<double>(1, 1);
    double cx = camera_params_.camera_matrix.at<double>(0, 2);
    double cy = camera_params_.camera_matrix.at<double>(1, 2);
    
    double x = (imagePoint.x - cx) * depth / fx;
    double y = (imagePoint.y - cy) * depth / fy;
    
    return cv::Point3f(x, y, depth);
}

cv::Mat EnhancedImageProcessor::alignThermalToRGB(const cv::Mat& rgbImage, const cv::Mat& thermalImage) {
    // Find homography between thermal and RGB images
    cv::Mat homography = findHomographyBetweenImages(thermalImage, rgbImage);
    
    if (homography.empty()) {
        // If homography fails, return resized thermal image
        cv::Mat resized;
        cv::resize(thermalImage, resized, rgbImage.size());
        return resized;
    }
    
    // Warp thermal image to align with RGB
    cv::Mat aligned;
    cv::warpPerspective(thermalImage, aligned, homography, rgbImage.size());
    
    return aligned;
}

cv::Mat EnhancedImageProcessor::findHomographyBetweenImages(const cv::Mat& img1, const cv::Mat& img2) {
    // Convert to grayscale if needed
    cv::Mat gray1, gray2;
    if (img1.channels() == 3) {
        cv::cvtColor(img1, gray1, cv::COLOR_BGR2GRAY);
    } else {
        gray1 = img1;
    }
    
    if (img2.channels() == 3) {
        cv::cvtColor(img2, gray2, cv::COLOR_BGR2GRAY);
    } else {
        gray2 = img2;
    }
    
    // Detect keypoints and compute descriptors
    cv::Ptr<cv::ORB> detector = cv::ORB::create();
    
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    
    detector->detectAndCompute(gray1, cv::noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(gray2, cv::noArray(), keypoints2, descriptors2);
    
    if (keypoints1.size() < 4 || keypoints2.size() < 4) {
        std::cerr << "Warning: Insufficient keypoints for homography estimation" << std::endl;
        return cv::Mat();
    }
    
    // Match descriptors
    cv::BFMatcher matcher;
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors1, descriptors2, matches);
    
    if (matches.size() < 4) {
        std::cerr << "Warning: Insufficient matches for homography estimation" << std::endl;
        return cv::Mat();
    }
    
    // Extract matched points
    std::vector<cv::Point2f> points1, points2;
    for (const auto& match : matches) {
        points1.push_back(keypoints1[match.queryIdx].pt);
        points2.push_back(keypoints2[match.trainIdx].pt);
    }
    
    // Find homography
    cv::Mat homography = cv::findHomography(points1, points2, cv::RANSAC);
    
    return homography;
}

double EnhancedImageProcessor::assessImageSharpness(const cv::Mat& image) {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image;
    }
    
    // Calculate Laplacian variance as sharpness measure
    cv::Mat laplacian;
    cv::Laplacian(gray, laplacian, CV_64F);
    
    cv::Scalar mean, stddev;
    cv::meanStdDev(laplacian, mean, stddev);
    
    return stddev[0] * stddev[0];  // Variance
}

double EnhancedImageProcessor::assessImageBrightness(const cv::Mat& image) {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image;
    }
    
    cv::Scalar meanBrightness = cv::mean(gray);
    return meanBrightness[0] / 255.0;  // Normalized to 0-1
}

double EnhancedImageProcessor::assessImageContrast(const cv::Mat& image) {
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image;
    }
    
    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev);
    
    return stddev[0] / 255.0;  // Normalized to 0-1
}

bool EnhancedImageProcessor::validateImageQuality(const cv::Mat& image, double minSharpness, 
                                                 double minBrightness, double minContrast) {
    double sharpness = assessImageSharpness(image);
    double brightness = assessImageBrightness(image);
    double contrast = assessImageContrast(image);
    
    return (sharpness >= minSharpness && 
            brightness >= minBrightness && 
            contrast >= minContrast);
}

cv::Mat EnhancedImageProcessor::undistortImage(const cv::Mat& image) {
    if (!camera_calibrated_) {
        return image;
    }
    
    cv::Mat undistorted;
    cv::undistort(image, undistorted, camera_params_.camera_matrix, 
                 camera_params_.distortion_coeffs);
    
    return undistorted;
}

cv::Mat EnhancedImageProcessor::cropROI(const cv::Mat& image, const cv::Rect& roi) {
    // Ensure ROI is within image bounds
    cv::Rect safeROI = roi & cv::Rect(0, 0, image.cols, image.rows);
    return image(safeROI);
}

// Helper method implementations
double EnhancedImageProcessor::computeContourArea(const std::vector<cv::Point>& contour) {
    return cv::contourArea(contour);
}

double EnhancedImageProcessor::computeContourPerimeter(const std::vector<cv::Point>& contour) {
    return cv::arcLength(contour, true);
}

cv::Point2f EnhancedImageProcessor::computeContourCentroid(const std::vector<cv::Point>& contour) {
    cv::Moments moments = cv::moments(contour);
    if (moments.m00 != 0) {
        return cv::Point2f(moments.m10 / moments.m00, moments.m01 / moments.m00);
    }
    return cv::Point2f(0, 0);
}

double EnhancedImageProcessor::computeAspectRatio(const cv::Rect& boundingRect) {
    return static_cast<double>(boundingRect.width) / boundingRect.height;
}

double EnhancedImageProcessor::computeSolidity(const std::vector<cv::Point>& contour) {
    double contourArea = cv::contourArea(contour);
    
    std::vector<cv::Point> hull;
    cv::convexHull(contour, hull);
    double hullArea = cv::contourArea(hull);
    
    return hullArea > 0 ? contourArea / hullArea : 0.0;
}

bool EnhancedImageProcessor::validateROI(const cv::Rect& roi, const cv::Size& imageSize) {
    return (roi.x >= 0 && roi.y >= 0 && 
            roi.x + roi.width <= imageSize.width && 
            roi.y + roi.height <= imageSize.height &&
            roi.width > 0 && roi.height > 0);
}

bool EnhancedImageProcessor::validateThermalImage(const cv::Mat& thermalImage) {
    return !thermalImage.empty() && 
           (thermalImage.type() == CV_8UC1 || thermalImage.type() == CV_16UC1 || 
            thermalImage.type() == CV_32FC1);
}

// Utility functions for fuzzy logic integration
FuzzyInputParameters convertToFuzzyInputs(const GeometricFeatures& geoFeatures,
                                        const ThermalFeatures& thermalFeatures,
                                        const std::string& componentType,
                                        const std::string& defectType,
                                        double expertScore) {
    FuzzyInputParameters params;
    
    params.defect_size_mm2 = geoFeatures.area_real_mm2;
    params.thermal_signature_delta_t = thermalFeatures.delta_t_max;
    params.expert_score = expertScore;
    params.component_type = componentType;
    params.defect_type = defectType;
    
    // Location normalization would need blade-specific information
    // For now, use a simplified approach based on centroid position
    params.location_normalized = 0.5;  // Default to mid-span
    
    return params;
}

double normalizeBladeLocation(const cv::Point2f& position, double bladeLength) {
    // Simplified normalization - would need actual blade geometry
    return std::min(1.0, std::max(0.0, position.y / bladeLength));
}

std::string classifyComponent(const cv::Point2f& position, const GeometricFeatures& features) {
    // Simplified component classification based on position and features
    // In practice, this would use more sophisticated analysis
    
    if (features.location_normalized < 0.33) {
        return "blade_root";
    } else if (features.location_normalized < 0.67) {
        return "blade_mid_span";
    } else {
        return "blade_tip";
    }
}