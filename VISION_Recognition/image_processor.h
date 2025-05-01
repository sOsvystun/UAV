#ifndef IMAGE_PROCESSOR_H
#define IMAGE_PROCESSOR_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class ImageProcessor {
public:
    ImageProcessor();

    cv::Mat preprocess(const cv::Mat& image);
    cv::Mat combineRgbAndThermal(const cv::Mat& rgbImage, const cv::Mat& thermalImage);
    std::vector<cv::Mat> augmentImage(const cv::Mat& image);

private:
    cv::Mat enhanceContrast(const cv::Mat& image);
    cv::Mat alignImages(const cv::Mat& rgbImage, const cv::Mat& thermalImage);
};

#endif // IMAGE_PROCESSOR_H