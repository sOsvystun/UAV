#include "image_processor.h"

ImageProcessor::ImageProcessor() {}

cv::Mat ImageProcessor::preprocess(const cv::Mat& image) {
    cv::Mat processedImage;
    cv::resize(image, processedImage, cv::Size(640, 480));
    processedImage = enhanceContrast(processedImage);
    return processedImage;
}

cv::Mat ImageProcessor::combineRgbAndThermal(const cv::Mat& rgbImage, const cv::Mat& thermalImage) {
    cv::Mat alignedThermal = alignImages(rgbImage, thermalImage);
    cv::Mat combined;
    cv::addWeighted(rgbImage, 0.6, alignedThermal, 0.4, 0, combined);
    return combined;
}

std::vector<cv::Mat> ImageProcessor::augmentImage(const cv::Mat& image) {
    std::vector<cv::Mat> augmentedImages;

    augmentedImages.push_back(image);

    cv::Mat flipped;
    cv::flip(image, flipped, 1);
    augmentedImages.push_back(flipped);

    cv::Mat rotated;
    cv::rotate(image, rotated, cv::ROTATE_90_CLOCKWISE);
    augmentedImages.push_back(rotated);

    return augmentedImages;
}

cv::Mat ImageProcessor::enhanceContrast(const cv::Mat& image) {
    cv::Mat enhanced;
    cv::cvtColor(image, enhanced, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(enhanced, enhanced);
    cv::cvtColor(enhanced, enhanced, cv::COLOR_GRAY2BGR);
    return enhanced;
}

cv::Mat ImageProcessor::alignImages(const cv::Mat& rgbImage, const cv::Mat& thermalImage) {
    cv::Mat aligned;
    // Placeholder for alignment logic
    cv::resize(thermalImage, aligned, rgbImage.size());
    return aligned;
}