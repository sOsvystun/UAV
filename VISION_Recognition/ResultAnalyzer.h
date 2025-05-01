#ifndef RESULT_ANALYZER_H
#define RESULT_ANALYZER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct DetectionResult;

class ResultAnalyzer {
public:
    ResultAnalyzer();

    void analyzeResults(const std::vector<DetectionResult>& detections, const cv::Mat& image);
    void saveAnalysisReport(const std::string& outputPath);

private:
    std::vector<DetectionResult> detections;
    cv::Mat annotatedImage;

    void annotateImage();
    std::string generateStatistics();
};

#endif