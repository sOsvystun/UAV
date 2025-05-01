#include "result_analyzer.h"
#include "model_ensemble.h"
#include <fstream>
#include <sstream>

ResultAnalyzer::ResultAnalyzer() {}

void ResultAnalyzer::analyzeResults(const std::vector<DetectionResult>& detections, const cv::Mat& image) {
    this->detections = detections;
    annotatedImage = image.clone();
    annotateImage();
}

void ResultAnalyzer::annotateImage() {
    for (const auto& detection : detections) {
        cv::rectangle(annotatedImage, detection.bbox, cv::Scalar(0, 255, 0), 2);
        std::ostringstream label;
        label << "Class: " << detection.classId << ", Conf: " << detection.confidence;
        cv::putText(annotatedImage, label.str(),
                    detection.bbox.tl(), cv::FONT_HERSHEY_SIMPLEX,
                    0.5, cv::Scalar(0, 255, 0), 1);
    }
}

std::string ResultAnalyzer::generateStatistics() {
    std::ostringstream report;
    std::map<int, int> classCounts;
    for (const auto& detection : detections) {
        classCounts[detection.classId]++;
    }

    report << "Detection Report:\n";
    for (const auto& [classId, count] : classCounts) {
        report << "Class " << classId << ": " << count << " occurrences\n";
    }

    return report.str();
}

void ResultAnalyzer::saveAnalysisReport(const std::string& outputPath) {
    std::ofstream reportFile(outputPath + "/analysis_report.txt");
    reportFile << generateStatistics();
    reportFile.close();

    cv::imwrite(outputPath + "/annotated_image.jpg", annotatedImage);
}