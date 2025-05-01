#include "model_ensemble.h"
#include "image_processor.h"
#include "result_analyzer.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <pqxx/pqxx>

extern "C" void runEnsembleDetection(const char* rgbPath, const char* thermalPath, const char* outputDir) {
    std::vector<std::string> detectronModels = {
        "model1.pth",
        "model2.pth"
    };
    std::string yoloModelPath = "yolov8.onnx";

    ModelEnsemble ensemble(detectronModels, yoloModelPath);
    ImageProcessor processor;
    ResultAnalyzer analyzer;

    cv::Mat rgbImage = cv::imread(rgbPath);
    cv::Mat thermalImage = cv::imread(thermalPath);

    if (rgbImage.empty() || thermalImage.empty()) {
        std::cerr << "Error: Cannot load images." << std::endl;
        return;
    }

    cv::Mat combinedImage = processor.combineRgbAndThermal(rgbImage, thermalImage);
    cv::Mat preprocessedImage = processor.preprocess(combinedImage);

    auto detections = ensemble.performEnsembleDetection(preprocessedImage);

    analyzer.analyzeResults(detections, combinedImage);
    analyzer.saveAnalysisReport(outputDir);

    std::cout << "Detection and analysis completed successfully." << std::endl;
}

void storeResultsInDatabase(const std::vector<DetectionResult>& detections) {
    try {
        pqxx::connection conn(".....port=5432");

        if (conn.is_open()) {
            pqxx::work txn(conn);

            for (const auto& detection : detections) {
                txn.exec0("INSERT INTO detections (class_id, confidence, bbox_x, bbox_y, bbox_width, bbox_height) VALUES (" +
                          txn.quote(detection.classId) + ", " +
                          txn.quote(detection.confidence) + ", " +
                          txn.quote(detection.bbox.x) + ", " +
                          txn.quote(detection.bbox.y) + ", " +
                          txn.quote(detection.bbox.width) + ", " +
                          txn.quote(detection.bbox.height) + ");");
            }

            txn.commit();
            std::cout << "Results successfully stored in database." << std::endl;
        } else {
            std::cerr << "Unable to open database connection." << std::endl;
        }

        conn.disconnect();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
}

int main() {
    std::cout << "Initializing database connection..." << std::endl;

    const char* rgbPath = "rgb.jpg";
    const char* thermalPath = "thermal.jpg";
    const char* outputDir = "output";

    runEnsembleDetection(rgbPath, thermalPath, outputDir);

    std::cout << "Closing database connection." << std::endl;

    return 0;
}