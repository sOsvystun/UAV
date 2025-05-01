#ifndef MODEL_ENSEMBLE_H
#define MODEL_ENSEMBLE_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>
#include "detectron_handler.h"

struct DetectionResult {
    int classId;
    cv::Rect bbox;
    float confidence;
};

class ModelEnsemble {
public:
    ModelEnsemble(const std::vector<std::string>& detectronModels, const std::string& yoloModelPath);
    
    std::vector<DetectionResult> performEnsembleDetection(const cv::Mat& image);

private:
    std::vector<DetectronHandler> detectronHandlers;
    cv::dnn::Net yoloNet;
    std::vector<std::string> classNames;

    std::vector<DetectionResult> detectWithYOLO(const cv::Mat& image);
    std::vector<DetectionResult> combineDetections(const std::vector<std::vector<DetectionResult>>& allDetections);
    
    float computeIoU(const cv::Rect& boxA, const cv::Rect& boxB);
    float combineProbabilities(float p1, float p2, float gamma);
    cv::Rect combineBoundingBoxes(const cv::Rect& boxA, const cv::Rect& boxB, float confA, float confB);
};

#endif