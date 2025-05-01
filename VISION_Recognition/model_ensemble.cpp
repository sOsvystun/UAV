#include "model_ensemble.h"
#include <numeric>

ModelEnsemble::ModelEnsemble(const std::vector<std::string>& detectronModels, const std::string& yoloModelPath) {
    for (const auto& modelPath : detectronModels) {
        detectronHandlers.emplace_back(DetectronHandler(modelPath));
    }
    yoloNet = cv::dnn::readNetFromONNX(yoloModelPath);
    classNames = {"crack", "corrosion", "overheating"};
}

std::vector<DetectionResult> ModelEnsemble::performEnsembleDetection(const cv::Mat& image) {
    std::vector<std::vector<DetectionResult>> allDetections;

    for (auto& handler : detectronHandlers) {
        std::vector<cv::Rect> boxes = handler.detectObjects(image);
        std::vector<DetectionResult> results;
        for (auto& box : boxes) {
            results.push_back({0, box, 0.85f});
        }
        allDetections.push_back(results);
    }

    allDetections.push_back(detectWithYOLO(image));

    return combineDetections(allDetections);
}

std::vector<DetectionResult> ModelEnsemble::detectWithYOLO(const cv::Mat& image) {
    std::vector<DetectionResult> detections;
    cv::Mat blob;
    cv::dnn::blobFromImage(image, blob, 1.0/255, cv::Size(640, 640), cv::Scalar(), true);
    yoloNet.setInput(blob);
    cv::Mat output = yoloNet.forward();

    float xFactor = image.cols / 640.f;
    float yFactor = image.rows / 640.f;

    const float* data = (float*)output.data;
    int dimensions = output.size[2];

    for (int i = 0; i < output.size[1]; ++i) {
        float confidence = data[4];
        if (confidence > 0.6f) {
            cv::Mat scores(1, classNames.size(), CV_32FC1, (float*)data + 5);
            cv::Point classId;
            double maxScore;
            cv::minMaxLoc(scores, 0, &maxScore, 0, &classId);
            if (maxScore > 0.6f) {
                float cx = data[0], cy = data[1], w = data[2], h = data[3];
                int left = int((cx - 0.5f * w) * xFactor);
                int top = int((cy - 0.5f * h) * yFactor);
                detections.push_back({classId.x, cv::Rect(left, top, int(w*xFactor), int(h*yFactor)), confidence});
            }
        }
        data += dimensions;
    }

    return detections;
}

std::vector<DetectionResult> ModelEnsemble::combineDetections(const std::vector<std::vector<DetectionResult>>& allDetections) {
    std::vector<DetectionResult> ensembleResults;

    for (const auto& detections : allDetections[0]) {
        DetectionResult combinedResult = detections;

        for (size_t i = 1; i < allDetections.size(); ++i) {
            for (const auto& candidate : allDetections[i]) {
                float iou = computeIoU(combinedResult.bbox, candidate.bbox);
                if (iou > 0.5 && combinedResult.classId == candidate.classId) {
                    combinedResult.confidence = combineProbabilities(combinedResult.confidence, candidate.confidence, 0.5f);
                    combinedResult.bbox = combineBoundingBoxes(combinedResult.bbox, candidate.bbox, combinedResult.confidence, candidate.confidence);
                }
            }
        }

        ensembleResults.push_back(combinedResult);
    }

    return ensembleResults;
}

float ModelEnsemble::computeIoU(const cv::Rect& boxA, const cv::Rect& boxB) {
    float intersectionArea = (boxA & boxB).area();
    float unionArea = boxA.area() + boxB.area() - intersectionArea;
    return intersectionArea / unionArea;
}

float ModelEnsemble::combineProbabilities(float p1, float p2, float gamma) {
    return gamma * p1 + (1.0f - gamma) * p2;
}

cv::Rect ModelEnsemble::combineBoundingBoxes(const cv::Rect& boxA, const cv::Rect& boxB, float confA, float confB) {
    float totalConf = confA + confB;
    int x = (boxA.x * confA + boxB.x * confB) / totalConf;
    int y = (boxA.y * confA + boxB.y * confB) / totalConf;
    int w = (boxA.width * confA + boxB.width * confB) / totalConf;
    int h = (boxA.height * confA + boxB.height * confB) / totalConf;
    return cv::Rect(x, y, w, h);
}