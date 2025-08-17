#ifndef YOLO_ENSEMBLE_DETECTOR_H
#define YOLO_ENSEMBLE_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <string>
#include <memory>
#include "enhanced_image_processor.h"

/**
 * YOLOv8 Ensemble Detector implementing the detection pipeline from
 * "Criticality Assessment of Wind Turbine Defects via Multispectral UAV Fusion and Fuzzy Logic"
 * 
 * This class implements:
 * - YOLOv8 ensemble detection with multiple models
 * - Non-Maximum Suppression (NMS) for ensemble results
 * - Integration with multispectral image processing
 * - Confidence-based filtering and validation
 */

struct YOLODetection {
    cv::Rect bounding_box;       // Detection bounding box
    float confidence;            // Detection confidence score
    int class_id;               // Detected class ID
    std::string class_name;     // Human-readable class name
    cv::Point2f center;         // Center point of detection
    int model_id;               // ID of the model that made this detection
};

struct EnsembleDetection {
    cv::Rect bounding_box;       // Final ensemble bounding box
    float ensemble_confidence;   // Combined confidence from all models
    int class_id;               // Detected class ID
    std::string class_name;     // Human-readable class name
    std::vector<float> model_confidences;  // Individual model confidences
    int detection_count;        // Number of models that detected this object
    cv::Point2f center;         // Center point of detection
};

struct ModelConfiguration {
    std::string model_path;      // Path to ONNX model file
    std::string model_name;      // Human-readable model name
    float confidence_threshold;  // Minimum confidence threshold
    float nms_threshold;        // NMS threshold for this model
    cv::Size input_size;        // Model input size
    bool enabled;               // Whether this model is enabled
};

class YOLOEnsembleDetector {
public:
    YOLOEnsembleDetector();
    ~YOLOEnsembleDetector();
    
    // Configuration methods
    bool loadModel(const ModelConfiguration& config);
    bool loadModels(const std::vector<ModelConfiguration>& configs);
    void setClassNames(const std::vector<std::string>& classNames);
    void setEnsembleParameters(float confidenceThreshold, float nmsThreshold, int minDetectionCount = 1);
    
    // Detection methods
    std::vector<YOLODetection> detectSingleModel(const cv::Mat& image, int modelIndex);
    std::vector<EnsembleDetection> detectEnsemble(const cv::Mat& image);
    std::vector<EnsembleDetection> detectMultispectral(const cv::Mat& rgbImage, const cv::Mat& thermalImage);
    
    // Post-processing methods
    std::vector<YOLODetection> applyNMS(const std::vector<YOLODetection>& detections, float nmsThreshold);
    std::vector<EnsembleDetection> combineDetections(const std::vector<std::vector<YOLODetection>>& allDetections);
    std::vector<EnsembleDetection> filterByConfidence(const std::vector<EnsembleDetection>& detections, float threshold);
    
    // Validation and quality assessment
    bool validateDetection(const EnsembleDetection& detection, const cv::Mat& image);
    double calculateDetectionQuality(const EnsembleDetection& detection, const cv::Mat& image);
    std::vector<EnsembleDetection> rankDetectionsByQuality(const std::vector<EnsembleDetection>& detections, 
                                                          const cv::Mat& image);
    
    // Visualization methods
    cv::Mat visualizeDetections(const cv::Mat& image, const std::vector<EnsembleDetection>& detections);
    cv::Mat visualizeEnsembleConfidence(const cv::Mat& image, const std::vector<EnsembleDetection>& detections);
    void drawDetectionInfo(cv::Mat& image, const EnsembleDetection& detection);
    
    // Integration with image processor
    std::vector<DefectROI> convertToDefectROIs(const std::vector<EnsembleDetection>& detections);
    
    // Performance monitoring
    void enablePerformanceMonitoring(bool enable);
    void getPerformanceStats(double& avgInferenceTime, double& avgNMSTime, double& avgEnsembleTime);
    void resetPerformanceStats();
    
    // Model management
    bool isModelLoaded(int modelIndex) const;
    int getModelCount() const;
    std::string getModelInfo(int modelIndex) const;
    void enableModel(int modelIndex, bool enable);
    
    // Utility methods
    static cv::Mat preprocessImage(const cv::Mat& image, const cv::Size& targetSize);
    static std::vector<cv::Mat> createImagePyramid(const cv::Mat& image, int levels = 3);
    static cv::Rect scaleDetection(const cv::Rect& detection, float scaleX, float scaleY);
    
private:
    // Model storage and management
    std::vector<cv::dnn::Net> models_;
    std::vector<ModelConfiguration> model_configs_;
    std::vector<std::string> class_names_;
    std::vector<bool> model_enabled_;
    
    // Ensemble parameters
    float ensemble_confidence_threshold_;
    float ensemble_nms_threshold_;
    int min_detection_count_;
    
    // Performance monitoring
    bool performance_monitoring_enabled_;
    std::vector<double> inference_times_;
    std::vector<double> nms_times_;
    std::vector<double> ensemble_times_;
    
    // Internal processing methods
    std::vector<YOLODetection> processModelOutput(const std::vector<cv::Mat>& outputs, 
                                                 const cv::Size& originalSize,
                                                 const cv::Size& inputSize,
                                                 float confidenceThreshold,
                                                 int modelId);
    
    cv::Mat prepareInput(const cv::Mat& image, const cv::Size& inputSize);
    std::vector<cv::Mat> runInference(const cv::Mat& input, int modelIndex);
    
    // NMS and ensemble processing
    float calculateIoU(const cv::Rect& box1, const cv::Rect& box2);
    std::vector<int> performNMS(const std::vector<cv::Rect>& boxes, 
                               const std::vector<float>& confidences, 
                               float nmsThreshold);
    
    EnsembleDetection mergeDetections(const std::vector<YOLODetection>& detections);
    float calculateEnsembleConfidence(const std::vector<float>& confidences);
    
    // Quality assessment helpers
    double calculateSharpnessInROI(const cv::Mat& image, const cv::Rect& roi);
    double calculateContrastInROI(const cv::Mat& image, const cv::Rect& roi);
    bool checkDetectionBounds(const cv::Rect& detection, const cv::Size& imageSize);
    
    // Visualization helpers
    cv::Scalar getClassColor(int classId);
    std::string formatConfidence(float confidence);
    void drawBoundingBox(cv::Mat& image, const cv::Rect& box, const cv::Scalar& color, int thickness = 2);
    void drawLabel(cv::Mat& image, const std::string& label, const cv::Point& position, 
                  const cv::Scalar& color);
    
    // Constants
    static constexpr float DEFAULT_CONFIDENCE_THRESHOLD = 0.5f;
    static constexpr float DEFAULT_NMS_THRESHOLD = 0.4f;
    static constexpr int DEFAULT_MIN_DETECTION_COUNT = 1;
    static constexpr int MAX_MODELS = 10;
    
    // Default class names for wind turbine defects
    static const std::vector<std::string> DEFAULT_CLASS_NAMES;
};

// Utility functions for integration
struct DetectionStatistics {
    int total_detections;
    int crack_detections;
    int erosion_detections;
    int hotspot_detections;
    double average_confidence;
    double detection_density;  // detections per square meter
};

/**
 * Calculate detection statistics for analysis
 */
DetectionStatistics calculateDetectionStats(const std::vector<EnsembleDetection>& detections,
                                           const cv::Size& imageSize,
                                           double scalingFactor = 1.0);

/**
 * Filter detections by defect type
 */
std::vector<EnsembleDetection> filterDetectionsByType(const std::vector<EnsembleDetection>& detections,
                                                     const std::string& defectType);

/**
 * Group nearby detections (useful for clustering analysis)
 */
std::vector<std::vector<EnsembleDetection>> groupNearbyDetections(const std::vector<EnsembleDetection>& detections,
                                                                 double maxDistance = 50.0);

/**
 * Convert detection coordinates to normalized blade coordinates
 */
cv::Point2f convertToBladeCoordinates(const cv::Point2f& imagePoint, 
                                     const cv::Size& imageSize,
                                     const cv::Rect& bladeROI);

#endif // YOLO_ENSEMBLE_DETECTOR_H