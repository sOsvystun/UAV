#ifndef ENHANCED_IMAGE_PROCESSOR_H
#define ENHANCED_IMAGE_PROCESSOR_H

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <vector>
#include <string>
#include <memory>

/**
 * Enhanced Image Processor implementing the complete Block 1 pipeline
 * from "Criticality Assessment of Wind Turbine Defects via Multispectral UAV Fusion and Fuzzy Logic"
 * 
 * This class implements:
 * - Multispectral (RGB + Thermal) image fusion
 * - Advanced preprocessing pipeline with bilateral filtering and CLAHE
 * - Geometric feature extraction with photogrammetric scaling
 * - Thermal signature analysis
 * - Integration with YOLOv8 ensemble detection
 */

struct CameraParameters {
    double focal_length_mm;      // Lens focal length in mm
    double pixel_size_um;        // Physical pixel size in micrometers
    cv::Size sensor_size;        // Sensor dimensions in pixels
    cv::Mat camera_matrix;       // Camera intrinsic matrix
    cv::Mat distortion_coeffs;   // Distortion coefficients
};

struct DefectROI {
    cv::Rect bounding_box;       // Bounding box from YOLO detection
    double confidence;           // Detection confidence
    std::string defect_class;    // Defect type (crack, erosion, hotspot)
    int detection_id;            // Unique detection identifier
};

struct GeometricFeatures {
    double area_pixels;          // Area in pixels
    double area_real_mm2;        // Real-world area in mm²
    double length_real_mm;       // Real-world length in mm
    double width_avg_mm;         // Average width in mm
    double perimeter_real_mm;    // Perimeter in mm
    double curvature_avg;        // Average curvature
    cv::Point2f centroid;        // Centroid coordinates
    double aspect_ratio;         // Width/height ratio
    double solidity;             // Area/convex hull area
    std::vector<cv::Point> contour;  // Main defect contour
};

struct ThermalFeatures {
    double delta_t_max;          // Maximum temperature difference in °C
    double delta_t_avg;          // Average temperature difference in °C
    double thermal_gradient;     // Temperature Laplacian (∇²T)
    double thermal_contrast;     // Thermal contrast measure
    cv::Point2f hotspot_center;  // Center of thermal anomaly
    double thermal_area_ratio;   // Ratio of thermal anomaly to total area
};

struct PhotogrammetricData {
    double distance_to_target_m; // Distance from UAV to target in meters
    double scaling_factor;       // Pixels to mm conversion factor
    cv::Point3f world_coordinates; // 3D world coordinates
    double ground_sample_distance; // GSD in mm/pixel
};

class EnhancedImageProcessor {
public:
    EnhancedImageProcessor();
    ~EnhancedImageProcessor();
    
    // Configuration methods
    void setCameraParameters(const CameraParameters& params);
    void setPhotogrammetricData(const PhotogrammetricData& data);
    
    // Main processing pipeline methods
    cv::Mat preprocessImage(const cv::Mat& image);
    cv::Mat combineRgbAndThermal(const cv::Mat& rgbImage, const cv::Mat& thermalImage);
    std::vector<cv::Mat> augmentImage(const cv::Mat& image);
    
    // Block 1: Complete processing pipeline
    GeometricFeatures extractGeometricFeatures(const cv::Mat& image, const DefectROI& roi);
    ThermalFeatures extractThermalFeatures(const cv::Mat& thermalImage, const DefectROI& roi);
    
    // Advanced preprocessing methods
    cv::Mat applyBilateralFilter(const cv::Mat& image, int d = 9, 
                                double sigmaColor = 75.0, double sigmaSpace = 75.0);
    cv::Mat applyCLAHE(const cv::Mat& image, double clipLimit = 2.0, 
                       cv::Size tileGridSize = cv::Size(8, 8));
    cv::Mat adaptiveThresholding(const cv::Mat& image, int blockSize = 11, double C = 2.0);
    
    // Morphological operations
    cv::Mat morphologicalFiltering(const cv::Mat& binaryImage, 
                                  cv::MorphShapes shape = cv::MORPH_ELLIPSE,
                                  cv::Size kernelSize = cv::Size(5, 5));
    
    // Geometric analysis methods
    std::vector<cv::Point> findLargestContour(const cv::Mat& binaryImage);
    double calculateCurvature(const std::vector<cv::Point>& contour);
    double calculateRealWorldArea(double pixelArea, double scalingFactor);
    double calculateRealWorldLength(double pixelLength, double scalingFactor);
    
    // Thermal analysis methods
    double calculateTemperatureDifference(const cv::Mat& thermalImage, const cv::Rect& roi);
    double calculateThermalGradient(const cv::Mat& thermalImage, const cv::Rect& roi);
    double calculateThermalContrast(const cv::Mat& thermalImage, const cv::Rect& roi);
    
    // Photogrammetric methods
    double calculateScalingFactor(double distanceToTarget, double focalLength, double pixelSize);
    cv::Point3f convertToWorldCoordinates(const cv::Point2f& imagePoint, double depth);
    
    // Image alignment and registration
    cv::Mat alignThermalToRGB(const cv::Mat& rgbImage, const cv::Mat& thermalImage);
    cv::Mat findHomographyBetweenImages(const cv::Mat& img1, const cv::Mat& img2);
    
    // Quality assessment
    double assessImageSharpness(const cv::Mat& image);
    double assessImageBrightness(const cv::Mat& image);
    double assessImageContrast(const cv::Mat& image);
    bool validateImageQuality(const cv::Mat& image, double minSharpness = 0.1, 
                             double minBrightness = 0.2, double minContrast = 0.3);
    
    // Utility methods
    cv::Mat undistortImage(const cv::Mat& image);
    cv::Mat cropROI(const cv::Mat& image, const cv::Rect& roi);
    void saveProcessingResults(const std::string& outputDir, const cv::Mat& image, 
                              const GeometricFeatures& geoFeatures,
                              const ThermalFeatures& thermalFeatures);
    
    // Debug and visualization methods
    cv::Mat visualizeContours(const cv::Mat& image, const std::vector<cv::Point>& contour);
    cv::Mat visualizeThermalOverlay(const cv::Mat& rgbImage, const cv::Mat& thermalImage, 
                                   double alpha = 0.6);
    void drawFeatureAnnotations(cv::Mat& image, const GeometricFeatures& features);
    
private:
    CameraParameters camera_params_;
    PhotogrammetricData photogrammetric_data_;
    bool camera_calibrated_;
    bool photogrammetric_initialized_;
    
    // Internal processing methods
    cv::Mat enhanceContrast(const cv::Mat& image);
    cv::Mat reduceNoise(const cv::Mat& image);
    std::vector<cv::KeyPoint> detectKeyPoints(const cv::Mat& image);
    cv::Mat computeDescriptors(const cv::Mat& image, std::vector<cv::KeyPoint>& keypoints);
    
    // Thermal processing helpers
    cv::Mat convertThermalToTemperature(const cv::Mat& thermalImage);
    cv::Mat computeTemperatureLaplacian(const cv::Mat& temperatureImage);
    
    // Geometric analysis helpers
    double computeContourArea(const std::vector<cv::Point>& contour);
    double computeContourPerimeter(const std::vector<cv::Point>& contour);
    cv::Point2f computeContourCentroid(const std::vector<cv::Point>& contour);
    double computeAspectRatio(const cv::Rect& boundingRect);
    double computeSolidity(const std::vector<cv::Point>& contour);
    
    // Validation helpers
    bool validateROI(const cv::Rect& roi, const cv::Size& imageSize);
    bool validateThermalImage(const cv::Mat& thermalImage);
    
    // Constants for processing parameters
    static constexpr double DEFAULT_BILATERAL_D = 9.0;
    static constexpr double DEFAULT_BILATERAL_SIGMA_COLOR = 75.0;
    static constexpr double DEFAULT_BILATERAL_SIGMA_SPACE = 75.0;
    static constexpr double DEFAULT_CLAHE_CLIP_LIMIT = 2.0;
    static constexpr int DEFAULT_ADAPTIVE_BLOCK_SIZE = 11;
    static constexpr double DEFAULT_ADAPTIVE_C = 2.0;
};

// Utility functions for integration with fuzzy logic system
struct FuzzyInputParameters {
    double defect_size_mm2;
    double location_normalized;
    double thermal_signature_delta_t;
    double expert_score;
    std::string component_type;
    std::string defect_type;
};

/**
 * Convert extracted features to fuzzy logic input parameters
 */
FuzzyInputParameters convertToFuzzyInputs(const GeometricFeatures& geoFeatures,
                                        const ThermalFeatures& thermalFeatures,
                                        const std::string& componentType,
                                        const std::string& defectType,
                                        double expertScore = 0.0);

/**
 * Normalize blade location from absolute position to 0-1 scale
 */
double normalizeBladeLocation(const cv::Point2f& position, double bladeLength);

/**
 * Determine component type based on location and geometric features
 */
std::string classifyComponent(const cv::Point2f& position, const GeometricFeatures& features);

#endif // ENHANCED_IMAGE_PROCESSOR_H