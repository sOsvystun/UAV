#include "enhanced_image_processor.h"
#include "yolo_ensemble_detector.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <filesystem>
#include <json/json.h>

/**
 * Enhanced Main Application implementing the complete framework from
 * "Criticality Assessment of Wind Turbine Defects via Multispectral UAV Fusion and Fuzzy Logic"
 * 
 * This application demonstrates the complete Block 1 pipeline:
 * - Multispectral image processing
 * - YOLOv8 ensemble detection
 * - Feature extraction and analysis
 * - Integration with fuzzy logic system
 */

struct ProcessingConfiguration {
    // Input/Output paths
    std::string rgb_image_path;
    std::string thermal_image_path;
    std::string output_directory;
    std::string config_file_path;
    
    // Camera parameters
    CameraParameters camera_params;
    PhotogrammetricData photogrammetric_data;
    
    // Detection parameters
    std::vector<ModelConfiguration> yolo_models;
    float ensemble_confidence_threshold = 0.5f;
    float ensemble_nms_threshold = 0.4f;
    
    // Processing options
    bool enable_multispectral_fusion = true;
    bool enable_quality_assessment = true;
    bool enable_visualization = true;
    bool save_intermediate_results = false;
    
    // Integration options
    bool integrate_with_fuzzy_logic = true;
    std::string python_script_path = "../VISION_Fuzzy/enhanced_fis_core.py";
};

class EnhancedWindTurbineInspectionSystem {
public:
    EnhancedWindTurbineInspectionSystem() {
        image_processor_ = std::make_unique<EnhancedImageProcessor>();
        yolo_detector_ = std::make_unique<YOLOEnsembleDetector>();
    }
    
    bool initialize(const ProcessingConfiguration& config) {
        config_ = config;
        
        // Initialize image processor
        image_processor_->setCameraParameters(config.camera_params);
        image_processor_->setPhotogrammetricData(config.photogrammetric_data);
        
        // Initialize YOLO ensemble detector
        if (!yolo_detector_->loadModels(config.yolo_models)) {
            std::cerr << "Error: Failed to load YOLO models" << std::endl;
            return false;
        }
        
        yolo_detector_->setEnsembleParameters(
            config.ensemble_confidence_threshold,
            config.ensemble_nms_threshold
        );
        
        // Set up class names for wind turbine defects
        std::vector<std::string> classNames = {"crack", "erosion", "hotspot"};
        yolo_detector_->setClassNames(classNames);
        
        // Create output directory
        std::filesystem::create_directories(config.output_directory);
        
        return true;
    }
    
    bool processInspectionImages() {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::cout << "Starting enhanced wind turbine inspection processing..." << std::endl;
        
        // Step 1: Load and validate input images
        cv::Mat rgbImage = cv::imread(config_.rgb_image_path);
        cv::Mat thermalImage = cv::imread(config_.thermal_image_path, cv::IMREAD_GRAYSCALE);
        
        if (rgbImage.empty()) {
            std::cerr << "Error: Cannot load RGB image from " << config_.rgb_image_path << std::endl;
            return false;
        }
        
        if (thermalImage.empty()) {
            std::cerr << "Error: Cannot load thermal image from " << config_.thermal_image_path << std::endl;
            return false;
        }
        
        std::cout << "Loaded RGB image: " << rgbImage.size() << std::endl;
        std::cout << "Loaded thermal image: " << thermalImage.size() << std::endl;
        
        // Step 2: Image quality assessment
        if (config_.enable_quality_assessment) {
            if (!assessImageQuality(rgbImage, thermalImage)) {
                std::cerr << "Warning: Image quality assessment failed" << std::endl;
            }
        }
        
        // Step 3: Preprocess images
        std::cout << "Preprocessing images..." << std::endl;
        cv::Mat preprocessedRGB = image_processor_->preprocessImage(rgbImage);
        cv::Mat preprocessedThermal = image_processor_->preprocessImage(thermalImage);
        
        // Step 4: Multispectral fusion
        cv::Mat fusedImage;
        if (config_.enable_multispectral_fusion) {
            std::cout << "Performing multispectral fusion..." << std::endl;
            fusedImage = image_processor_->combineRgbAndThermal(preprocessedRGB, preprocessedThermal);
        } else {
            fusedImage = preprocessedRGB;
        }
        
        // Step 5: YOLOv8 ensemble detection
        std::cout << "Running YOLOv8 ensemble detection..." << std::endl;
        std::vector<EnsembleDetection> detections;
        
        if (config_.enable_multispectral_fusion) {
            detections = yolo_detector_->detectMultispectral(preprocessedRGB, preprocessedThermal);
        } else {
            detections = yolo_detector_->detectEnsemble(fusedImage);
        }
        
        std::cout << "Found " << detections.size() << " defects" << std::endl;
        
        // Step 6: Feature extraction for each detection
        std::cout << "Extracting features for each detection..." << std::endl;
        std::vector<ProcessedDefect> processedDefects;
        
        for (size_t i = 0; i < detections.size(); ++i) {
            ProcessedDefect defect = processDetection(detections[i], fusedImage, thermalImage, i);
            processedDefects.push_back(defect);
        }
        
        // Step 7: Integration with fuzzy logic system
        if (config_.integrate_with_fuzzy_logic) {
            std::cout << "Integrating with fuzzy logic system..." << std::endl;
            integrateWithFuzzyLogic(processedDefects);
        }
        
        // Step 8: Generate results and visualizations
        generateResults(processedDefects, fusedImage, thermalImage);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "Processing completed in " << duration.count() << " ms" << std::endl;
        
        return true;
    }
    
private:
    struct ProcessedDefect {
        EnsembleDetection detection;
        GeometricFeatures geometric_features;
        ThermalFeatures thermal_features;
        FuzzyInputParameters fuzzy_inputs;
        double criticality_score = 0.0;
        std::string epri_level;
        int defect_id;
    };
    
    std::unique_ptr<EnhancedImageProcessor> image_processor_;
    std::unique_ptr<YOLOEnsembleDetector> yolo_detector_;
    ProcessingConfiguration config_;
    
    bool assessImageQuality(const cv::Mat& rgbImage, const cv::Mat& thermalImage) {
        std::cout << "Assessing image quality..." << std::endl;
        
        // Assess RGB image quality
        double rgbSharpness = image_processor_->assessImageSharpness(rgbImage);
        double rgbBrightness = image_processor_->assessImageBrightness(rgbImage);
        double rgbContrast = image_processor_->assessImageContrast(rgbImage);
        
        std::cout << "RGB Image Quality:" << std::endl;
        std::cout << "  Sharpness: " << rgbSharpness << std::endl;
        std::cout << "  Brightness: " << rgbBrightness << std::endl;
        std::cout << "  Contrast: " << rgbContrast << std::endl;
        
        // Assess thermal image quality
        double thermalSharpness = image_processor_->assessImageSharpness(thermalImage);
        double thermalBrightness = image_processor_->assessImageBrightness(thermalImage);
        double thermalContrast = image_processor_->assessImageContrast(thermalImage);
        
        std::cout << "Thermal Image Quality:" << std::endl;
        std::cout << "  Sharpness: " << thermalSharpness << std::endl;
        std::cout << "  Brightness: " << thermalBrightness << std::endl;
        std::cout << "  Contrast: " << thermalContrast << std::endl;
        
        // Validate quality thresholds
        bool rgbQualityOK = image_processor_->validateImageQuality(rgbImage, 0.1, 0.2, 0.3);
        bool thermalQualityOK = image_processor_->validateImageQuality(thermalImage, 0.05, 0.1, 0.2);
        
        std::cout << "Quality Assessment: RGB=" << (rgbQualityOK ? "PASS" : "FAIL") 
                  << ", Thermal=" << (thermalQualityOK ? "PASS" : "FAIL") << std::endl;
        
        return rgbQualityOK && thermalQualityOK;
    }
    
    ProcessedDefect processDetection(const EnsembleDetection& detection, 
                                   const cv::Mat& fusedImage, 
                                   const cv::Mat& thermalImage, 
                                   int defectId) {
        ProcessedDefect processed;
        processed.detection = detection;
        processed.defect_id = defectId;
        
        // Convert to DefectROI
        DefectROI roi;
        roi.bounding_box = detection.bounding_box;
        roi.confidence = detection.ensemble_confidence;
        roi.defect_class = detection.class_name;
        roi.detection_id = defectId;
        
        // Extract geometric features
        processed.geometric_features = image_processor_->extractGeometricFeatures(fusedImage, roi);
        
        // Extract thermal features
        processed.thermal_features = image_processor_->extractThermalFeatures(thermalImage, roi);
        
        // Convert to fuzzy logic inputs
        processed.fuzzy_inputs = convertToFuzzyInputs(
            processed.geometric_features,
            processed.thermal_features,
            classifyComponent(detection.center, processed.geometric_features),
            detection.class_name
        );
        
        std::cout << "Processed defect " << defectId << " (" << detection.class_name << "):" << std::endl;
        std::cout << "  Area: " << processed.geometric_features.area_real_mm2 << " mm²" << std::endl;
        std::cout << "  Thermal ΔT: " << processed.thermal_features.delta_t_max << "°C" << std::endl;
        std::cout << "  Location: " << processed.fuzzy_inputs.location_normalized << std::endl;
        
        return processed;
    }
    
    void integrateWithFuzzyLogic(std::vector<ProcessedDefect>& defects) {
        // Create input file for Python fuzzy logic system
        std::string inputFile = config_.output_directory + "/defects_input.json";
        std::string outputFile = config_.output_directory + "/defects_output.json";
        
        // Write defect data to JSON file
        Json::Value root(Json::arrayValue);
        
        for (const auto& defect : defects) {
            Json::Value defectJson;
            defectJson["defect_id"] = defect.defect_id;
            defectJson["defect_type"] = defect.fuzzy_inputs.defect_type;
            defectJson["component_type"] = defect.fuzzy_inputs.component_type;
            defectJson["area_mm2"] = defect.fuzzy_inputs.defect_size_mm2;
            defectJson["location_normalized"] = defect.fuzzy_inputs.location_normalized;
            defectJson["thermal_delta_t"] = defect.fuzzy_inputs.thermal_signature_delta_t;
            defectJson["confidence"] = defect.detection.ensemble_confidence;
            
            root.append(defectJson);
        }
        
        // Write to file
        std::ofstream file(inputFile);
        Json::StreamWriterBuilder builder;
        std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
        writer->write(root, &file);
        file.close();
        
        // Call Python fuzzy logic system
        std::string pythonCommand = "python " + config_.python_script_path + 
                                   " --input " + inputFile + 
                                   " --output " + outputFile;
        
        std::cout << "Executing: " << pythonCommand << std::endl;
        int result = std::system(pythonCommand.c_str());
        
        if (result == 0) {
            // Read results back
            std::ifstream resultFile(outputFile);
            if (resultFile.is_open()) {
                Json::Value resultRoot;
                Json::CharReaderBuilder readerBuilder;
                std::string errors;
                
                if (Json::parseFromStream(readerBuilder, resultFile, &resultRoot, &errors)) {
                    // Update defects with criticality scores
                    for (size_t i = 0; i < defects.size() && i < resultRoot.size(); ++i) {
                        defects[i].criticality_score = resultRoot[static_cast<int>(i)]["criticality_score"].asDouble();
                        defects[i].epri_level = resultRoot[static_cast<int>(i)]["epri_level"].asString();
                    }
                    
                    std::cout << "Successfully integrated with fuzzy logic system" << std::endl;
                } else {
                    std::cerr << "Error parsing fuzzy logic results: " << errors << std::endl;
                }
                
                resultFile.close();
            }
        } else {
            std::cerr << "Error executing Python fuzzy logic system" << std::endl;
        }
    }
    
    void generateResults(const std::vector<ProcessedDefect>& defects, 
                        const cv::Mat& fusedImage, 
                        const cv::Mat& thermalImage) {
        std::cout << "Generating results and visualizations..." << std::endl;
        
        // Generate summary report
        generateSummaryReport(defects);
        
        // Generate detailed CSV report
        generateDetailedReport(defects);
        
        // Generate visualizations
        if (config_.enable_visualization) {
            generateVisualizations(defects, fusedImage, thermalImage);
        }
        
        // Save intermediate results if requested
        if (config_.save_intermediate_results) {
            saveIntermediateResults(fusedImage, thermalImage);
        }
    }
    
    void generateSummaryReport(const std::vector<ProcessedDefect>& defects) {
        std::string reportPath = config_.output_directory + "/inspection_summary.txt";
        std::ofstream report(reportPath);
        
        report << "=== WIND TURBINE INSPECTION SUMMARY ===" << std::endl;
        report << "Generated: " << getCurrentTimestamp() << std::endl;
        report << "Total defects detected: " << defects.size() << std::endl;
        report << std::endl;
        
        // Count defects by type
        std::map<std::string, int> defectCounts;
        std::map<std::string, int> criticalityCounts;
        double totalCriticality = 0.0;
        
        for (const auto& defect : defects) {
            defectCounts[defect.detection.class_name]++;
            criticalityCounts[defect.epri_level]++;
            totalCriticality += defect.criticality_score;
        }
        
        report << "DEFECT BREAKDOWN:" << std::endl;
        for (const auto& pair : defectCounts) {
            report << "  " << pair.first << ": " << pair.second << std::endl;
        }
        report << std::endl;
        
        report << "CRITICALITY BREAKDOWN:" << std::endl;
        for (const auto& pair : criticalityCounts) {
            report << "  " << pair.first << ": " << pair.second << std::endl;
        }
        report << std::endl;
        
        if (!defects.empty()) {
            report << "AVERAGE CRITICALITY SCORE: " << (totalCriticality / defects.size()) << std::endl;
        }
        
        // Find most critical defect
        auto maxCriticalityIt = std::max_element(defects.begin(), defects.end(),
            [](const ProcessedDefect& a, const ProcessedDefect& b) {
                return a.criticality_score < b.criticality_score;
            });
        
        if (maxCriticalityIt != defects.end()) {
            report << std::endl;
            report << "MOST CRITICAL DEFECT:" << std::endl;
            report << "  ID: " << maxCriticalityIt->defect_id << std::endl;
            report << "  Type: " << maxCriticalityIt->detection.class_name << std::endl;
            report << "  Criticality Score: " << maxCriticalityIt->criticality_score << std::endl;
            report << "  EPRI Level: " << maxCriticalityIt->epri_level << std::endl;
            report << "  Area: " << maxCriticalityIt->geometric_features.area_real_mm2 << " mm²" << std::endl;
        }
        
        report.close();
        std::cout << "Summary report saved to: " << reportPath << std::endl;
    }
    
    void generateDetailedReport(const std::vector<ProcessedDefect>& defects) {
        std::string csvPath = config_.output_directory + "/detailed_results.csv";
        std::ofstream csv(csvPath);
        
        // CSV header
        csv << "defect_id,defect_type,component_type,confidence,area_mm2,length_mm,width_mm,"
            << "location_normalized,thermal_delta_t,thermal_gradient,criticality_score,epri_level,"
            << "bbox_x,bbox_y,bbox_width,bbox_height" << std::endl;
        
        // CSV data
        for (const auto& defect : defects) {
            csv << defect.defect_id << ","
                << defect.detection.class_name << ","
                << defect.fuzzy_inputs.component_type << ","
                << defect.detection.ensemble_confidence << ","
                << defect.geometric_features.area_real_mm2 << ","
                << defect.geometric_features.length_real_mm << ","
                << defect.geometric_features.width_avg_mm << ","
                << defect.fuzzy_inputs.location_normalized << ","
                << defect.thermal_features.delta_t_max << ","
                << defect.thermal_features.thermal_gradient << ","
                << defect.criticality_score << ","
                << defect.epri_level << ","
                << defect.detection.bounding_box.x << ","
                << defect.detection.bounding_box.y << ","
                << defect.detection.bounding_box.width << ","
                << defect.detection.bounding_box.height << std::endl;
        }
        
        csv.close();
        std::cout << "Detailed report saved to: " << csvPath << std::endl;
    }
    
    void generateVisualizations(const std::vector<ProcessedDefect>& defects, 
                               const cv::Mat& fusedImage, 
                               const cv::Mat& thermalImage) {
        // Convert detections for visualization
        std::vector<EnsembleDetection> detectionList;
        for (const auto& defect : defects) {
            detectionList.push_back(defect.detection);
        }
        
        // Create annotated image
        cv::Mat annotatedImage = yolo_detector_->visualizeDetections(fusedImage, detectionList);
        
        // Add criticality information
        for (const auto& defect : defects) {
            cv::Point labelPos(defect.detection.bounding_box.x, 
                              defect.detection.bounding_box.y - 10);
            
            std::string criticalityText = "Score: " + std::to_string(defect.criticality_score).substr(0, 4);
            cv::putText(annotatedImage, criticalityText, labelPos, 
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 2);
        }
        
        // Save annotated image
        std::string annotatedPath = config_.output_directory + "/annotated_results.jpg";
        cv::imwrite(annotatedPath, annotatedImage);
        
        // Create thermal overlay
        cv::Mat thermalOverlay = image_processor_->visualizeThermalOverlay(fusedImage, thermalImage);
        std::string thermalPath = config_.output_directory + "/thermal_overlay.jpg";
        cv::imwrite(thermalPath, thermalOverlay);
        
        std::cout << "Visualizations saved to:" << std::endl;
        std::cout << "  " << annotatedPath << std::endl;
        std::cout << "  " << thermalPath << std::endl;
    }
    
    void saveIntermediateResults(const cv::Mat& fusedImage, const cv::Mat& thermalImage) {
        cv::imwrite(config_.output_directory + "/fused_image.jpg", fusedImage);
        cv::imwrite(config_.output_directory + "/processed_thermal.jpg", thermalImage);
    }
    
    std::string getCurrentTimestamp() {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
        return ss.str();
    }
};

// Configuration loading function
ProcessingConfiguration loadConfiguration(const std::string& configPath) {
    ProcessingConfiguration config;
    
    // Set default values
    config.ensemble_confidence_threshold = 0.5f;
    config.ensemble_nms_threshold = 0.4f;
    config.enable_multispectral_fusion = true;
    config.enable_quality_assessment = true;
    config.enable_visualization = true;
    config.save_intermediate_results = false;
    config.integrate_with_fuzzy_logic = true;
    
    // Default camera parameters
    config.camera_params.focal_length_mm = 24.0;
    config.camera_params.pixel_size_um = 5.5;
    config.camera_params.sensor_size = cv::Size(1920, 1080);
    
    // Default photogrammetric data
    config.photogrammetric_data.distance_to_target_m = 10.0;
    config.photogrammetric_data.scaling_factor = 0.1;  // mm per pixel
    
    // Default YOLO model configurations
    ModelConfiguration yoloConfig;
    yoloConfig.model_path = "models/yolov8n.onnx";
    yoloConfig.model_name = "YOLOv8n";
    yoloConfig.confidence_threshold = 0.5f;
    yoloConfig.nms_threshold = 0.4f;
    yoloConfig.input_size = cv::Size(640, 640);
    yoloConfig.enabled = true;
    
    config.yolo_models.push_back(yoloConfig);
    
    // TODO: Load from actual config file if it exists
    if (std::filesystem::exists(configPath)) {
        std::cout << "Loading configuration from: " << configPath << std::endl;
        // Implementation for loading from JSON/XML config file
    }
    
    return config;
}

int main(int argc, char* argv[]) {
    std::cout << "Enhanced Wind Turbine Inspection System" << std::endl;
    std::cout << "=======================================" << std::endl;
    
    // Parse command line arguments
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <rgb_image> <thermal_image> <output_dir> [config_file]" << std::endl;
        std::cout << "Example: " << argv[0] << " rgb.jpg thermal.jpg output/ config.json" << std::endl;
        return -1;
    }
    
    // Load configuration
    std::string configPath = (argc >= 5) ? argv[4] : "config.json";
    ProcessingConfiguration config = loadConfiguration(configPath);
    
    // Set input/output paths from command line
    config.rgb_image_path = argv[1];
    config.thermal_image_path = argv[2];
    config.output_directory = argv[3];
    config.config_file_path = configPath;
    
    // Initialize and run the inspection system
    EnhancedWindTurbineInspectionSystem system;
    
    if (!system.initialize(config)) {
        std::cerr << "Error: Failed to initialize inspection system" << std::endl;
        return -1;
    }
    
    if (!system.processInspectionImages()) {
        std::cerr << "Error: Failed to process inspection images" << std::endl;
        return -1;
    }
    
    std::cout << "Inspection processing completed successfully!" << std::endl;
    std::cout << "Results saved to: " << config.output_directory << std::endl;
    
    return 0;
}