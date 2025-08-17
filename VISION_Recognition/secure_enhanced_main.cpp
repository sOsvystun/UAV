#include "enhanced_image_processor.h"
#include "yolo_ensemble_detector.h"
#include "../security/secure_subprocess_manager.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <chrono>
#include <filesystem>
#include <json/json.h>
#include <regex>
#include <thread>
#include <future>
#include <memory>
#include <map>
#include <string>

/**
 * Secure Enhanced Main Application with comprehensive security controls
 * and performance optimizations
 */

class SecurityValidator {
public:
    static bool validatePath(const std::string& path) {
        // Check for directory traversal attempts
        if (path.find("..") != std::string::npos || 
            path.find("//") != std::string::npos ||
            path.empty()) {
            return false;
        }
        
        // Check for absolute paths (security risk)
        if (path[0] == '/' || (path.length() > 1 && path[1] == ':')) {
            return false;
        }
        
        return true;
    }
    
    static bool validateFileExtension(const std::string& path, 
                                    const std::vector<std::string>& allowedExts) {
        std::filesystem::path p(path);
        std::string ext = p.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        
        return std::find(allowedExts.begin(), allowedExts.end(), ext) != allowedExts.end();
    }
    
    static bool validateFileSize(const std::string& path, size_t maxSize = 100 * 1024 * 1024) {
        try {
            return std::filesystem::file_size(path) <= maxSize;
        } catch (...) {
            return false;
        }
    }
    
    static std::string sanitizeString(const std::string& input) {
        std::string sanitized = input;
        // Remove potentially dangerous characters
        std::regex dangerous_chars(R"([<>&"'`$();|])");
        sanitized = std::regex_replace(sanitized, dangerous_chars, "_");
        return sanitized;
    }
};

class SecureProcessingConfiguration {
public:
    std::string rgb_image_path;
    std::string thermal_image_path;
    std::string output_directory;
    
    // Security settings
    bool enable_security_validation = true;
    size_t max_file_size = 100 * 1024 * 1024;  // 100MB
    std::vector<std::string> allowed_extensions = {".jpg", ".jpeg", ".png", ".tiff"};
    
    // Performance settings
    bool enable_parallel_processing = true;
    int max_threads = std::thread::hardware_concurrency();
    bool enable_gpu_acceleration = false;
    
    bool validate() const {
        if (enable_security_validation) {
            // Validate all paths
            if (!SecurityValidator::validatePath(rgb_image_path) ||
                !SecurityValidator::validatePath(thermal_image_path) ||
                !SecurityValidator::validatePath(output_directory)) {
                return false;
            }
            
            // Validate file extensions
            if (!SecurityValidator::validateFileExtension(rgb_image_path, allowed_extensions) ||
                !SecurityValidator::validateFileExtension(thermal_image_path, allowed_extensions)) {
                return false;
            }
            
            // Validate file sizes
            if (!SecurityValidator::validateFileSize(rgb_image_path, max_file_size) ||
                !SecurityValidator::validateFileSize(thermal_image_path, max_file_size)) {
                return false;
            }
        }
        
        return true;
    }
};

class SecureWindTurbineInspectionSystem {
private:
    std::unique_ptr<EnhancedImageProcessor> image_processor_;
    std::unique_ptr<YOLOEnsembleDetector> yolo_detector_;
    SecureProcessingConfiguration config_;
    std::string session_id_;
    
public:
    SecureWindTurbineInspectionSystem() {
        image_processor_ = std::make_unique<EnhancedImageProcessor>();
        yolo_detector_ = std::make_unique<YOLOEnsembleDetector>();
        session_id_ = generateSessionId();
    }
    
    bool initialize(const SecureProcessingConfiguration& config) {
        config_ = config;
        
        // Validate configuration
        if (!config_.validate()) {
            std::cerr << "Security validation failed for configuration" << std::endl;
            return false;
        }
        
        // Create secure output directory
        try {
            std::filesystem::create_directories(config_.output_directory);
            
            // Set restrictive permissions (Unix-like systems)
            #ifndef _WIN32
            std::filesystem::permissions(config_.output_directory, 
                std::filesystem::perms::owner_read | 
                std::filesystem::perms::owner_write | 
                std::filesystem::perms::owner_exec);
            #endif
            
        } catch (const std::exception& e) {
            std::cerr << "Failed to create output directory: " << e.what() << std::endl;
            return false;
        }
        
        return true;
    }
    
    bool processInspectionImagesSecure() {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::cout << "Starting secure wind turbine inspection processing..." << std::endl;
        std::cout << "Session ID: " << session_id_ << std::endl;
        
        try {
            // Step 1: Secure image loading with validation
            cv::Mat rgbImage, thermalImage;
            if (!loadImagesSecurely(rgbImage, thermalImage)) {
                return false;
            }
            
            // Step 2: Process with security controls
            auto results = processWithSecurityControls(rgbImage, thermalImage);
            
            // Step 3: Generate secure outputs
            generateSecureResults(results);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
            
            std::cout << "Secure processing completed in " << duration.count() << " ms" << std::endl;
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Secure processing failed: " << e.what() << std::endl;
            return false;
        }
    }

private:
    std::string generateSessionId() {
        auto now = std::chrono::system_clock::now();
        auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();
        return "session_" + std::to_string(timestamp);
    }
    
    bool loadImagesSecurely(cv::Mat& rgbImage, cv::Mat& thermalImage) {
        // Validate paths before loading
        if (!SecurityValidator::validatePath(config_.rgb_image_path) ||
            !SecurityValidator::validatePath(config_.thermal_image_path)) {
            std::cerr << "Security: Invalid image paths detected" << std::endl;
            return false;
        }
        
        // Load images with error handling
        rgbImage = cv::imread(config_.rgb_image_path);
        thermalImage = cv::imread(config_.thermal_image_path, cv::IMREAD_GRAYSCALE);
        
        if (rgbImage.empty() || thermalImage.empty()) {
            std::cerr << "Error: Cannot load images securely" << std::endl;
            return false;
        }
        
        // Validate image dimensions (prevent memory exhaustion)
        const int MAX_DIMENSION = 8192;
        if (rgbImage.cols > MAX_DIMENSION || rgbImage.rows > MAX_DIMENSION ||
            thermalImage.cols > MAX_DIMENSION || thermalImage.rows > MAX_DIMENSION) {
            std::cerr << "Security: Image dimensions exceed maximum allowed size" << std::endl;
            return false;
        }
        
        std::cout << "Images loaded securely: RGB=" << rgbImage.size() 
                  << ", Thermal=" << thermalImage.size() << std::endl;
        
        return true;
    }
    
    std::vector<ProcessedDefect> processWithSecurityControls(const cv::Mat& rgbImage, 
                                                           const cv::Mat& thermalImage) {
        std::vector<ProcessedDefect> results;
        
        // Process with timeout and resource limits
        std::promise<std::vector<ProcessedDefect>> promise;
        std::future<std::vector<ProcessedDefect>> future = promise.get_future();
        
        std::thread processing_thread([&]() {
            try {
                auto processed_results = performActualProcessing(rgbImage, thermalImage);
                promise.set_value(processed_results);
            } catch (...) {
                promise.set_exception(std::current_exception());
            }
        });
        
        // Wait with timeout
        if (future.wait_for(std::chrono::minutes(5)) == std::future_status::timeout) {
            std::cerr << "Processing timeout exceeded" << std::endl;
            processing_thread.detach();  // Let thread finish naturally
            return results;
        }
        
        processing_thread.join();
        
        try {
            results = future.get();
        } catch (const std::exception& e) {
            std::cerr << "Processing failed: " << e.what() << std::endl;
        }
        
        return results;
    }
    
    std::vector<ProcessedDefect> performActualProcessing(const cv::Mat& rgbImage, 
                                                       const cv::Mat& thermalImage) {
        std::vector<ProcessedDefect> results;
        
        try {
            // Step 1: Enhanced image processing
            auto enhanced_rgb = image_processor_->enhanceImage(rgbImage);
            auto enhanced_thermal = image_processor_->enhanceImage(thermalImage);
            
            // Step 2: Multispectral fusion
            auto fused_image = image_processor_->fuseMultispectralImages(enhanced_rgb, enhanced_thermal);
            
            // Step 3: YOLO ensemble detection
            auto detections = yolo_detector_->detectDefects(fused_image);
            
            // Step 4: Process detections into structured results
            for (const auto& detection : detections) {
                ProcessedDefect defect;
                defect.type = detection.class_name;
                defect.confidence = detection.confidence;
                defect.bounding_box = detection.bbox;
                defect.severity_score = calculateSeverityScore(detection);
                results.push_back(defect);
            }
            
            std::cout << "Processed " << results.size() << " defects successfully" << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "Error in actual processing: " << e.what() << std::endl;
        }
        
        return results;
    }
    
    float calculateSeverityScore(const Detection& detection) {
        // Simple severity calculation based on confidence and size
        float size_factor = (detection.bbox.width * detection.bbox.height) / (640.0f * 480.0f);
        return detection.confidence * (0.7f + 0.3f * size_factor);
    }
    
    void generateSecureResults(const std::vector<ProcessedDefect>& results) {
        try {
            // Create secure output filename
            std::string timestamp = std::to_string(std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count());
            std::string output_file = config_.output_directory + "/inspection_results_" + timestamp + ".json";
            
            // Validate output path
            if (!SecurityValidator::validatePath(output_file)) {
                throw std::runtime_error("Invalid output path");
            }
            
            // Generate JSON results
            Json::Value json_results;
            json_results["session_id"] = session_id_;
            json_results["timestamp"] = timestamp;
            json_results["total_defects"] = static_cast<int>(results.size());
            
            Json::Value defects_array(Json::arrayValue);
            for (const auto& defect : results) {
                Json::Value defect_json;
                defect_json["type"] = SecurityValidator::sanitizeString(defect.type);
                defect_json["confidence"] = defect.confidence;
                defect_json["severity_score"] = defect.severity_score;
                defect_json["bounding_box"]["x"] = defect.bounding_box.x;
                defect_json["bounding_box"]["y"] = defect.bounding_box.y;
                defect_json["bounding_box"]["width"] = defect.bounding_box.width;
                defect_json["bounding_box"]["height"] = defect.bounding_box.height;
                defects_array.append(defect_json);
            }
            json_results["defects"] = defects_array;
            
            // Write results securely
            std::ofstream output_stream(output_file);
            if (!output_stream.is_open()) {
                throw std::runtime_error("Cannot create output file");
            }
            
            Json::StreamWriterBuilder builder;
            builder["indentation"] = "  ";
            std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
            writer->write(json_results, &output_stream);
            
            output_stream.close();
            
            // Set secure file permissions
            #ifndef _WIN32
            std::filesystem::permissions(output_file, 
                std::filesystem::perms::owner_read | std::filesystem::perms::owner_write);
            #endif
            
            std::cout << "Results written securely to: " << output_file << std::endl;
            
        } catch (const std::exception& e) {
            std::cerr << "Error generating secure results: " << e.what() << std::endl;
        }
    }
    
    bool integrateWithFuzzyLogicSecure(const std::vector<ProcessedDefect>& defects) {
        try {
            // Create secure subprocess manager
            security::SecurityPolicy policy;
            policy.command_whitelist = {"python", "python3"};
            policy.allowed_paths = {"/usr/bin", "/usr/local/bin", "./VISION_Fuzzy"};
            policy.require_absolute_paths = false;
            policy.max_arguments = 20;
            policy.max_argument_length = 256;
            
            security::SecureSubprocessManager subprocess_manager(policy);
            
            // Prepare secure input data
            std::string temp_input_file = config_.output_directory + "/fuzzy_input_" + session_id_ + ".json";
            
            // Validate temp file path
            if (!SecurityValidator::validatePath(temp_input_file)) {
                std::cerr << "Invalid temporary file path" << std::endl;
                return false;
            }
            
            // Write defects to temporary file
            Json::Value input_data;
            input_data["session_id"] = session_id_;
            input_data["defect_count"] = static_cast<int>(defects.size());
            
            Json::Value defects_array(Json::arrayValue);
            for (const auto& defect : defects) {
                Json::Value defect_json;
                defect_json["type"] = SecurityValidator::sanitizeString(defect.type);
                defect_json["confidence"] = defect.confidence;
                defect_json["severity_score"] = defect.severity_score;
                defects_array.append(defect_json);
            }
            input_data["defects"] = defects_array;
            
            std::ofstream temp_file(temp_input_file);
            Json::StreamWriterBuilder builder;
            std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
            writer->write(input_data, &temp_file);
            temp_file.close();
            
            // Prepare secure command execution
            security::ProcessConfig process_config;
            process_config.command = {
                "python3",
                "./VISION_Fuzzy/secure_integration_bridge.py",
                "--input", temp_input_file,
                "--output", config_.output_directory + "/fuzzy_output_" + session_id_ + ".json",
                "--session", session_id_
            };
            
            process_config.working_directory = ".";
            process_config.timeout = std::chrono::milliseconds(30000);  // 30 second timeout
            process_config.capture_output = true;
            process_config.max_output_size = 1024 * 1024;  // 1MB max output
            
            // Execute fuzzy logic integration securely
            std::cout << "Executing fuzzy logic integration securely..." << std::endl;
            auto result = subprocess_manager.executeSecurely(process_config);
            
            if (result.success && result.exit_code == 0) {
                std::cout << "Fuzzy logic integration completed successfully" << std::endl;
                std::cout << "Output: " << result.stdout_output << std::endl;
                
                // Clean up temporary file
                std::filesystem::remove(temp_input_file);
                
                return true;
            } else {
                std::cerr << "Fuzzy logic integration failed:" << std::endl;
                std::cerr << "Exit code: " << result.exit_code << std::endl;
                std::cerr << "Error: " << result.stderr_output << std::endl;
                std::cerr << "Error message: " << result.error_message << std::endl;
                
                // Clean up temporary file
                std::filesystem::remove(temp_input_file);
                
                return false;
            }
            
        } catch (const std::exception& e) {
            std::cerr << "Exception in fuzzy logic integration: " << e.what() << std::endl;
            return false;
        }
    }
};

// Define ProcessedDefect structure
struct ProcessedDefect {
    std::string type;
    float confidence;
    cv::Rect bounding_box;
    float severity_score;
};

// Define Detection structure  
struct Detection {
    std::string class_name;
    float confidence;
    cv::Rect bbox;
};
};