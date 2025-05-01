cmake_minimum_required(VERSION 3.30)
project(UAV_Ensemble_Module)

set(CMAKE_CXX_STANDARD 20)

add_executable(untitled7 main.cpp
        detectron_handler.h
        DetectronHandler.cpp
        image_processor.h
        ImageProcessor.cpp
        model_ensemble.h
        model_ensemble.cpp
        ResultAnalyzer.h
        result_analyzer.cpp)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Python3_INCLUDE_DIRS})

add_executable(UAV_Ensemble_Module
        src/main.cpp
        src/detectron_handler.cpp
        src/detectron_handler.h
        src/image_processor.cpp
        src/image_processor.h
        src/model_ensemble.cpp
        src/model_ensemble.h
        src/result_analyzer.cpp
        src/result_analyzer.h
)

target_link_libraries(UAV_Ensemble_Module ${OpenCV_LIBS} Python3::Python)

target_compile_options(UAV_Ensemble_Module PRIVATE -Wall -Wextra -pedantic)

set_target_properties(UAV_Ensemble_Module PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
