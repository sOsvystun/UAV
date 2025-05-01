#ifndef DETECTRON_HANDLER_H
#define DETECTRON_HANDLER_H

#include <Python.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class DetectronHandler {
public:
    DetectronHandler(const std::string& modelPath);
    ~DetectronHandler();

    void initialize();
    std::vector<cv::Rect> detectObjects(const cv::Mat& image);

private:
    PyObject* pModule;
    PyObject* pFunc;
    std::string modelPath;

    void loadPythonModule();
};

#endif // DETECTRON_HANDLER_H