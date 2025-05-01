#include "detectron_handler.h"
#include <stdexcept>
#include <iostream>
#include <numpy/arrayobject.h>

DetectronHandler::DetectronHandler(const std::string& modelPath)
    : pModule(nullptr), pFunc(nullptr), modelPath(modelPath) {
    initialize();
}

DetectronHandler::~DetectronHandler() {
    Py_XDECREF(pFunc);
    Py_XDECREF(pModule);
    if (Py_IsInitialized()) {
        Py_Finalize();
    }
}

void DetectronHandler::initialize() {
    Py_Initialize();
    import_array();
    loadPythonModule();
}

void DetectronHandler::loadPythonModule() {
    PyObject* pName = PyUnicode_FromString("detectron_inference");
    pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (!pModule) {
        PyErr_Print();
        throw std::runtime_error("Failed to load Python module");
    }

    pFunc = PyObject_GetAttrString(pModule, "detect_objects");
    if (!pFunc || !PyCallable_Check(pFunc)) {
        PyErr_Print();
        throw std::runtime_error("Failed to load Python function");
    }
}

std::vector<cv::Rect> DetectronHandler::detectObjects(const cv::Mat& image) {
    npy_intp dims[3] = { image.rows, image.cols, image.channels() };
    PyObject* pImageArray = PyArray_SimpleNewFromData(3, dims, NPY_UINT8, image.data);

    PyObject* pArgs = PyTuple_New(1);
    PyTuple_SetItem(pArgs, 0, pImageArray);

    PyObject* pResult = PyObject_CallObject(pFunc, pArgs);
    Py_DECREF(pArgs);

    if (!pResult) {
        PyErr_Print();
        throw std::runtime_error("Failed to get result from Python function");
    }

    std::vector<cv::Rect> detections;

    Py_ssize_t size = PyList_Size(pResult);
    for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject* item = PyList_GetItem(pResult, i);
        int x = PyLong_AsLong(PyTuple_GetItem(item, 0));
        int y = PyLong_AsLong(PyTuple_GetItem(item, 1));
        int w = PyLong_AsLong(PyTuple_GetItem(item, 2));
        int h = PyLong_AsLong(PyTuple_GetItem(item, 3));
        detections.emplace_back(cv::Rect(x, y, w, h));
    }

    Py_DECREF(pResult);

    return detections;
}