#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include "detector.h"
#include "nms.h"
#include "preprocess.h"

std::vector<Detection> detectAndPostProcess(
	const ROIResult& roi,
	Detector& detector,
	cv::Mat* outEnhanced = nullptr);

void drawDetections(cv::Mat& image,
					const std::vector<Detection>& detections);
