#pragma once
#include <vector>
#include <opencv2/opencv.hpp>

struct Detection;

// 用NMS类用于检测结果的后处理。
// 包括经典的非极大值抑制、ROI边界过滤、Top-K保留、聚类合并、亮度过滤和掩膜过滤。
// 为的是减少误检和重复检测。
class NMS
{
public:
	// 经典的非极大值抑制，按置信度从高到低排序，
	// 依次保留当前最高分的检测，抑制与之IoU超过阈值的其他检测。
	static std::vector<Detection> apply(
		std::vector<Detection>& detections,
		float iouThreshold = 0.3f);

	// 过滤掉中心不在ROI内部指定边距范围内的检测，因为靠近ROI边缘的检测往往是不完整的。
	static std::vector<Detection> filterByROI(
		const std::vector<Detection>& detections,
		const cv::Rect& roiRect,
		float margin = 0.05f);

	// 只保留置信度最高的前maxCount个检测结果。
	static std::vector<Detection> keepTopK(
		std::vector<Detection>& detections,
		int maxCount = 4);

	// 基于距离的聚类合并。距离小于阈值的检测归为同一簇，
	// 每个簇用加权平均合成一个检测结果，置信度取簇内最大值。
	static std::vector<Detection> clusterMerge(
		const std::vector<Detection>& detections,
		float distThreshold);

	// 过滤掉检测框内平均亮度超过全图均值一定比例的检测，因为根管口应该是暗区域。
	static std::vector<Detection> filterByBrightness(
		const std::vector<Detection>& detections,
		const cv::Mat& grayImage,
		float maxBrightnessRatio = 1.0f);

	// 过滤掉中心不在掩膜区域上的检测，确保检测结果落在牙齿内。
	static std::vector<Detection> filterByMask(
		const std::vector<Detection>& detections,
		const cv::Mat& mask);

private:
	// 计算两个框的IOU，用于NMS和匹配。
	static float computeIoU(const cv::Rect& a, const cv::Rect& b);
};
