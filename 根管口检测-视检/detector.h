#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <vector>
#include <string>

// 用一个Detection结构体保存单个检测结果，包括检测框位置和置信度分数。
// 后续的NMS、诊断等模块都会用到这个结构体作为数据传递的基本单元。
struct Detection
{
	cv::Rect box;
	float confidence;
};

// Detector类封装了基于HOG特征和SVM分类器的根管口检测流程。
// 使用方法是先调用loadModel加载训练好的SVM模型文件，
// 然后对每张图像调用detect方法获取检测结果列表。
// 内部会对候选区域提取HOG特征，送入SVM判断是否为根管口。
class Detector
{
public:
	// 加载SVM模型文件，同时初始化HOG参数并校验特征维度是否匹配。
	bool loadModel(const std::string& modelPath);

	// 在给定的灰度ROI图像上执行完整的根管口检测流程。
	// roiGray是裁剪后的灰度图，roiOffset是ROI左上角在原图中的坐标偏移，
	// toothMask和pulpMask分别是牙齿区域和牙髓区域的二值掩膜，
	// confThresh是最终输出的最低置信度阈值。
	std::vector<Detection> detect(
		const cv::Mat& roiGray,
		cv::Point roiOffset,
		const cv::Mat& toothMask,
		const cv::Mat& pulpMask,
		float confThresh = 0.05f);

	// 返回当前HOG窗口的边长，外部模块可能需要知道这个值来做patch裁剪。
	int getWinSize() const
	{
		return winSize_;
	}

private:
	// 对输入图像提取HOG特征向量。图像会先被缩放到winSize x winSize大小，
	// 然后用内置的HOGDescriptor计算特征。
	void computeHOGFeatures(const cv::Mat& img,
							std::vector<float>& descriptors);

	// 在灰度图gray上以center为中心、patchSize为边长裁剪一个正方形区域，
	// 提取HOG特征后送入SVM得到原始决策值。越负说明越像根管口。
	// 如果裁剪区域超出图像边界则返回999.0f表示无效。
	float svmScoreAt(const cv::Mat& gray, cv::Point2f center,
					 int patchSize);

	cv::Ptr<cv::ml::SVM> svm_;
	int winSize_ = 64;
	int blockSize_ = 8;
	int blockStride_ = 4;
	int cellSize_ = 4;
	int nbins_ = 9;
};
