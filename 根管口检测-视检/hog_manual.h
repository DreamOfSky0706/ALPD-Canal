#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

// ManualHOG实现了梯度计算、cell直方图构建、block归一化这几步，
// 和OpenCV内置的HOGDescriptor功能类似但参数和实现细节略有不同。
// 使用方法是直接创建对象，调用compute提取特征或者visualize查看可视化结果。
class ManualHOG
{
public:
	int winW = 32, winH = 32;
	int blockW = 16, blockH = 16;
	int cellW = 8, cellH = 8;
	int strideW = 8, strideH = 8;
	int numBins = 9;

	// 对单个窗口图像提取HOG返回拼接后的完整特征向量。
	std::vector<float> compute(const cv::Mat& window) const;

	// 将HOG特征可视化，在每个cell中心画出各方向的线段，
	// 线段长度对应该方向的梯度强度。返回一张放大后的可视化图。
	// 其实这里给人的感觉和当初藻华画光流的过程很像。
	cv::Mat visualize(const cv::Mat& window) const;

private:
	// 计算图像的梯度幅值和方向。对多通道图像会分别计算每个通道的梯度，
	// 取幅值最大的通道。方向范围是0到180度。
	void computeGradients(const cv::Mat& img,
						  cv::Mat& magnitude, cv::Mat& angle) const;

	// 对单个cell区域构建方向直方图。使用插值把梯度幅值按比例分配到相邻的两个bin中，这样可以减少离散造成的信息损失。
	std::vector<float> computeCellHistogram(
		const cv::Mat& mag, const cv::Mat& ang,
		int cellY, int cellX) const;

	// 先做L2归一化，然后把大于0.2的值截断到0.2，再做一次L2归一化。
	// 这是HOG论文中推荐的归一化方式，目的是减少局部光照变化的影响。
	void normalizeBlock(std::vector<float>& blockFeat) const;
};
