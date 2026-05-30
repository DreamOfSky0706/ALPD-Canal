#pragma once
#include <opencv2/opencv.hpp>

// 要保存预处理阶段的所有输出。
struct ROIResult
{
	cv::Mat roiImage;
	cv::Rect roiRect;
	cv::Mat toothMask;
	cv::Mat convergeMask;
	cv::Mat roiToothMask;
	cv::Mat roiPulpMask;
	cv::Mat resizedImage;
};

class Preprocessor
{
public:
	// 主函数。
	static ROIResult extractROI(const cv::Mat& inputBGR,
								const std::string& debugDir = "");
	// 填补二值掩膜中的孔洞，找到外轮廓后用FILLED模式重新绘制。
	static cv::Mat fillHoles(const cv::Mat& mask);

private:
	// 从彩色图像中提取牙齿区域的二值掩膜。
	// 内部尝试颜色分割，如果效果不好则回退到GrabCut或Otsu。
	static cv::Mat extractToothMask(const cv::Mat& bgr,
									const std::string& debugDir);

};
