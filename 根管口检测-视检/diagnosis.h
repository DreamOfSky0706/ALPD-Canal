#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

struct Detection;

// DiagnosticReport保存一次诊断分析的所有结果，
// 包括检测到的根管数量、推测的牙齿类型、风险等级、建议、
// 各根管的中心坐标和面积，以及对称性评分。
// 这个结构体同时用于可视化绘制和文本报告输出。
struct DiagnosticReport
{
	int numCanals = 0;
	std::string toothType;
	std::string riskLevel;
	std::string advice;
	float symmetryScore = 0.0f;
	std::vector<cv::Point> canalCenters;
	std::vector<int> canalAreas;
};

// 需要一个Diagnosis类提供静态方法，负责根据检测结果生成诊断报告、可视化结果图像以及保存文本报告。
// 我没学过医，所有方法都是静态的，不需要实例化就可以调用。
class Diagnosis
{
public:
	// 根据检测结果列表分析根管数量、推测牙齿类型、评估风险等级、计算对称性评分，返回完整的诊断报告。
	static DiagnosticReport analyze(
		const std::vector<Detection>& detections,
		const cv::Mat& roiImage,
		const cv::Rect& roiRect);

	// 在原图右侧拼接一个信息面板，绘制检测框和诊断信息，返回可视化结果图。
	static cv::Mat visualizeReport(
		const cv::Mat& fullImage,
		const std::vector<Detection>& detections,
		const DiagnosticReport& report);

	// 把诊断报告以文本形式写入文件，同时也打印到控制台。
	static void saveReport(const DiagnosticReport& report,
						   const std::string& filepath);
};
