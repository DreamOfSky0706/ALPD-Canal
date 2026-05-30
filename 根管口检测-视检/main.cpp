/*
以下为本文件的提示词：
我正在做一个根管口检测项目，所有已完成模块见文件，现在需要编写 main.cpp 把这些模块串联起来形成完整的处理流程，并在每一步保存中间结果图像。
首先读入图像，然后调用 Preprocessor::extractROI 提取 ROI 区域、牙齿掩膜和暗区域掩膜。把 ROI 框、牙齿轮廓、裁剪后的 ROI、暗区域掩膜、牙齿掩膜以及牙齿掩膜的半透明叠加效果分别保存成带编号的图像，方便调试。
接下来，在 ROI 图像中心取一个 32×32 的小块，用 ManualHOG 计算特征并保存可视化图，同时用 Sobel 算子算梯度幅值也保存一张图，用来验证 HOG 实现是否正常。
然后检测之。先把 ROI 转灰度，用 CLAHE 增强对比度，自行设计相关参数，再调用 detector.detect 得到候选检测框。
检测之后依次做如下后处理，其中未提及的参数由你决定：先用 IoU 做 NMS，再过滤掉靠近 ROI 边缘的检测，然后用牙齿掩膜过滤掉不在牙齿区域内的检测，接着做亮度过滤，最后做聚类合并，并只保留置信度最高的前若干个。每一步之后打印剩余检测数量。把增强后的 ROI 和标注了所有候选框及置信度的图像也保存下来。
后处理结束后，用 Diagnosis::analyze 生成诊断报告，包括根管数量、牙齿类型和风险等级。用 Diagnosis::visualizeReport 保存为最终结果图，再调用 Diagnosis::saveReport 保存文本报告。在控制台输出根管数、类型和风险等级。
把上述逻辑封装成一个 processImage 函数。main 函数提供四种运行模式，可以通过命令行参数或交互输入选择：处理单张图像、批量处理一个目录下所有图像、处理视频文件、或者全部执行。批量处理时按文件名排序逐一调用 processImage。视频处理调用 VideoTracker 的 processVideo 方法。SVM 模型文件名为 64_svm_model.xml，图像目录默认为 data/，视频文件为 vid.mp4。用 OutputManager 统一管理输出路径，为每张图像创建独立的子目录。
*/

#include <iostream>
#include <string>
#include <filesystem>
#include <algorithm>
#include <opencv2/opencv.hpp>

#include "preprocess.h"
#include "hog_manual.h"
#include "detector.h"
#include "nms.h"
#include "diagnosis.h"
#include "output_manager.h"
#include "general.h"

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

void processImage(const string& imagePath,
				  Detector& detector,
				  OutputManager& outMgr)
{
	cout << "\n====== 正在处理: " << imagePath << " ======" << endl;

	Mat img = imread(imagePath, IMREAD_COLOR);
	if (img.empty())
	{
		cerr << "无法读取图像: " << imagePath << endl;
		return;
	}

	string outDir = outMgr.createImageDir(imagePath);
	cout << "输出目录: " << outDir << endl;

	resize(img, img, Size(1080, 1272));
	ROIResult roi = Preprocessor::extractROI(img, outDir);

	{
		Mat vis01 = roi.resizedImage.clone();
		rectangle(vis01, roi.roiRect, Scalar(0, 255, 0), 2);
		vector<vector<Point>> tc;
		findContours(roi.toothMask.clone(), tc, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		drawContours(vis01, tc, -1, Scalar(0, 255, 255), 2);
		imwrite(outMgr.getPath(outDir, "01_roi_and_tooth.png"), vis01);
		imwrite(outMgr.getPath(outDir, "02_roi_cropped.png"), roi.roiImage);
		imwrite(outMgr.getPath(outDir, "03_dark_regions.png"), roi.convergeMask);
		imwrite(outMgr.getPath(outDir, "04_tooth_mask.png"), roi.toothMask);

		Mat vis05 = roi.resizedImage.clone();
		Mat toothOv = Mat::zeros(vis05.size(), CV_8UC3);
		toothOv.setTo(Scalar(0, 255, 0), roi.toothMask);
		addWeighted(vis05, 0.6, toothOv, 0.4, 0, vis05);
		drawContours(vis05, tc, -1, Scalar(0, 255, 255), 2);
		rectangle(vis05, roi.roiRect, Scalar(0, 255, 0), 2);
		imwrite(outMgr.getPath(outDir, "05_tooth_overlay.png"), vis05);
	}

	ManualHOG manualHog;
	if (roi.roiImage.cols >= 32 && roi.roiImage.rows >= 32)
	{
		int cxPatch = roi.roiImage.cols / 2 - 16;
		int cyPatch = roi.roiImage.rows / 2 - 16;
		Mat patch = roi.roiImage(Rect(cxPatch, cyPatch, 32, 32));
		vector<float> hf = manualHog.compute(patch);
		cout << "手工HOG特征维度: " << hf.size() << endl;
		imwrite(outMgr.getPath(outDir, "06_hog_visualization.png"),
				manualHog.visualize(patch));

		Mat gp;
		cvtColor(patch, gp, COLOR_BGR2GRAY);
		Mat gx, gy, mag, ang;
		gp.convertTo(gp, CV_32F);
		Sobel(gp, gx, CV_32F, 1, 0, 1);
		Sobel(gp, gy, CV_32F, 0, 1, 1);
		cartToPolar(gx, gy, mag, ang, true);
		Mat magVis;
		normalize(mag, magVis, 0, 255, NORM_MINMAX, CV_8UC1);
		imwrite(outMgr.getPath(outDir, "07_gradient_magnitude.png"), magVis);
	}

	// 调用统一的检测+后处理流程，同时获取增强后的ROI图像用于保存
	Mat roiEnhanced;
	vector<Detection> detections = detectAndPostProcess(roi, detector, &roiEnhanced);

	imwrite(outMgr.getPath(outDir, "07_roi_enhanced.png"), roiEnhanced);

	// 用统一的绘制函数绘制候选框（方框+红十字+置信度）
	{
		Mat visAll = roi.resizedImage.clone();
		drawDetections(visAll, detections);
		imwrite(outMgr.getPath(outDir, "08_all_candidates.png"), visAll);
	}

	DiagnosticReport report = Diagnosis::analyze(detections, roi.roiImage, roi.roiRect);

	{
		Mat visFinal = Diagnosis::visualizeReport(roi.resizedImage, detections, report);
		imwrite(outMgr.getPath(outDir, "10_final_result.png"), visFinal);
		cout << "  已保存 10_final_result.png" << endl;
	}

	{
		Diagnosis::saveReport(report, outMgr.getPath(outDir, "11_report.txt"));
		cout << "  已保存 11_report.txt" << endl;
	}

	cout << "\n  => 根管数: " << report.numCanals
		<< ", 类型: " << report.toothType
		<< ", 风险: " << report.riskLevel << endl;
	cout << "  => 所有结果已保存到: " << outDir << endl;
}

int main(int argc, char** argv)
{
	string modelPath = "128_svm_model.xml";
	string imageDir = "data/";

	if (!fs::exists(modelPath))
	{
		fs::path exeDir = fs::path(argv[0]).parent_path();
		if (!exeDir.empty() && fs::exists(exeDir / modelPath))
		{
			fs::current_path(exeDir);
			cout << "切换到exe目录: " << fs::current_path() << endl;
		}

		if (!fs::exists(modelPath))
		{
			fs::path projDir = fs::path(argv[0]).parent_path().parent_path().parent_path();
			if (!projDir.empty() && fs::exists(projDir / modelPath))
			{
				fs::current_path(projDir);
				cout << "切换到项目目录: " << fs::current_path() << endl;
			}
		}
	}

	cout << "工作目录: " << fs::current_path() << endl;

	if (!fs::exists(modelPath))
	{
		cerr << "模型未找到: " << modelPath << endl;
		cerr << "查找位置: " << fs::current_path() << endl;
		cerr << "按回车退出..." << endl;
		cin.get();
		return -1;
	}

	if (!fs::exists(imageDir))
	{
		cerr << "数据目录未找到: " << imageDir << endl;
		cerr << "查找位置: " << fs::current_path() << endl;
		cerr << "按回车退出..." << endl;
		cin.get();
		return -1;
	}

	OutputManager outMgr("output");

	Detector detector;
	if (!detector.loadModel(modelPath))
	{
		return -1;
	}

	cout << "请选择运行模式:\n"
		<< "  1 - 处理单张图像\n"
		<< "  2 - 批量处理目录下所有图像\n"
		<< "  3 - 执行以上全部\n"
		<< "> ";

	int mode = 3;
	if (argc > 1)
	{
		mode = atoi(argv[1]);
	}
	else
	{
		cin >> mode;
	}

	if (mode == 1 || mode == 3)
	{
		string imagePath = (argc > 2) ? argv[2] : "data/177.1.png";
		processImage(imagePath, detector, outMgr);
	}

	if (mode == 2 || mode == 3)
	{
		cout << "\n====== 批量处理 ======" << endl;
		vector<string> paths;
		for (auto& entry : fs::directory_iterator(imageDir))
		{
			string ext = entry.path().extension().string();
			transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
			if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp")
			{
				paths.push_back(entry.path().string());
			}
		}
		sort(paths.begin(), paths.end());
		cout << "找到 " << paths.size() << " 张图像." << endl;
		for (const auto& p : paths)
		{
			processImage(p, detector, outMgr);
		}
	}

	cout << "\n全部完成! 结果保存在: " << outMgr.rootDir() << endl;
	return 0;
}
