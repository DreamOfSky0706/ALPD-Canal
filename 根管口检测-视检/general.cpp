/*
悼念我没能做好的视频检测。
*/

#include "general.h"
#include <iostream>
#include <algorithm>

using namespace cv;
using namespace std;

// 视频帧和图像都调用这一个函数，避免逻辑重复。
vector<Detection> detectAndPostProcess(
	const ROIResult& roi,
	Detector& detector,
	Mat* outEnhanced)
{
	// 转灰度+CLAHE增强对比度
	Mat roiGray;
	cvtColor(roi.roiImage, roiGray, COLOR_BGR2GRAY);

	Ptr<CLAHE> clahe = createCLAHE(3.0, Size(8, 8));
	Mat roiEnhanced;
	clahe->apply(roiGray, roiEnhanced);

	// 图像模式保存中间结果通过指针返回。
	if (outEnhanced)
	{
		*outEnhanced = roiEnhanced.clone();
	}

	// 检测
	float confThresh = 0.02f;
	vector<Detection> detections = detector.detect(
		roiEnhanced,
		roi.roiRect.tl(),
		roi.roiToothMask,
		roi.roiPulpMask,
		confThresh);

	cout << "原始检测数量: " << detections.size() << endl;

	// 后处理流程。
	detections = NMS::apply(detections, 0.27f);
	cout << "NMS后: " << detections.size() << endl;

	detections = NMS::filterByROI(detections, roi.roiRect, 0.02f);
	cout << "ROI边缘过滤后: " << detections.size() << endl;

	detections = NMS::filterByMask(detections, roi.toothMask);
	cout << "掩膜过滤后: " << detections.size() << endl;

	// 亮度过滤+fallback
	{
		Mat fullGray;
		cvtColor(roi.resizedImage, fullGray, COLOR_BGR2GRAY);
		auto beforeBrightness = detections;
		detections = NMS::filterByBrightness(detections, fullGray, 0.85f);
		cout << "亮度过滤后: " << detections.size() << endl;

		// 如果亮度过滤清空了所有检测，用更宽松的阈值重试，再不行保留top3。
		if (detections.empty() && !beforeBrightness.empty())
		{
			cout << "  亮度过滤清空, 宽松重试(0.75)..." << endl;
			detections = NMS::filterByBrightness(beforeBrightness, fullGray, 0.75f);
			if (detections.empty())
			{
				cout << "  仍为空, 保留置信度top3..." << endl;
				sort(beforeBrightness.begin(), beforeBrightness.end(),
					 [](const Detection& a, const Detection& b)
					 {
						 return a.confidence > b.confidence;
					 });
				int keep = min(3, static_cast<int>(beforeBrightness.size()));
				detections.assign(beforeBrightness.begin(),
								  beforeBrightness.begin() + keep);
			}
			cout << "  fallback后: " << detections.size() << endl;
		}
	}

	detections = NMS::clusterMerge(detections, 70.0f);
	cout << "聚类合并后: " << detections.size() << endl;

	int maxCanals = 5;
	detections = NMS::keepTopK(detections, maxCanals);
	cout << "最终检测数量: " << detections.size() << endl;

	return detections;
}

void drawDetections(Mat& image, const vector<Detection>& detections)
{
	for (int i = 0; i < static_cast<int>(detections.size()); i++)
	{
		const auto& det = detections[i];
		Rect box = det.box & Rect(0, 0, image.cols, image.rows);
		if (box.area() <= 0)
		{
			continue;
		}

		Point c(box.x + box.width / 2, box.y + box.height / 2);

		rectangle(image, box, Scalar(0, 255, 0), 2);

		int crossSize = 10;
		line(image, Point(c.x - crossSize, c.y),
			 Point(c.x + crossSize, c.y), Scalar(0, 0, 255), 2);
		line(image, Point(c.x, c.y - crossSize),
			 Point(c.x, c.y + crossSize), Scalar(0, 0, 255), 2);

		char buf[32];
		snprintf(buf, sizeof(buf), "%.3f", det.confidence);
		putText(image, buf, Point(c.x + 12, c.y - 5),
				FONT_HERSHEY_SIMPLEX, 0.4, Scalar(0, 255, 255), 1);
	}
}
