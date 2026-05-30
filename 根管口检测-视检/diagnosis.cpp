#include "diagnosis.h"
#include "detector.h"
#include <fstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>

using namespace cv;
using namespace std;

// analyze根据检测到的根管口位置来推断牙齿类型和风险等级。
// 一颗牙有几个根管口是判断牙齿类型的重要依据（前牙通常1个，磨牙3到4个）。
// 对称性评分衡量各根管口到它们重心的距离是否均匀，
// 对称性好说明根管口分布规则，更容易进行常规治疗。
DiagnosticReport Diagnosis::analyze(
	const vector<Detection>& detections,
	const Mat& roiImage,
	const Rect& roiRect)
{
	// roiImage和roiRect貌似用不着，先否了。
	(void)roiImage;
	(void)roiRect;

	DiagnosticReport report;
	report.numCanals = static_cast<int>(detections.size());

	// 从每个检测框计算中心坐标和面积，存入报告供后续使用。
	for (const auto& det : detections)
	{
		Point center(det.box.x + det.box.width / 2,
					 det.box.y + det.box.height / 2);
		report.canalCenters.push_back(center);
		report.canalAreas.push_back(det.box.area());
	}

	// 根据根管数量推测牙齿类型并给出风险等级和建议。
	// 前牙1个根管，前磨牙1到2个，上颌磨牙通常3个，下颌磨牙可能有4个，更多则属于异常。
	switch (report.numCanals)
	{
	case 0:
		report.toothType = "No canal detected";
		report.riskLevel = "High Risk";
		report.advice = "Failed to detect root canals. Manual examination required.";
		break;
	case 1:
		report.toothType = "Single Canal (Anterior/Premolar)";
		report.riskLevel = "Low Risk";
		report.advice = "Single canal detected. Standard endodontic treatment.";
		break;
	case 2:
		report.toothType = "Two Canals (Premolar/Molar)";
		report.riskLevel = "Low Risk";
		report.advice = "Two canals detected. Standard endodontic treatment.";
		break;
	case 3:
		report.toothType = "Three Canals (Maxillary Molar)";
		report.riskLevel = "Low Risk";
		report.advice = "Three canals detected. Standard endodontic treatment.";
		break;
	case 4:
		report.toothType = "Four Canals (MandibuLar Molar / Variant)";
		report.riskLevel = "Medium Risk";
		report.advice = "Four canals detected. Specialist referral recommended.";
		break;
	default:
		report.toothType = "Multiple Canals (Anomaly)";
		report.riskLevel = "High Risk";
		report.advice = "Unusual number of canals. Specialist referral recommended.";
		break;
	}

	// 对称性评分的计算方法：先求所有根管口中心的重心，
	// 然后算每个中心到重心的距离，看这些距离的离散程度。
	// 标准差除以均值就是变异系数，1减去它就是对称性分数，越接近1越对称。
	if (report.numCanals >= 2)
	{
		Point centroid(0, 0);
		for (const auto& c : report.canalCenters)
		{
			centroid.x += c.x;
			centroid.y += c.y;
		}
		centroid.x /= report.numCanals;
		centroid.y /= report.numCanals;

		vector<float> distances;
		for (const auto& c : report.canalCenters)
		{
			float d = static_cast<float>(sqrt(
				static_cast<double>((c.x - centroid.x) * (c.x - centroid.x)) +
				static_cast<double>((c.y - centroid.y) * (c.y - centroid.y))));
			distances.push_back(d);
		}

		float meanDist = accumulate(distances.begin(), distances.end(), 0.0f) /
			static_cast<float>(distances.size());
		float variance = 0.0f;
		for (float d : distances)
		{
			variance += (d - meanDist) * (d - meanDist);
		}
		variance /= static_cast<float>(distances.size());
		float stdDev = sqrt(variance);

		report.symmetryScore = (meanDist > 0) ?
			max(0.0f, 1.0f - stdDev / meanDist) : 0.0f;
	}

	return report;
}

// visualizeReport在原图右侧拼接一个深灰色信息面板，
// 在图像上画出检测框和十字标记，面板上显示诊断的文字信息。
// 这里是完全托付给AI做的。
Mat Diagnosis::visualizeReport(
	const Mat& fullImage,
	const vector<Detection>& detections,
	const DiagnosticReport& report)
{
	int panelW = 400;
	Mat canvas(fullImage.rows,
			   fullImage.cols + panelW,
			   CV_8UC3, Scalar(40, 40, 40));

	fullImage.copyTo(canvas(Rect(0, 0, fullImage.cols, fullImage.rows)));

	for (int i = 0; i < static_cast<int>(detections.size()); i++)
	{
		Scalar color(0, 255, 0);
		rectangle(canvas, detections[i].box, color, 2);

		Point center(detections[i].box.x + detections[i].box.width / 2,
					 detections[i].box.y + detections[i].box.height / 2);
		int cs = 8;
		line(canvas, Point(center.x - cs, center.y),
			 Point(center.x + cs, center.y), Scalar(0, 0, 255), 2);
		line(canvas, Point(center.x, center.y - cs),
			 Point(center.x, center.y + cs), Scalar(0, 0, 255), 2);

		string label = "#" + to_string(i + 1);
		putText(canvas, label,
				Point(detections[i].box.x, detections[i].box.y - 8),
				FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);
	}

	int panelX = fullImage.cols + 20;
	int lineY = 40;
	int lineH = 35;
	Scalar white(255, 255, 255);
	Scalar green(0, 255, 0);
	Scalar yellow(0, 255, 255);
	Scalar red(0, 0, 255);

	putText(canvas, "=== Root Canal Report ===",
			Point(panelX, lineY), FONT_HERSHEY_SIMPLEX, 0.7, yellow, 2);
	lineY += lineH + 10;

	putText(canvas, "Canals: " + to_string(report.numCanals),
			Point(panelX, lineY), FONT_HERSHEY_SIMPLEX, 0.6, white, 1);
	lineY += lineH;

	putText(canvas, "Type: " + report.toothType,
			Point(panelX, lineY), FONT_HERSHEY_SIMPLEX, 0.5, white, 1);
	lineY += lineH;

	Scalar riskColor = green;
	if (report.riskLevel == "Medium Risk")
	{
		riskColor = yellow;
	}
	else if (report.riskLevel == "High Risk")
	{
		riskColor = red;
	}

	putText(canvas, "Risk: " + report.riskLevel,
			Point(panelX, lineY), FONT_HERSHEY_SIMPLEX, 0.6, riskColor, 2);
	lineY += lineH;

	char symBuf[64];
	snprintf(symBuf, sizeof(symBuf), "Symmetry: %.2f", report.symmetryScore);
	putText(canvas, string(symBuf),
			Point(panelX, lineY), FONT_HERSHEY_SIMPLEX, 0.6, white, 1);
	lineY += lineH + 10;

	putText(canvas, "Advice:",
			Point(panelX, lineY), FONT_HERSHEY_SIMPLEX, 0.5, yellow, 1);
	lineY += lineH;

	string adv = report.advice;
	int maxChars = 35;
	while (!adv.empty())
	{
		string line = adv.substr(0, min(static_cast<int>(adv.size()), maxChars));
		putText(canvas, line,
				Point(panelX, lineY), FONT_HERSHEY_SIMPLEX, 0.45, white, 1);
		lineY += 25;
		adv = adv.substr(min(static_cast<int>(adv.size()), maxChars));
	}

	lineY += 15;
	putText(canvas, "--- Canal Details ---",
			Point(panelX, lineY), FONT_HERSHEY_SIMPLEX, 0.5, yellow, 1);
	lineY += lineH;

	for (int i = 0; i < static_cast<int>(report.canalCenters.size()); i++)
	{
		char buf[128];
		snprintf(buf, sizeof(buf), "#%d  Pos=(%d,%d)  Area=%d",
				 i + 1,
				 report.canalCenters[i].x,
				 report.canalCenters[i].y,
				 report.canalAreas[i]);
		putText(canvas, string(buf),
				Point(panelX, lineY), FONT_HERSHEY_SIMPLEX, 0.4, white, 1);
		lineY += 25;
	}

	return canvas;
}

void Diagnosis::saveReport(const DiagnosticReport& report,
						   const string& filepath)
{
	ofstream f(filepath);
	if (!f.is_open())
	{
		cerr << "无法写入报告文件: " << filepath << endl;
		return;
	}

	f << "检测报告" << endl;
	f << endl;
	f << "检测到的根管数量: " << report.numCanals << endl;
	f << "牙齿类型: " << report.toothType << endl;
	f << "对称性评分: " << report.symmetryScore << endl;
	f << "风险等级: " << report.riskLevel << endl;
	f << endl;
	f << "建议: " << report.advice << endl;
	f << endl;
	f << "根管详情:" << endl;
	for (int i = 0; i < static_cast<int>(report.canalCenters.size()); i++)
	{
		f << "  根管 #" << (i + 1)
			<< "  中心=(" << report.canalCenters[i].x
			<< "," << report.canalCenters[i].y
			<< ")  面积=" << report.canalAreas[i] << endl;
	}
	f << endl;
	f << endl;
	f.close();

	cout << "\n检测报告" << endl;
	cout << "根管数: " << report.numCanals << endl;
	cout << "类型: " << report.toothType << endl;
	cout << "风险: " << report.riskLevel << endl;
	cout << "对称性: " << report.symmetryScore << endl;
	cout << "建议: " << report.advice << endl;
	for (int i = 0; i < static_cast<int>(report.canalCenters.size()); i++)
	{
		cout << "  #" << (i + 1)
			<< "  (" << report.canalCenters[i].x
			<< "," << report.canalCenters[i].y
			<< ")  面积=" << report.canalAreas[i] << endl;
	}
}