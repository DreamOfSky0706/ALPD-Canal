#include "nms.h"
#include "detector.h"
#include <algorithm>
#include <cmath>
#include <iostream>

using namespace cv;
using namespace std;

float NMS::computeIoU(const Rect& a, const Rect& b)
{
	int x1 = max(a.x, b.x);
	int y1 = max(a.y, b.y);
	int x2 = min(a.x + a.width, b.x + b.width);
	int y2 = min(a.y + a.height, b.y + b.height);

	int interW = max(0, x2 - x1);
	int interH = max(0, y2 - y1);
	float interArea = static_cast<float>(interW * interH);

	float areaA = static_cast<float>(a.width * a.height);
	float areaB = static_cast<float>(b.width * b.height);
	float unionArea = areaA + areaB - interArea;

	return (unionArea > 0) ? interArea / unionArea : 0.0f;
}

// 先按置信度降序排列，选中当前最高分的检测后，把和它重叠过多的低分检测都标记为抑制。
vector<Detection> NMS::apply(vector<Detection>& detections,
							 float iouThreshold)
{
	if (detections.empty())
	{
		return {};
	}

	sort(detections.begin(), detections.end(),
		 [](const Detection& a, const Detection& b)
		 {
			 return a.confidence > b.confidence;
		 });

	vector<bool> suppressed(detections.size(), false);
	vector<Detection> result;

	for (int i = 0; i < static_cast<int>(detections.size()); i++)
	{
		if (suppressed[i])
		{
			continue;
		}
		result.push_back(detections[i]);
		for (int j = i + 1; j < static_cast<int>(detections.size()); j++)
		{
			if (!suppressed[j] &&
				computeIoU(detections[i].box, detections[j].box) > iouThreshold)
			{
				suppressed[j] = true;
			}
		}
	}
	return result;
}

// 去掉中心点落在ROI边缘附近的检测结果。
vector<Detection> NMS::filterByROI(const vector<Detection>& detections,
								   const Rect& roiRect, float margin)
{
	vector<Detection> filtered;
	int mx = static_cast<int>(roiRect.width * margin);
	int my = static_cast<int>(roiRect.height * margin);
	Rect validArea(roiRect.x + mx, roiRect.y + my,
				   roiRect.width - 2 * mx, roiRect.height - 2 * my);

	for (const auto& det : detections)
	{
		Point center(det.box.x + det.box.width / 2,
					 det.box.y + det.box.height / 2);
		if (validArea.contains(center))
		{
			filtered.push_back(det);
		}
	}
	return filtered;
}

// 按置信度排序后只保留前maxCount个结果，避免输出过多的低质量检测。
vector<Detection> NMS::keepTopK(vector<Detection>& detections, int maxCount)
{
	if (static_cast<int>(detections.size()) <= maxCount)
	{
		return detections;
	}
	sort(detections.begin(), detections.end(),
		 [](const Detection& a, const Detection& b)
		 {
			 return a.confidence > b.confidence;
		 });
	return vector<Detection>(detections.begin(),
							 detections.begin() + maxCount);
}

// 用基于距离的广度优先搜索把空间上相近的检测归为同一簇，然后对每个簇内的检测做加权平均，
// 合并成一个代表性的检测结果，这样多个指向同一根管口的检测会融合成一个。
vector<Detection> NMS::clusterMerge(const vector<Detection>& detections,
									float distThreshold)
{
	if (detections.empty())
	{
		return {};
	}

	int n = static_cast<int>(detections.size());
	vector<int> label(n, -1);
	int clusterId = 0;

	for (int i = 0; i < n; i++)
	{
		if (label[i] >= 0)
		{
			continue;
		}
		label[i] = clusterId;

		vector<int> queue = { i };
		int head = 0;
		while (head < static_cast<int>(queue.size()))
		{
			int cur = queue[head++];
			Point cA(detections[cur].box.x + detections[cur].box.width / 2,
					 detections[cur].box.y + detections[cur].box.height / 2);

			for (int j = 0; j < n; j++)
			{
				if (label[j] >= 0)
				{
					continue;
				}
				Point cB(detections[j].box.x + detections[j].box.width / 2,
						 detections[j].box.y + detections[j].box.height / 2);
				float dist = static_cast<float>(norm(cA - cB));
				if (dist < distThreshold)
				{
					label[j] = clusterId;
					queue.push_back(j);
				}
			}
		}
		clusterId++;
	}

	// 对每个簇，用置信度作为权重对位置和大小做加权平均。
	vector<Detection> result;
	for (int c = 0; c < clusterId; c++)
	{
		float sumX = 0;
		float sumY = 0;
		float sumW = 0;
		float sumH = 0;
		float sumConf = 0;
		float maxConf = -1e9f;
		int count = 0;

		for (int i = 0; i < n; i++)
		{
			if (label[i] != c)
			{
				continue;
			}
			float conf = max(0.01f, detections[i].confidence);
			sumX += static_cast<float>(detections[i].box.x) * conf;
			sumY += static_cast<float>(detections[i].box.y) * conf;
			sumW += static_cast<float>(detections[i].box.width) * conf;
			sumH += static_cast<float>(detections[i].box.height) * conf;
			sumConf += conf;
			if (detections[i].confidence > maxConf)
			{
				maxConf = detections[i].confidence;
			}
			count++;
		}

		if (sumConf > 0)
		{
			Detection avgDet;
			avgDet.box.x = cvRound(sumX / sumConf);
			avgDet.box.y = cvRound(sumY / sumConf);
			avgDet.box.width = cvRound(sumW / sumConf);
			avgDet.box.height = cvRound(sumH / sumConf);
			avgDet.confidence = maxConf;
			result.push_back(avgDet);

			cout << "  聚类 " << c << ": " << count << " 个检测 -> 中心=("
				<< avgDet.box.x + avgDet.box.width / 2 << ","
				<< avgDet.box.y + avgDet.box.height / 2
				<< ") 置信度=" << maxConf << endl;
		}
	}
	return result;
}

// 过滤掉太亮的检测框。
// 根管口是暗区域，如果检测框内平均亮度超过全图均值的一定比例说明大概率是误检。
vector<Detection> NMS::filterByBrightness(
	const vector<Detection>& detections,
	const Mat& grayImage,
	float maxBrightnessRatio)
{
	if (detections.empty() || grayImage.empty())
	{
		return detections;
	}

	float imgMean = static_cast<float>(mean(grayImage)[0]);
	float threshold = imgMean * maxBrightnessRatio;

	vector<Detection> filtered;
	for (const auto& det : detections)
	{
		Rect clipped = det.box & Rect(0, 0, grayImage.cols, grayImage.rows);
		if (clipped.area() <= 0)
		{
			continue;
		}

		float boxMean = static_cast<float>(mean(grayImage(clipped))[0]);
		if (boxMean <= threshold)
		{
			filtered.push_back(det);
		}
		else
		{
			cout << "  已过滤(过亮): 均值=" << boxMean
				<< " > " << threshold
				<< " 位置=(" << det.box.x << "," << det.box.y << ")" << endl;
		}
	}
	return filtered;
}

// 记笔记，根管口一定在牙齿上。
vector<Detection> NMS::filterByMask(
	const vector<Detection>& detections,
	const Mat& mask)
{
	if (detections.empty() || mask.empty())
	{
		return detections;
	}

	vector<Detection> filtered;
	for (const auto& det : detections)
	{
		Point center(det.box.x + det.box.width / 2,
					 det.box.y + det.box.height / 2);
		if (center.y >= 0 && center.y < mask.rows &&
			center.x >= 0 && center.x < mask.cols &&
			mask.at<uchar>(center.y, center.x) > 0)
		{
			filtered.push_back(det);
		}
		else
		{
			cout << "  已过滤(掩膜外): 中心=("
				<< center.x << "," << center.y << ")" << endl;
		}
	}
	return filtered;
}
