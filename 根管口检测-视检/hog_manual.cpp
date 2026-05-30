#include "hog_manual.h"
#include <cmath>
#include <algorithm>
#include <numeric>

using namespace cv;
using namespace std;

// 用Sobel算子计算图像在x和y方向上的梯度，
// 然后用cartToPolar转换成极坐标形式得到幅值和方向。
// 对三通道图像会分别处理每个通道，最终每个像素取幅值最大的那个通道。
// 彩色图像用这种方式比先转灰度再算梯度效果更好。
void ManualHOG::computeGradients(const Mat& img,
								 Mat& magnitude, Mat& angle) const
{
	Mat input;

	if (img.channels() == 3)
	{
		vector<Mat> channels;
		split(img, channels);

		magnitude = Mat::zeros(img.size(), CV_32F);
		angle = Mat::zeros(img.size(), CV_32F);

		for (int c = 0; c < 3; c++)
		{
			Mat gx, gy, mag, ang;
			channels[c].convertTo(input, CV_32F);

			// Sobel的ksize设为1对应最简单的差分核，相当于[-1, 0, 1]。
			Sobel(input, gx, CV_32F, 1, 0, 1);
			Sobel(input, gy, CV_32F, 0, 1, 1);

			// 以度为单位输出，范围是0到360。
			cartToPolar(gx, gy, mag, ang, true);

			// 把360度范围的有符号方向转成180度范围的无符号方向，因为HOG不区分梯度的正反。
			for (int y = 0; y < mag.rows; y++)
			{
				for (int x = 0; x < mag.cols; x++)
				{
					float a = ang.at<float>(y, x);
					if (a >= 180.0f)
					{
						a -= 180.0f;
					}

					if (mag.at<float>(y, x) > magnitude.at<float>(y, x))
					{
						magnitude.at<float>(y, x) = mag.at<float>(y, x);
						angle.at<float>(y, x) = a;
					}
				}
			}
		}
	}
	else
	{
		img.convertTo(input, CV_32F);
		Mat gx, gy;
		Sobel(input, gx, CV_32F, 1, 0, 1);
		Sobel(input, gy, CV_32F, 0, 1, 1);

		cartToPolar(gx, gy, magnitude, angle, true);

		for (int y = 0; y < angle.rows; y++)
		{
			for (int x = 0; x < angle.cols; x++)
			{
				float& a = angle.at<float>(y, x);
				if (a >= 180.0f)
				{
					a -= 180.0f;
				}
			}
		}
	}
}

// 对一个cell区域内所有像素的梯度进行统计，按照方向分配到numBins个bin中。
// 这里用了双线性插值，一个像素的梯度幅值会按角度位置的比例分配到相邻两个bin里，这样直方图更平滑。
vector<float> ManualHOG::computeCellHistogram(
	const Mat& mag, const Mat& ang, int cellY, int cellX) const
{
	vector<float> hist(numBins, 0.0f);
	float binWidth = 180.0f / static_cast<float>(numBins);

	int startY = cellY * cellH;
	int startX = cellX * cellW;

	for (int y = startY; y < startY + cellH; y++)
	{
		for (int x = startX; x < startX + cellW; x++)
		{
			float m = mag.at<float>(y, x);
			float a = ang.at<float>(y, x);

			float binPos = a / binWidth - 0.5f;
			int binLow = static_cast<int>(floor(binPos));
			int binHigh = binLow + 1;
			float weight = binPos - static_cast<float>(binLow);

			// 保证bin索引首尾相接。
			binLow = ((binLow % numBins) + numBins) % numBins;
			binHigh = binHigh % numBins;

			hist[binLow] += m * (1.0f - weight);
			hist[binHigh] += m * weight;
		}
	}
	return hist;
}

// 实现L2-Hys归一化。
// 第一步是标准的L2归一化，每个元素除以向量的L2范数，
// 第二步是截断，防止某个方向的梯度过于主导，
// 第三步是再做一次L2归一化恢复单位长度。
// 这种方法对光照变化和局部对比度差异表现不错。
void ManualHOG::normalizeBlock(vector<float>& blockFeat) const
{
	float eps = 1e-6f;

	float sumSq = 0.0f;
	for (float v : blockFeat)
	{
		sumSq += v * v;
	}
	float norm = sqrt(sumSq + eps * eps);
	for (float& v : blockFeat)
	{
		v /= norm;
	}

	for (float& v : blockFeat)
	{
		v = min(v, 0.2f);
	}

	sumSq = 0.0f;
	for (float v : blockFeat)
	{
		sumSq += v * v;
	}
	norm = sqrt(sumSq + eps * eps);
	for (float& v : blockFeat)
	{
		v /= norm;
	}
}

// 最后是完整的HOG特征提取流程。
// 先把输入图像resize到固定窗口大小，算出梯度，
// 然后按cell网格构建所有cell的方向直方图，
// 再用滑动的block把相邻cell的直方图拼起来做归一化，
// 最终所有block的归一化结果拼成一个长向量就是HOG特征。
vector<float> ManualHOG::compute(const Mat& window) const
{
	Mat resized;
	if (window.cols != winW || window.rows != winH)
	{
		resize(window, resized, Size(winW, winH));
	}
	else
	{
		resized = window;
	}

	Mat magnitude, angle;
	computeGradients(resized, magnitude, angle);

	int cellsX = winW / cellW;
	int cellsY = winH / cellH;

	vector<vector<float>> cellHists(static_cast<size_t>(cellsY * cellsX));
	for (int cy = 0; cy < cellsY; cy++)
	{
		for (int cx = 0; cx < cellsX; cx++)
		{
			cellHists[static_cast<size_t>(cy * cellsX + cx)] =
				computeCellHistogram(magnitude, angle, cy, cx);
		}
	}

	int blockCellsX = blockW / cellW;
	int blockCellsY = blockH / cellH;
	int blocksX = (winW - blockW) / strideW + 1;
	int blocksY = (winH - blockH) / strideH + 1;

	vector<float> descriptor;

	for (int by = 0; by < blocksY; by++)
	{
		for (int bx = 0; bx < blocksX; bx++)
		{
			vector<float> blockFeat;
			int cellStartY = by;
			int cellStartX = bx;

			for (int cy = cellStartY; cy < cellStartY + blockCellsY; cy++)
			{
				for (int cx = cellStartX; cx < cellStartX + blockCellsX; cx++)
				{
					const auto& h = cellHists[static_cast<size_t>(cy * cellsX + cx)];
					blockFeat.insert(blockFeat.end(), h.begin(), h.end());
				}
			}

			normalizeBlock(blockFeat);

			descriptor.insert(descriptor.end(),
							  blockFeat.begin(), blockFeat.end());
		}
	}

	return descriptor;
}

Mat ManualHOG::visualize(const Mat& window) const
{
	Mat resized;
	if (window.cols != winW || window.rows != winH)
	{
		resize(window, resized, Size(winW, winH));
	}
	else
	{
		resized = window.clone();
	}

	Mat magnitude, angle;
	computeGradients(resized, magnitude, angle);

	int cellsX = winW / cellW;
	int cellsY = winH / cellH;

	int scale = 10;
	Mat vis = Mat::zeros(winH * scale, winW * scale, CV_8UC3);

	for (int cy = 0; cy < cellsY; cy++)
	{
		for (int cx = 0; cx < cellsX; cx++)
		{
			auto hist = computeCellHistogram(magnitude, angle, cy, cx);

			float maxVal = *max_element(hist.begin(), hist.end());
			if (maxVal < 1e-5f)
			{
				continue;
			}

			float centerX = (cx + 0.5f) * static_cast<float>(cellW * scale);
			float centerY = (cy + 0.5f) * static_cast<float>(cellH * scale);
			float halfLen = static_cast<float>(cellW * scale) * 0.4f;

			for (int b = 0; b < numBins; b++)
			{
				float angleRad = (static_cast<float>(b) * 20.0f + 10.0f) *
					static_cast<float>(CV_PI) / 180.0f;
				float strength = hist[b] / maxVal;

				float dx = cos(angleRad) * halfLen * strength;
				float dy = sin(angleRad) * halfLen * strength;

				Point p1(static_cast<int>(centerX - dx),
						 static_cast<int>(centerY - dy));
				Point p2(static_cast<int>(centerX + dx),
						 static_cast<int>(centerY + dy));

				line(vis, p1, p2, Scalar(0, 255, 0), 1, LINE_AA);
			}
		}
	}
	return vis;
}
