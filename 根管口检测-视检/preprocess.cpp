#include "preprocess.h"
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace cv;
using namespace std;

// 用自适应阈值找出图像中较暗的区域。
// 自适应阈值的好处是它根据像素的局部邻域来决定阈值，不受全局光照不均匀的影响，和cv2.adaptiveThreshold一样。
// 找到的暗区域后续作为牙髓腔的参考信息。
static Mat extractDarkRegions(const Mat& gray)
{
	Mat darkMask;
	adaptiveThreshold(gray, darkMask, 255,
					  ADAPTIVE_THRESH_GAUSSIAN_C,
					  THRESH_BINARY_INV, 81, 5);
	Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
	morphologyEx(darkMask, darkMask, MORPH_CLOSE, kernel);
	cout << "  暗区域像素数: " << countNonZero(darkMask) << " px" << endl;
	return darkMask;
}

// 通过找外轮廓再填充的方式来堵住掩膜中的孔洞。
Mat Preprocessor::fillHoles(const Mat& mask)
{
	vector<vector<Point>> contours;
	findContours(mask.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	Mat filled = Mat::zeros(mask.size(), CV_8UC1);
	for (int i = 0; i < static_cast<int>(contours.size()); i++)
	{
		drawContours(filled, contours, i, Scalar(255), FILLED);
	}
	return filled;
}

// 基于梯度的边界检测分割。
// 核心思路是无论牙齿是什么颜色，牙齿和背景之间总有一条亮度或颜色的过渡边界。
// 用大核高斯模糊消除牙齿内部的纹理和根管口细节，只保留牙齿轮廓边缘，
// 然后用检测这些边缘，在非边缘区域做连通域分析，取包含中心的连通域就是牙齿。
// 这个方法的好处是不依赖颜色，对暗黄牙齿和偏亮牙腔都能正确工作。
static Mat gradientFloodSegmentation(const Mat& bgr, int cx, int cy,
									 const string& debugDir)
{
	int h = bgr.rows;
	int w = bgr.cols;

	Mat gray;
	cvtColor(bgr, gray, COLOR_BGR2GRAY);

	// 用LAB颜色空间的三通道梯度取最大值来检测边缘。
	// 暗黄牙齿在灰度上和背景对比度很低，但在LAB的a*和b*通道上牙齿和背景的色调差异仍然比较明显，用颜色梯度比灰度梯度更可靠。
	Mat lab;
	cvtColor(bgr, lab, COLOR_BGR2Lab);
	vector<Mat> labCh;
	split(lab, labCh);

	Mat gradKernel = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));

	// 在两个尺度上分别计算梯度然后合并，
	// 小尺度找清晰边界，大尺度找模糊渐变的边界。
	int blurSizes[] = { 21, 41 };
	double blurSigmas[] = { 7, 14 };
	Mat maxGrad = Mat::zeros(bgr.rows, bgr.cols, CV_32F);

	for (int si = 0; si < 2; si++)
	{
		for (int c = 0; c < 3; c++)
		{
			Mat chFloat;
			labCh[c].convertTo(chFloat, CV_32F);
			Mat blurred;
			GaussianBlur(chFloat, blurred,
						 Size(blurSizes[si], blurSizes[si]), blurSigmas[si]);
			Mat dilated, eroded;
			dilate(blurred, dilated, gradKernel);
			erode(blurred, eroded, gradKernel);
			Mat grad;
			subtract(dilated, eroded, grad);
			maxGrad = max(maxGrad, grad);
		}
	}

	// 用Otsu自动选取梯度阈值，把强边缘和弱边缘分开。
	Mat gradU8;
	normalize(maxGrad, gradU8, 0, 255, NORM_MINMAX, CV_8UC1);
	Mat edgeMask;
	threshold(gradU8, edgeMask, 0, 255, THRESH_BINARY | THRESH_OTSU);

	// 膨胀边缘让它们连成封闭的轮廓线。
	Mat dk = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
	dilate(edgeMask, edgeMask, dk);

	// 非边缘区域就是可能的牙齿内部。
	Mat nonEdge;
	bitwise_not(edgeMask, nonEdge);

	// 连通域分析，找包含图像中心的那个连通域。
	Mat labelMap;
	int nLabels = connectedComponents(nonEdge, labelMap, 8, CV_32S);

	int centerLabel = labelMap.at<int>(cy, cx);

	// 如果中心恰好落在边缘上，在附近搜索一个有效的连通域。
	if (centerLabel <= 0)
	{
		int searchR = 30;
		double bestArea = 0;
		for (int dy = -searchR; dy <= searchR; dy += 3)
		{
			for (int dx = -searchR; dx <= searchR; dx += 3)
			{
				int ny = cy + dy;
				int nx = cx + dx;
				if (ny >= 0 && ny < h && nx >= 0 && nx < w)
				{
					int lab = labelMap.at<int>(ny, nx);
					if (lab > 0)
					{
						double area = static_cast<double>(countNonZero(labelMap == lab));
						if (area > bestArea)
						{
							bestArea = area;
							centerLabel = lab;
						}
					}
				}
			}
		}
	}

	Mat result = Mat::zeros(h, w, CV_8UC1);
	if (centerLabel > 0)
	{
		result.setTo(255, labelMap == centerLabel);
	}

	// 形态学平滑。
	Mat ck = getStructuringElement(MORPH_ELLIPSE, Size(15, 15));
	morphologyEx(result, result, MORPH_CLOSE, ck);
	Mat ok = getStructuringElement(MORPH_ELLIPSE, Size(9, 9));
	morphologyEx(result, result, MORPH_OPEN, ok);

	result = Preprocessor::fillHoles(result);

	double resCov = static_cast<double>(countNonZero(result)) /
		(static_cast<double>(h) * w);
	cout << "  梯度边界分割覆盖率: " << (resCov * 100) << "%" << endl;

	if (!debugDir.empty())
	{
		imwrite(debugDir + "/mask_grad_edges.png", edgeMask);
		imwrite(debugDir + "/mask_grad_result.png", result);
	}

	return result;
}

// 中心加权颜色分割。
// 在LAB颜色空间中，以图像中心小圆域为种子采样牙齿颜色统计量，
// 然后对每个像素计算与其马氏距离，结合到中心的高斯空间权重，综合得分高的像素判定为牙齿。
// 这种方法比GrabCut快很多，且对黄色偏亮背景有更好的区分能力，
// 因为LAB空间中亮度和色度是分开的。
static Mat centerWeightedSegmentation(const Mat& bgr, int cx, int cy,
									  const Mat& coreSeed, const string& debugDir)
{
	int h = bgr.rows;
	int w = bgr.cols;

	Mat gray;
	cvtColor(bgr, gray, COLOR_BGR2GRAY);

	// 转LAB颜色空间，亮度和色度分离，对光照变化表现更好。
	Mat lab;
	cvtColor(bgr, lab, COLOR_BGR2Lab);
	vector<Mat> labCh;
	split(lab, labCh);

	// 如果有颜色法的部分结果且覆盖了中心就用它，否则用中心小圆。
	Mat seedMask;
	bool usedCoreSeed = false;
	if (!coreSeed.empty() && countNonZero(coreSeed) > 500)
	{
		// 用腐蚀后的种子，确保采样到的是确定的牙齿内部像素。
		Mat ek = getStructuringElement(MORPH_ELLIPSE, Size(21, 21));
		erode(coreSeed, seedMask, ek);
		if (countNonZero(seedMask) > 200)
		{
			usedCoreSeed = true;
		}
	}
	if (!usedCoreSeed)
	{
		seedMask = Mat::zeros(h, w, CV_8UC1);
		int seedR = min(w, h) / 6;
		circle(seedMask, Point(cx, cy), seedR, Scalar(255), FILLED);
	}

	// 从种子区域采样牙齿颜色的均值和标准差。
	Scalar meanL, meanA, meanB;
	Scalar stdL, stdA, stdB;
	Scalar dummy;
	meanL = mean(labCh[0], seedMask);
	meanA = mean(labCh[1], seedMask);
	meanB = mean(labCh[2], seedMask);
	meanStdDev(labCh[0], dummy, stdL, seedMask);
	meanStdDev(labCh[1], dummy, stdA, seedMask);
	meanStdDev(labCh[2], dummy, stdB, seedMask);

	// 标准差下限，避免除零。
	double sL = max(5.0, stdL[0]);
	double sA = max(3.0, stdA[0]);
	double sB = max(3.0, stdB[0]);

	cout << "  中心加权分割: 种子L=" << meanL[0] << " a=" << meanA[0]
		<< " b=" << meanB[0] << " stdL=" << sL << " stdA=" << sA
		<< " stdB=" << sB
		<< (usedCoreSeed ? " (用颜色种子)" : " (用中心圆)") << endl;

	// 离中心越远权重越低。先降一下权重，给后续处理留下空间。
	float sigX = w * 0.40f;
	float sigY = h * 0.40f;

	Mat score = Mat::zeros(h, w, CV_32F);
	for (int y = 0; y < h; y++)
	{
		const uchar* pL = labCh[0].ptr<uchar>(y);
		const uchar* pA = labCh[1].ptr<uchar>(y);
		const uchar* pB = labCh[2].ptr<uchar>(y);
		float* pS = score.ptr<float>(y);

		for (int x = 0; x < w; x++)
		{
			// 归一化颜色距离。
			float dL = static_cast<float>((pL[x] - meanL[0]) / (sL * 2.0));
			float da = static_cast<float>((pA[x] - meanA[0]) / (sA * 2.0));
			float db = static_cast<float>((pB[x] - meanB[0]) / (sB * 2.0));
			float colorDist = sqrt(dL * dL + da * da + db * db);

			// 空间权重。
			float dx = static_cast<float>(x - cx);
			float dy = static_cast<float>(y - cy);
			float spatialW = exp(-(dx * dx) / (2.0f * sigX * sigX)
								 - (dy * dy) / (2.0f * sigY * sigY));

			// 颜色相似度越高且越靠近中心则得分越高。
			pS[x] = spatialW * exp(-colorDist * 1.0f);
		}
	}

	// 归一化，然后用Otsu找最佳阈值。
	Mat scoreU8;
	normalize(score, scoreU8, 0, 255, NORM_MINMAX, CV_8UC1);
	Mat blurred;
	GaussianBlur(scoreU8, blurred, Size(15, 15), 3);
	Mat binary;
	threshold(blurred, binary, 0, 255, THRESH_BINARY | THRESH_OTSU);

	// 形态学清理。
	Mat closeK = getStructuringElement(MORPH_ELLIPSE, Size(25, 25));
	morphologyEx(binary, binary, MORPH_CLOSE, closeK);
	Mat openK = getStructuringElement(MORPH_ELLIPSE, Size(11, 11));
	morphologyEx(binary, binary, MORPH_OPEN, openK);

	// 取包含中心的最大连通域。
	vector<vector<Point>> contours;
	findContours(binary.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	Mat result = Mat::zeros(h, w, CV_8UC1);
	if (!contours.empty())
	{
		Point center(cx, cy);
		int bestIdx = -1;
		double bestArea = 0;

		for (int i = 0; i < static_cast<int>(contours.size()); i++)
		{
			double area = contourArea(contours[i]);
			if (area > bestArea &&
				pointPolygonTest(contours[i],
								 Point2f(static_cast<float>(center.x),
										 static_cast<float>(center.y)),
								 false) >= 0)
			{
				bestArea = area;
				bestIdx = i;
			}
		}
		if (bestIdx < 0)
		{
			bestArea = 0;
			for (int i = 0; i < static_cast<int>(contours.size()); i++)
			{
				double a = contourArea(contours[i]);
				if (a > bestArea)
				{
					bestArea = a;
					bestIdx = i;
				}
			}
		}
		if (bestIdx >= 0)
		{
			drawContours(result, contours, bestIdx, Scalar(255), FILLED);
		}

		double resCov = bestArea / (static_cast<double>(h) * w);
		cout << "  中心加权分割覆盖率: " << (resCov * 100) << "%" << endl;
	}

	if (!debugDir.empty())
	{
		Mat vis = bgr.clone();
		Mat ov = Mat::zeros(bgr.size(), CV_8UC3);
		ov.setTo(Scalar(0, 255, 0), result);
		addWeighted(vis, 0.6, ov, 0.4, 0, vis);
		imwrite(debugDir + "/mask_cw_result.png", vis);
		imwrite(debugDir + "/mask_cw_score.png", scoreU8);
	}

	return result;
}

// 当掩膜面积过小时，从现有掩膜向外扩展。
// 扩展在两个条件同时满足时才继续：
// 1.新增像素的亮度不能太暗
// 2.新增像素不能跨越强梯度边界
// 我感觉用梯度停止比纯亮度停止更可靠，因为牙齿与背景的边界总有梯度，即使二者亮度接近。
static Mat expandSmallMask(const Mat& mask, const Mat& gray,
						   double targetCoverage, const string& debugDir)
{
	if (mask.empty() || gray.empty())
	{
		return mask;
	}

	int h = gray.rows;
	int w = gray.cols;
	double imgArea = static_cast<double>(h) * w;
	double currentCov = static_cast<double>(countNonZero(mask)) / imgArea;

	if (currentCov >= targetCoverage)
	{
		return mask;
	}

	cout << "  掩膜扩展: 当前覆盖率" << (currentCov * 100)
		<< "% < 目标" << (targetCoverage * 100) << "%" << endl;

	// 用掩膜内部的亮度统计来设定扩展门槛。
	Scalar mu, sd;
	meanStdDev(gray, mu, sd, mask);
	double innerMean = mu[0];
	double innerStd = sd[0];

	// 对于内部很亮的情况，牙齿表面比里面暗但仍应该被纳入掩膜。
	double expandThresh = max(25.0, innerMean - 2.0 * innerStd);

	// 计算梯度用于边界检测。
	Mat blurForGrad;
	GaussianBlur(gray, blurForGrad, Size(15, 15), 5);
	Mat gx, gy, gradMag;
	Sobel(blurForGrad, gx, CV_32F, 1, 0, 3);
	Sobel(blurForGrad, gy, CV_32F, 0, 1, 3);
	magnitude(gx, gy, gradMag);
	Scalar gMu, gSd;
	meanStdDev(gradMag, gMu, gSd);
	double gradThresh = gMu[0] + gSd[0];
	Mat gradU8;
	normalize(gradMag, gradU8, 0, 255, NORM_MINMAX, CV_8UC1);
	Mat gradStop;
	threshold(gradU8, gradStop, gradThresh * 255.0 / (gMu[0] + 3 * gSd[0] + 1e-6),
			  255, THRESH_BINARY);
	Mat gdk = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	dilate(gradStop, gradStop, gdk);
	Mat canExpand;
	bitwise_not(gradStop, canExpand);

	cout << "  扩展亮度门槛: " << expandThresh
		<< " (内部均值=" << innerMean << " 标准差=" << innerStd << ")" << endl;

	Mat expanded = mask.clone();
	Mat dilK = getStructuringElement(MORPH_ELLIPSE, Size(11, 11));

	// 迭代扩展，每步膨胀一小圈。
	for (int iter = 0; iter < 50; iter++)
	{
		Mat dilated;
		dilate(expanded, dilated, dilK);

		// 新增区域 = 膨胀后 - 当前。
		Mat newRegion;
		subtract(dilated, expanded, newRegion);

		// 亮度达标。
		Mat brightEnough;
		threshold(gray, brightEnough, expandThresh, 255, THRESH_BINARY);

		// 不跨越强梯度。
		Mat validNew;
		bitwise_and(newRegion, brightEnough, validNew);
		bitwise_and(validNew, canExpand, validNew);

		if (countNonZero(validNew) < 50)
		{
			break;
		}

		bitwise_or(expanded, validNew, expanded);

		double newCov = static_cast<double>(countNonZero(expanded)) / imgArea;
		if (newCov >= targetCoverage)
		{
			break;
		}
	}

	// 平滑
	Mat smoothK = getStructuringElement(MORPH_ELLIPSE, Size(11, 11));
	morphologyEx(expanded, expanded, MORPH_CLOSE, smoothK);
	expanded = Preprocessor::fillHoles(expanded);

	double finalCov = static_cast<double>(countNonZero(expanded)) / imgArea;
	cout << "  扩展后覆盖率: " << (finalCov * 100) << "%" << endl;

	if (!debugDir.empty())
	{
		imwrite(debugDir + "/mask_04b_expanded.png", expanded);
		imwrite(debugDir + "/mask_04b_gradstop.png", gradStop);
	}

	return expanded;
}

// 对掩膜做凸包收紧+亮度边界清理+腐蚀，把边界上的不规则突起和背景残留都削掉。
// 无论哪条路径产出的掩膜都经过这一步，确保最终掩膜边界干净且不会延伸到牙齿外。
// 如果收紧后面积太小则说明收紧过度，就要回退。
static Mat tightenMask(const Mat& mask, int erodeSize,
					   const Mat& gray, const string& debugDir)
{
	if (mask.empty() || countNonZero(mask) < 100)
	{
		return mask;
	}

	int origPx = countNonZero(mask);

	vector<vector<Point>> contours;
	findContours(mask.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	if (contours.empty())
	{
		return mask;
	}

	// 找最大轮廓。
	int maxIdx = 0;
	double maxArea = 0;
	for (int i = 0; i < static_cast<int>(contours.size()); i++)
	{
		double a = contourArea(contours[i]);
		if (a > maxArea)
		{
			maxArea = a;
			maxIdx = i;
		}
	}

	// 去除轮廓上不规则的凹陷和突起。
	vector<Point> hull;
	convexHull(contours[maxIdx], hull);

	Mat hullMask = Mat::zeros(mask.size(), CV_8UC1);
	vector<vector<Point>> hullVec = { hull };
	drawContours(hullMask, hullVec, 0, Scalar(255), FILLED);

	// 与原始掩膜取交集。
	Mat tightened;
	bitwise_and(hullMask, mask, tightened);

	// 掩膜边缘附近如果像素亮度远低于掩膜内部均值，说明这些像素更像背景。
	// 把掩膜边缘向内腐蚀一小圈得到边界，在边界中去掉过暗的像素。
	if (!gray.empty())
	{
		// 计算掩膜内部核心区域的亮度均值作为参考。
		Mat coreK = getStructuringElement(MORPH_ELLIPSE, Size(31, 31));
		Mat coreMask;
		erode(tightened, coreMask, coreK);
		double coreMean = 128;
		if (countNonZero(coreMask) > 100)
		{
			coreMean = mean(gray, coreMask)[0];
		}

		// 边界带 = 原掩膜 - 腐蚀后掩膜。
		Mat borderK = getStructuringElement(MORPH_ELLIPSE, Size(21, 21));
		Mat innerMask;
		erode(tightened, innerMask, borderK);
		Mat borderZone;
		subtract(tightened, innerMask, borderZone);

		// 在边界带中，亮度低于核心均值减30的像素视为背景，从掩膜中去掉。
		double darkThresh = max(40.0, coreMean - 30.0);
		Mat darkBorder;
		threshold(gray, darkBorder, darkThresh, 255, THRESH_BINARY_INV);
		Mat toRemove;
		bitwise_and(darkBorder, borderZone, toRemove);
		subtract(tightened, toRemove, tightened);

		int removedPx = countNonZero(toRemove);
		if (removedPx > 0)
		{
			cout << "  亮度边界清理: 移除 " << removedPx << " px" << endl;
		}
	}

	// 腐蚀收缩边界
	if (erodeSize > 0)
	{
		Mat ek = getStructuringElement(MORPH_ELLIPSE, Size(erodeSize, erodeSize));
		erode(tightened, tightened, ek);
	}

	// 最终平滑
	Mat sk = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
	morphologyEx(tightened, tightened, MORPH_CLOSE, sk);
	morphologyEx(tightened, tightened, MORPH_OPEN, sk);

	// 去除碎片只保留最大连通域。
	{
		vector<vector<Point>> tc;
		findContours(tightened.clone(), tc, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		if (tc.size() > 1)
		{
			Mat cleaned = Mat::zeros(tightened.size(), CV_8UC1);
			int mi = 0;
			double ma = 0;
			for (int i = 0; i < static_cast<int>(tc.size()); i++)
			{
				double a = contourArea(tc[i]);
				if (a > ma)
				{
					ma = a; mi = i;
				}
			}
			drawContours(cleaned, tc, mi, Scalar(255), FILLED);
			tightened = cleaned;
		}
	}

	int tightPx = countNonZero(tightened);
	double retainRatio = static_cast<double>(tightPx) / max(1, origPx);

	// 收紧过度回退。
	if (retainRatio < 0.40 || tightPx < 500)
	{
		cout << "  掩膜收紧过度 (保留" << (retainRatio * 100)
			<< "%)，回退到轻度腐蚀" << endl;
		Mat lightErode;
		Mat lek = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
		erode(mask, lightErode, lek);
		tightPx = countNonZero(lightErode);
		cout << "  轻度腐蚀后: " << origPx << " -> " << tightPx << " px" << endl;

		if (!debugDir.empty())
		{
			imwrite(debugDir + "/mask_08_tightened.png", lightErode);
		}

		return lightErode;
	}

	if (!debugDir.empty())
	{
		imwrite(debugDir + "/mask_08_tightened.png", tightened);
	}

	cout << "  掩膜收紧: " << origPx << " -> " << tightPx
		<< " px (保留" << (retainRatio * 100) << "%)" << endl;

	return tightened;
}

// 在颜色分割法效果不好时留一个备选方案。
// GrabCut其实在OpenCV里直接提供了接口，但为了加速运算，先把图像缩小到一半再GrabCut，完成后放大回原尺寸。
// 初始化时把图像边框和四角标记为确定背景，中心区域标记为可能前景，
// 如果有颜色法的部分结果，可以作为确定前景的种子提供给GrabCut。
static Mat grabCutFallback(const Mat& bgr, const Mat& coreSeed,
						   int cx, int cy, const string& debugDir)
{
	int h = bgr.rows;
	int w = bgr.cols;

	double scale = 0.5;
	Mat smallBgr;
	resize(bgr, smallBgr, Size(), scale, scale);
	int sh = smallBgr.rows;
	int sw = smallBgr.cols;
	int scx = static_cast<int>(cx * scale);
	int scy = static_cast<int>(cy * scale);

	// 初始化GrabCut掩码，默认所有像素为可能背景。
	Mat gcMask(sh, sw, CV_8UC1, Scalar(GC_PR_BGD));

	// 牙齿不会贴着图片边缘，因此图像边框可以放心大胆设为确定背景。
	int bw = 8;
	gcMask(Rect(0, 0, sw, bw)).setTo(Scalar(GC_BGD));
	gcMask(Rect(0, sh - bw, sw, bw)).setTo(Scalar(GC_BGD));
	gcMask(Rect(0, 0, bw, sh)).setTo(Scalar(GC_BGD));
	gcMask(Rect(sw - bw, 0, bw, sh)).setTo(Scalar(GC_BGD));

	// 四角区域也设为确定背景，进一步约束。
	int cornerSz = max(sw, sh) / 8;
	gcMask(Rect(0, 0, cornerSz, cornerSz)).setTo(Scalar(GC_BGD));
	gcMask(Rect(sw - cornerSz, 0, cornerSz, cornerSz)).setTo(Scalar(GC_BGD));
	gcMask(Rect(0, sh - cornerSz, cornerSz, cornerSz)).setTo(Scalar(GC_BGD));
	gcMask(Rect(sw - cornerSz, sh - cornerSz, cornerSz, cornerSz)).setTo(Scalar(GC_BGD));

	// 中心区域设为可能前景。
	int rw = static_cast<int>(sw * 0.50);
	int rh = static_cast<int>(sh * 0.50);
	Rect centerRect(scx - rw / 2, scy - rh / 2, rw, rh);
	centerRect &= Rect(0, 0, sw, sh);
	gcMask(centerRect).setTo(Scalar(GC_PR_FGD));

	// 如果有种子掩膜，缩放后标记为确定前景，能帮助GrabCut更准确地收敛。
	if (!coreSeed.empty() && countNonZero(coreSeed) > 0)
	{
		Mat smallSeed;
		resize(coreSeed, smallSeed, Size(sw, sh), 0, 0, INTER_NEAREST);
		Mat ek = getStructuringElement(MORPH_ELLIPSE, Size(11, 11));
		erode(smallSeed, smallSeed, ek);
		if (countNonZero(smallSeed) > 100)
		{
			gcMask.setTo(GC_FGD, smallSeed);
		}
	}

	Mat bgModel, fgModel;
	cout << "  正在运行GrabCut (尺寸=" << sw << "x" << sh << ")..." << endl;
	grabCut(smallBgr, gcMask, Rect(), bgModel, fgModel, 5, GC_INIT_WITH_MASK);
	cout << "  GrabCut完成." << endl;

	// 把GrabCut判定为前景的像素提取出来。
	Mat smallResult = Mat::zeros(sh, sw, CV_8UC1);
	smallResult.setTo(255, (gcMask == GC_FGD) | (gcMask == GC_PR_FGD));

	Mat result;
	resize(smallResult, result, Size(w, h), 0, 0, INTER_NEAREST);

	if (!debugDir.empty())
	{
		Mat gcVis = bgr.clone();
		Mat gcOv = Mat::zeros(bgr.size(), CV_8UC3);
		gcOv.setTo(Scalar(0, 255, 0), result);
		addWeighted(gcVis, 0.6, gcOv, 0.4, 0, gcVis);
		imwrite(debugDir + "/mask_gc_result.png", gcVis);
	}

	return result;
}

// 从多个候选掩膜中选择最佳的一个。
// 评判标准：覆盖率合理、包含图像中心、形状紧凑。
// 如果某个候选明显优于其他候选就选它，否则优先选覆盖率更合理的。
static int selectBestMask(const vector<Mat>& masks, const vector<string>& names,
						  int cx, int cy, double imgArea)
{
	int bestIdx = -1;
	double bestScore = -1;

	for (int i = 0; i < static_cast<int>(masks.size()); i++)
	{
		if (masks[i].empty() || countNonZero(masks[i]) < 100)
		{
			continue;
		}

		double coverage = static_cast<double>(countNonZero(masks[i])) / imgArea;

		// 覆盖率不在合理范围的直接跳过。
		if (coverage < 0.08 || coverage > 0.75)
		{
			cout << "    " << names[i] << ": 覆盖率" << (coverage * 100)
				<< "% 不合理, 跳过" << endl;
			continue;
		}

		// 中心是否被覆盖。
		bool centerOk = false;
		if (cy >= 0 && cy < masks[i].rows && cx >= 0 && cx < masks[i].cols)
		{
			centerOk = (masks[i].at<uchar>(cy, cx) > 0);
		}
		if (!centerOk)
		{
			cout << "    " << names[i] << ": 中心未覆盖, 跳过" << endl;
			continue;
		}

		// 凸度越接近1形状越规则。
		double convexity = 0.5;
		{
			vector<vector<Point>> ctrs;
			findContours(masks[i].clone(), ctrs, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
			if (!ctrs.empty())
			{
				int mi = 0;
				double ma = 0;
				for (int j = 0; j < static_cast<int>(ctrs.size()); j++)
				{
					double a = contourArea(ctrs[j]);
					if (a > ma)
					{
						ma = a; mi = j;
					}
				}
				vector<Point> hull;
				convexHull(ctrs[mi], hull);
				double hullArea = contourArea(hull);
				if (hullArea > 0)
				{
					convexity = ma / hullArea;
				}
			}
		}

		// 综合评分
		double covScore = 1.0 - abs(coverage - 0.35) / 0.35;
		covScore = max(0.0, covScore);
		double score = covScore * 0.6 + convexity * 0.4;

		cout << "    " << names[i] << ": 覆盖率=" << (coverage * 100)
			<< "% 凸度=" << convexity
			<< " 综合=" << score << endl;

		if (score > bestScore)
		{
			bestScore = score;
			bestIdx = i;
		}
	}

	return bestIdx;
}

// extractToothMask大致分为五个步骤：
// 1.用红蓝差、亮度、红绿差做初步分割得到种子掩膜；
// 2.从种子中取包含图像中心的最大连通域；
// 3.检查分割质量；
// 4.根据质量检查结果选择直接使用颜色法结果还是回退到GrabCut；
// 5.做最终的形态学平滑和连通域清理。
// 单一方法很难在所有图像上都表现良好。
Mat Preprocessor::extractToothMask(const Mat& bgr, const string& debugDir)
{
	int h = bgr.rows;
	int w = bgr.cols;
	double imgArea = static_cast<double>(h) * w;
	int cx = w / 2;
	int cy = h / 2;

	Mat gray;
	cvtColor(bgr, gray, COLOR_BGR2GRAY);
	vector<Mat> ch;
	split(bgr, ch);

	// 1.利用多种颜色特征提取牙齿种子区域。
	// 取它们的并集作为初始掩膜。
	Mat rbDiff;
	subtract(ch[2], ch[0], rbDiff);
	Mat featA;
	threshold(rbDiff, featA, 12, 255, THRESH_BINARY);

	Mat featB;
	threshold(gray, featB, 160, 255, THRESH_BINARY);

	Mat rgDiff;
	subtract(ch[2], ch[1], rgDiff);
	Mat rgMask, gMask, featC;
	threshold(rgDiff, rgMask, 3, 255, THRESH_BINARY);
	threshold(ch[1], gMask, 110, 255, THRESH_BINARY);
	bitwise_and(rgMask, gMask, featC);

	Mat rawMask;
	bitwise_or(featA, featB, rawMask);
	bitwise_or(rawMask, featC, rawMask);

	// 先开后闭去噪点。
	Mat openK = getStructuringElement(MORPH_ELLIPSE, Size(9, 9));
	morphologyEx(rawMask, rawMask, MORPH_OPEN, openK);
	Mat closeK = getStructuringElement(MORPH_ELLIPSE, Size(25, 25));
	morphologyEx(rawMask, rawMask, MORPH_CLOSE, closeK);

	if (!debugDir.empty())
	{
		imwrite(debugDir + "/mask_01a_featA_RB.png", featA);
		imwrite(debugDir + "/mask_01b_featB_bright.png", featB);
		imwrite(debugDir + "/mask_01c_featC_RG.png", featC);
		imwrite(debugDir + "/mask_01d_raw_union.png", rawMask);
	}

	// 2.从所有连通域中取包含图像中心的最大一个。
	// 优先选包含中心的那个，如果不包含中心就选面积最大的。
	vector<vector<Point>> contours;
	findContours(rawMask.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	Mat largestMask = Mat::zeros(bgr.size(), CV_8UC1);
	double bestArea = 0;
	int bestIdx = -1;

	if (!contours.empty())
	{
		Point center(cx, cy);
		for (int i = 0; i < static_cast<int>(contours.size()); i++)
		{
			double area = contourArea(contours[i]);
			if (area > bestArea &&
				pointPolygonTest(contours[i],
								 Point2f(static_cast<float>(center.x),
										 static_cast<float>(center.y)),
								 false) >= 0)
			{
				bestArea = area;
				bestIdx = i;
			}
		}
		if (bestIdx < 0)
		{
			bestArea = 0;
			for (int i = 0; i < static_cast<int>(contours.size()); i++)
			{
				double a = contourArea(contours[i]);
				if (a > bestArea)
				{
					bestArea = a;
					bestIdx = i;
				}
			}
		}
		if (bestIdx >= 0)
		{
			drawContours(largestMask, contours, bestIdx, Scalar(255), FILLED);
		}

		cout << "  步骤2 最大连通域占比: " << (100.0 * bestArea / imgArea) << "%" << endl;
	}

	if (!debugDir.empty())
	{
		imwrite(debugDir + "/mask_02_largest.png", largestMask);
	}

	Mat filledMask = fillHoles(largestMask);

	if (!debugDir.empty())
	{
		imwrite(debugDir + "/mask_03_filled.png", filledMask);
	}

	// 3.质量检查，判断颜色分割是否成功。
	// 如果覆盖率太大说明没有区分能力，太小说明太严格，中心没被覆盖可能选了背景。
	// 这些情况都需要回退到GrabCut。
	double coverage = static_cast<double>(countNonZero(filledMask)) / imgArea;
	bool centerCovered = (filledMask.at<uchar>(cy, cx) > 0);

	double meanInside = 0;
	double meanOutside = 0;
	if (countNonZero(filledMask) > 0 && countNonZero(~filledMask) > 0)
	{
		meanInside = mean(gray, filledMask)[0];
		Mat invMask;
		bitwise_not(filledMask, invMask);
		meanOutside = mean(gray, invMask)[0];
	}
	double colorContrast = abs(meanInside - meanOutside);

	cout << "  质量检查:" << endl;
	cout << "    覆盖率:       " << (coverage * 100) << "%" << endl;
	cout << "    中心被覆盖:   " << (centerCovered ? "是" : "否") << endl;
	cout << "    颜色对比度:   " << colorContrast
		<< " (内部=" << meanInside << " 外部=" << meanOutside << ")" << endl;

	bool needFallback = false;
	string fbReason;

	if (coverage > 0.70)
	{
		needFallback = true;
		fbReason = "覆盖率过大 (" + to_string(static_cast<int>(coverage * 100)) + "%)";
	}
	else if (coverage < 0.10)
	{
		needFallback = true;
		fbReason = "覆盖率过小 (" + to_string(static_cast<int>(coverage * 100)) + "%)";
	}
	else if (!centerCovered)
	{
		needFallback = true;
		fbReason = "中心未被覆盖";
	}

	// 4.选择分割路径。
	// 一种是颜色法成功的情况，直接修剪边界后使用；
	// 另一种是颜色法失败需要回退的情况，同时运行梯度边界法和中心加权法，选择结果更好的那个。
	// 如果都不行再用GrabCut兜底，最终兜底用Otsu全局阈值。
	Mat resultMask;

	if (!needFallback)
	{
		cout << "  路径A: 颜色法分割成功" << endl;

		// 路径A的掩膜做一次亮度预清理，在掩膜边缘找到明显偏暗的像素直接去掉，
		// 防止颜色特征把暗背景角落误纳入。
		Mat borderK = getStructuringElement(MORPH_ELLIPSE, Size(15, 15));
		Mat innerA;
		erode(filledMask, innerA, borderK);
		Mat borderA;
		subtract(filledMask, innerA, borderA);

		double coreAMean = 128;
		if (countNonZero(innerA) > 100)
		{
			coreAMean = mean(gray, innerA)[0];
		}
		double darkA = max(40.0, coreAMean - 35.0);
		Mat darkBorderA;
		threshold(gray, darkBorderA, darkA, 255, THRESH_BINARY_INV);
		Mat toRemoveA;
		bitwise_and(darkBorderA, borderA, toRemoveA);
		Mat cleaned;
		subtract(filledMask, toRemoveA, cleaned);

		Mat erodeK = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));
		erode(cleaned, resultMask, erodeK);

		// 清理后如果变得太小说明颜色法本身有问题，应该转入路径B重新处理。
		double postCleanCov = static_cast<double>(countNonZero(resultMask)) / imgArea;
		if (postCleanCov < 0.10)
		{
			cout << "  路径A清理后覆盖率过低(" << (postCleanCov * 100)
				<< "%)，转入路径B" << endl;
			needFallback = true;
			fbReason = "路径A清理后覆盖率不足";
		}
	}

	if (needFallback)
	{
		cout << "  路径B: 回退 (" << fbReason << ")" << endl;

		// 同时运行梯度边界法和中心加权法，然后选最佳结果。
		// 梯度边界法不依赖颜色，对暗黄牙齿最有效；
		// 中心加权法对颜色有区分度的图像更精确。
		// 两者互补，选评分更高的那个。
		cout << "  运行梯度边界分割..." << endl;
		Mat gradResult = gradientFloodSegmentation(bgr, cx, cy, debugDir);
		Mat gradFilled = fillHoles(gradResult);

		cout << "  运行中心加权分割..." << endl;
		Mat seedForCW;
		if (centerCovered && coverage > 0.05 && coverage < 0.55)
		{
			seedForCW = filledMask.clone();
		}
		Mat cwResult = centerWeightedSegmentation(bgr, cx, cy, seedForCW, debugDir);
		Mat cwFilled = fillHoles(cwResult);

		// 选择最佳候选。
		cout << "  选择最佳掩膜候选:" << endl;
		vector<Mat> candidates = { gradFilled, cwFilled };
		vector<string> candNames = { "梯度边界", "中心加权" };

		int selected = selectBestMask(candidates, candNames, cx, cy, imgArea);

		if (selected >= 0)
		{
			cout << "  路径B: 选择 " << candNames[selected] << endl;

			// 取包含中心的最大连通域。
			Mat chosen = candidates[selected];
			contours.clear();
			findContours(chosen.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

			Mat bestMask = Mat::zeros(bgr.size(), CV_8UC1);
			if (!contours.empty())
			{
				Point center(cx, cy);
				int bIdx = -1;
				double bArea = 0;
				for (int i = 0; i < static_cast<int>(contours.size()); i++)
				{
					double area = contourArea(contours[i]);
					if (area > bArea &&
						pointPolygonTest(contours[i],
										 Point2f(static_cast<float>(center.x),
												 static_cast<float>(center.y)),
										 false) >= 0)
					{
						bArea = area;
						bIdx = i;
					}
				}
				if (bIdx < 0)
				{
					bArea = 0;
					for (int i = 0; i < static_cast<int>(contours.size()); i++)
					{
						double a = contourArea(contours[i]);
						if (a > bArea)
						{
							bArea = a; bIdx = i;
						}
					}
				}
				if (bIdx >= 0)
				{
					drawContours(bestMask, contours, bIdx, Scalar(255), FILLED);
				}
			}
			resultMask = fillHoles(bestMask);
		}
		else
		{
			cout << "  路径B: 梯度和中心加权均不理想, 回退到GrabCut" << endl;

			// 路径B第三选择是GrabCut。
			Mat seedForGC;
			if (centerCovered && coverage > 0.05 && coverage < 0.55)
			{
				seedForGC = filledMask.clone();
			}
			else
			{
				seedForGC = Mat::zeros(bgr.size(), CV_8UC1);
				int seedR = min(w, h) / 8;
				circle(seedForGC, Point(cx, cy), seedR, Scalar(255), FILLED);
			}

			Mat gcResult = grabCutFallback(bgr, seedForGC, cx, cy, debugDir);
			Mat gcFilled = fillHoles(gcResult);

			// 取包含中心的最大连通域。
			contours.clear();
			findContours(gcFilled.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

			Mat gcLargest = Mat::zeros(bgr.size(), CV_8UC1);
			if (!contours.empty())
			{
				Point center(cx, cy);
				int gBestIdx = -1;
				double gBestArea = 0;

				for (int i = 0; i < static_cast<int>(contours.size()); i++)
				{
					double area = contourArea(contours[i]);
					if (area > gBestArea &&
						pointPolygonTest(contours[i],
										 Point2f(static_cast<float>(center.x),
												 static_cast<float>(center.y)),
										 false) >= 0)
					{
						gBestArea = area;
						gBestIdx = i;
					}
				}
				if (gBestIdx < 0)
				{
					gBestArea = 0;
					for (int i = 0; i < static_cast<int>(contours.size()); i++)
					{
						double a = contourArea(contours[i]);
						if (a > gBestArea)
						{
							gBestArea = a; gBestIdx = i;
						}
					}
				}
				if (gBestIdx >= 0)
				{
					drawContours(gcLargest, contours, gBestIdx, Scalar(255), FILLED);
				}

				double gcCoverage = gBestArea / imgArea;
				cout << "  GrabCut覆盖率: " << (gcCoverage * 100) << "%" << endl;
			}

			resultMask = fillHoles(gcLargest);

			if (!debugDir.empty())
			{
				imwrite(debugDir + "/mask_gc_largest.png", resultMask);
			}

			// 如果GrabCut结果也不理想，用Otsu阈值分割。
			double gcCov = static_cast<double>(countNonZero(resultMask)) / imgArea;
			if (gcCov > 0.70 || gcCov < 0.08)
			{
				cerr << "  警告: GrabCut也失败了 (覆盖率=" << (gcCov * 100)
					<< "%), 回退到Otsu阈值分割" << endl;

				Mat blur;
				GaussianBlur(gray, blur, Size(7, 7), 2);
				Mat otsuBin;
				threshold(blur, otsuBin, 0, 255, THRESH_BINARY | THRESH_OTSU);
				Mat otsuK = getStructuringElement(MORPH_ELLIPSE, Size(25, 25));
				morphologyEx(otsuBin, otsuBin, MORPH_CLOSE, otsuK);

				contours.clear();
				findContours(otsuBin.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
				resultMask = Mat::zeros(bgr.size(), CV_8UC1);
				if (!contours.empty())
				{
					int mi = 0;
					double ma = 0;
					for (int i = 0; i < static_cast<int>(contours.size()); i++)
					{
						double a = contourArea(contours[i]);
						if (a > ma)
						{
							ma = a; mi = i;
						}
					}
					drawContours(resultMask, contours, mi, Scalar(255), FILLED);
				}
				resultMask = fillHoles(resultMask);
			}
		}
	}

	// 5.最终平滑和清理。
	Mat smoothK = getStructuringElement(MORPH_ELLIPSE, Size(11, 11));
	morphologyEx(resultMask, resultMask, MORPH_CLOSE, smoothK);
	morphologyEx(resultMask, resultMask, MORPH_OPEN, smoothK);

	contours.clear();
	findContours(resultMask.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	Mat finalMask = Mat::zeros(bgr.size(), CV_8UC1);
	if (!contours.empty())
	{
		int maxIdx = 0;
		double maxArea = 0;
		for (int i = 0; i < static_cast<int>(contours.size()); i++)
		{
			double a = contourArea(contours[i]);
			if (a > maxArea)
			{
				maxArea = a;
				maxIdx = i;
			}
		}
		drawContours(finalMask, contours, maxIdx, Scalar(255), FILLED);
	}
	finalMask = fillHoles(finalMask);

	int finalPx = countNonZero(finalMask);
	double finalCoverage = static_cast<double>(finalPx) / imgArea;
	cout << "  牙齿掩膜最终结果: " << finalPx << " px ("
		<< (finalCoverage * 100) << "%)" << endl;

	// 5a.小掩膜扩展。触发条件有两个：
	// 条件1.覆盖率低于20%，可能只分离到了牙腔部分。
	// 条件2.掩膜内部亮度 > 掩膜外部亮度，此时即使覆盖率到了20%~30%也要扩展，因为这种情况说明分割逻辑反了，把亮的牙腔当成了整颗牙齿。
	bool needExpand = false;
	double expandTarget = 0.30;

	if (finalCoverage < 0.20)
	{
		needExpand = true;
		cout << "  触发扩展: 覆盖率偏低(" << (finalCoverage * 100) << "%)" << endl;
	}
	else if (finalCoverage < 0.35)
	{
		// 检查是否是亮内暗外的情况。
		double maskInMean = mean(gray, finalMask)[0];
		Mat invFinal;
		bitwise_not(finalMask, invFinal);
		double maskOutMean = mean(gray, invFinal)[0];

		if (maskInMean > maskOutMean + 5)
		{
			needExpand = true;
			expandTarget = 0.35;
			cout << "  触发扩展: 亮内(" << maskInMean
				<< ")暗外(" << maskOutMean
				<< "), 覆盖率" << (finalCoverage * 100) << "%" << endl;
		}
	}

	if (needExpand)
	{
		finalMask = expandSmallMask(finalMask, gray, expandTarget, debugDir);
		finalPx = countNonZero(finalMask);
		finalCoverage = static_cast<double>(finalPx) / imgArea;
	}

	// 5b.掩膜收紧。
	// 所有路径的掩膜都经过这一步，确保边界不会过度延伸到牙齿外。
	// 腐蚀大小根据掩膜覆盖率自适应，覆盖率大的掩膜可以多削一些，覆盖率本来就不大的掩膜少削防止过度缩小。
	int adaptiveErode = 15;
	if (finalCoverage < 0.25)
	{
		adaptiveErode = 5;
		cout << "  覆盖率偏低(" << (finalCoverage * 100)
			<< "%)，减小收紧腐蚀核" << endl;
	}
	else if (finalCoverage < 0.35)
	{
		adaptiveErode = 9;
	}
	finalMask = tightenMask(finalMask, adaptiveErode, gray, debugDir);

	// 如果收紧后掩膜太小，说明整条流程对当前图像都太严格了，回退到步骤5结束时的版本。
	double postTightenCov = static_cast<double>(countNonZero(finalMask)) / imgArea;
	if (postTightenCov < 0.08)
	{
		cout << "  警告: 收紧后覆盖率仅" << (postTightenCov * 100)
			<< "%，回退到收紧前版本" << endl;

		// 重新从步骤5的结果出发。
		contours.clear();
		findContours(resultMask.clone(), contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		finalMask = Mat::zeros(bgr.size(), CV_8UC1);
		if (!contours.empty())
		{
			int mi = 0;
			double ma = 0;
			for (int i = 0; i < static_cast<int>(contours.size()); i++)
			{
				double a = contourArea(contours[i]);
				if (a > ma)
				{
					ma = a; mi = i;
				}
			}
			drawContours(finalMask, contours, mi, Scalar(255), FILLED);
		}
		finalMask = fillHoles(finalMask);

		Mat lightK = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
		erode(finalMask, finalMask, lightK);

		cout << "  回退后覆盖率: "
			<< (100.0 * countNonZero(finalMask) / imgArea) << "%" << endl;
	}

	if (!debugDir.empty())
	{
		imwrite(debugDir + "/mask_05_final.png", finalMask);

		Mat vis = bgr.clone();
		Mat overlay = Mat::zeros(vis.size(), CV_8UC3);
		overlay.setTo(Scalar(0, 255, 0), finalMask);
		addWeighted(vis, 0.6, overlay, 0.4, 0, vis);
		vector<vector<Point>> fContours;
		findContours(finalMask.clone(), fContours,
					 RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		drawContours(vis, fContours, -1, Scalar(0, 255, 255), 2);
		imwrite(debugDir + "/mask_06_overlay.png", vis);

		Mat right = bgr.clone();
		right.setTo(Scalar(0, 0, 0), ~finalMask);
		Mat compare;
		hconcat(bgr, right, compare);
		imwrite(debugDir + "/mask_07_compare.png", compare);
	}

	return finalMask;
}

// 最后，先将图像统一缩放到固定大小，然后提取牙齿掩膜和暗区域掩膜，
// 根据牙齿掩膜的外接矩形加上一圈padding确定ROI区域，
// 最后把所有结果打包到ROIResult中返回。
ROIResult Preprocessor::extractROI(const Mat& inputBGR, const string& debugDir)
{
	ROIResult result;

	Mat resized;
	resize(inputBGR, resized, Size(1080, 1272));
	result.resizedImage = resized.clone();

	Mat gray;
	cvtColor(resized, gray, COLOR_BGR2GRAY);

	result.toothMask = extractToothMask(resized, debugDir);
	result.convergeMask = extractDarkRegions(gray);

	// 根据牙齿掩膜的外接矩形计算ROI，四周加padding防止裁剪太紧。
	Rect toothBBox(0, 0, resized.cols, resized.rows);
	{
		vector<vector<Point>> contours;
		findContours(result.toothMask.clone(), contours,
					 RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		if (!contours.empty())
		{
			vector<Point> allPts;
			for (auto& c : contours)
			{
				allPts.insert(allPts.end(), c.begin(), c.end());
			}
			toothBBox = boundingRect(allPts);
		}
	}

	int padding = 30;
	Rect roiRect(
		max(0, toothBBox.x - padding),
		max(0, toothBBox.y - padding),
		min(resized.cols - max(0, toothBBox.x - padding),
			toothBBox.width + padding * 2),
		min(resized.rows - max(0, toothBBox.y - padding),
			toothBBox.height + padding * 2));
	roiRect &= Rect(0, 0, resized.cols, resized.rows);

	result.roiRect = roiRect;
	result.roiImage = resized(roiRect).clone();
	result.roiPulpMask = result.convergeMask(roiRect).clone();
	result.roiToothMask = result.toothMask(roiRect).clone();

	cout << "  牙齿外接矩形: " << toothBBox << endl;
	cout << "  ROI区域:       " << roiRect << endl;

	return result;
}
