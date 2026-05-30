#include "detector.h"
#include <iostream>
#include <cmath>
#include <algorithm>

using namespace cv;
using namespace cv::ml;
using namespace std;

// 加载已经训练好的SVM模型，然后根据模型期望的特征维度反向验证HOG参数是否正确。
bool Detector::loadModel(const string& modelPath)
{
	svm_ = SVM::load(modelPath);
	if (svm_->empty())
	{
		cerr << "无法加载SVM模型: " << modelPath << endl;
		return false;
	}

	int varCount = svm_->getVarCount();
	cout << "SVM特征维度 = " << varCount << endl;

	winSize_ = 160;
	blockSize_ = 8;
	blockStride_ = 4;
	cellSize_ = 4;
	nbins_ = 9;

	HOGDescriptor hog(
		Size(winSize_, winSize_),
		Size(blockSize_, blockSize_),
		Size(blockStride_, blockStride_),
		Size(cellSize_, cellSize_),
		nbins_);

	size_t hogDim = hog.getDescriptorSize();
	cout << "HOG特征维度 = " << hogDim << endl;

	// 校验维度是否匹配，如果不匹配就没法用，直接报错退出。
	if (static_cast<int>(hogDim) != varCount)
	{
		cerr << "错误: HOG维度 (" << hogDim
			<< ") 与 SVM特征维度 (" << varCount << ") 不匹配" << endl;
		return false;
	}

	cout << "HOG配置: 窗口=" << winSize_
		<< " 块=" << blockSize_
		<< " 步长=" << blockStride_
		<< " 细胞=" << cellSize_
		<< " 方向bin数=" << nbins_
		<< " 维度=" << hogDim << " 通过" << endl;

	return true;
}

// 把任意大小的图像先resize到统一的窗口尺寸，
// 然后用HOGDescriptor提取特征向量。这和Python里用hog.compute()是一样的道理，只是C++需要手动管理HOGDescriptor对象。
void Detector::computeHOGFeatures(const Mat& img,
								  vector<float>& descriptors)
{
	HOGDescriptor hog(
		Size(winSize_, winSize_),
		Size(blockSize_, blockSize_),
		Size(blockStride_, blockStride_),
		Size(cellSize_, cellSize_),
		nbins_);

	Mat resized;
	if (img.cols != winSize_ || img.rows != winSize_)
	{
		resize(img, resized, Size(winSize_, winSize_));
	}
	else
	{
		resized = img;
	}

	hog.compute(resized, descriptors);
}

// 在指定位置裁剪一个patch，提取HOG特征后送入SVM得到原始决策分数。
// 这里用RAW_OUTPUT是因为需要连续的决策值来做排序和评分，分数越小越倾向于正类。
float Detector::svmScoreAt(const Mat& gray, Point2f center, int patchSize)
{
	int h = gray.rows;
	int w = gray.cols;
	int px = static_cast<int>(center.x - patchSize / 2);
	int py = static_cast<int>(center.y - patchSize / 2);

	if (px < 0 || py < 0 || px + patchSize > w || py + patchSize > h)
	{
		return 999.0f;
	}

	Rect patchRect(px, py, patchSize, patchSize);
	Mat patch = gray(patchRect);

	vector<float> desc;
	computeHOGFeatures(patch, desc);
	if (static_cast<int>(desc.size()) != svm_->getVarCount())
	{
		return 999.0f;
	}

	Mat feat(1, static_cast<int>(desc.size()), CV_32F, desc.data());
	Mat rawOut;
	svm_->predict(feat, rawOut, StatModel::RAW_OUTPUT);
	return rawOut.at<float>(0, 0);
}

// DoG暗斑检测器。
// DoG近似LoG，和形态学bottom-hat相比，DoG对边界不锐利的根管口响应更强，
// 因为它直接度量的是该尺度下的局部暗度而非形态学开运算的残差。
// 最后返回二值掩膜，在检测到的暗斑位置画小圆，后续和morphological结果合并。
static Mat generateDoGMask(const Mat& gray, const Mat& toothMask,
						   float contrastFactor)
{
	int h = gray.rows;
	int w = gray.cols;
	Mat result = Mat::zeros(h, w, CV_8UC1);

	for (double sigma = 2.0; sigma <= 22.0; sigma *= 1.3)
	{
		double sigma2 = sigma * 1.6;

		int k1 = static_cast<int>(ceil(sigma * 6));
		if (k1 % 2 == 0) k1++;
		int k2 = static_cast<int>(ceil(sigma2 * 6));
		if (k2 % 2 == 0) k2++;

		Mat g1, g2;
		GaussianBlur(gray, g1, Size(k1, k1), sigma);
		GaussianBlur(gray, g2, Size(k2, k2), sigma2);

		// 暗斑中心处 g2 > g1。
		Mat g1f, g2f, dog;
		g1.convertTo(g1f, CV_32F);
		g2.convertTo(g2f, CV_32F);
		subtract(g2f, g1f, dog);

		// 尺度归一化。
		dog *= static_cast<float>(sigma * sigma);

		// 非极大值抑制找局部峰。
		int nmsR = max(3, static_cast<int>(sigma));
		Mat kernel = getStructuringElement(MORPH_ELLIPSE,
										   Size(2 * nmsR + 1, 2 * nmsR + 1));
		Mat dilated;
		dilate(dog, dilated, kernel);

		float thresh = max(0.5f, 1.5f * contrastFactor);
		int blobR = max(3, static_cast<int>(sigma * 1.5));
		int stride = max(1, nmsR / 2);

		for (int y = nmsR; y < h - nmsR; y += stride)
		{
			const float* pD = dog.ptr<float>(y);
			const float* pM = dilated.ptr<float>(y);
			for (int x = nmsR; x < w - nmsR; x += stride)
			{
				if (pD[x] > thresh && pD[x] == pM[x])
				{
					if (!toothMask.empty() &&
						toothMask.at<uchar>(y, x) == 0)
						continue;
					circle(result, Point(x, y), blobR, Scalar(255), FILLED);
				}
			}
		}
	}

	int nPx = countNonZero(result);
	if (nPx > 0)
	{
		cout << "  DoG检测: " << nPx << " px" << endl;
	}

	return result;
}

// 径向对比度暗点检测器。
// 直接度量每个像素与其邻域的亮度差。
// 返回二值掩膜。
static Mat generateRadialContrastMask(const Mat& gray, const Mat& toothMask,
									  float toothMean, float toothStd)
{
	int h = gray.rows;
	int w = gray.cols;
	Mat result = Mat::zeros(h, w, CV_8UC1);

	// 轻度平滑去噪。
	Mat smoothed;
	GaussianBlur(gray, smoothed, Size(5, 5), 1.5);

	// 牙齿越暗黄，门槛越低。
	float contrastThresh = max(2.0f, toothStd * 0.12f);

	// 内圆取均值 = 根管口亮度，外环取均值 = 周围牙齿亮度。
	int innerRadii[] = { 5, 10, 18, 30 };
	int outerRadii[] = { 15, 25, 40, 60 };
	int nPairs = 4;

	for (int pi = 0; pi < nPairs; pi++)
	{
		int ri = innerRadii[pi];
		int ro = outerRadii[pi];

		// 用boxFilter近似圆域。
		Mat meanInner, meanOuter;
		int ki = 2 * ri + 1;
		int ko = 2 * ro + 1;
		blur(smoothed, meanInner, Size(ki, ki));
		blur(smoothed, meanOuter, Size(ko, ko));

		// 对比度 = 外环均值 - 内圆均值。
		Mat contrast;
		Mat innerF, outerF;
		meanInner.convertTo(innerF, CV_32F);
		meanOuter.convertTo(outerF, CV_32F);
		subtract(outerF, innerF, contrast);

		// 非极大值抑制。
		int nmsR = ri;
		Mat kernel = getStructuringElement(MORPH_ELLIPSE,
										   Size(2 * nmsR + 1, 2 * nmsR + 1));
		Mat dilated;
		dilate(contrast, dilated, kernel);

		int stride = max(2, ri / 2);
		int blobR = max(3, ri);

		for (int y = ro; y < h - ro; y += stride)
		{
			const float* pC = contrast.ptr<float>(y);
			const float* pM = dilated.ptr<float>(y);
			for (int x = ro; x < w - ro; x += stride)
			{
				if (pC[x] > contrastThresh && pC[x] == pM[x])
				{
					if (!toothMask.empty() &&
						toothMask.at<uchar>(y, x) == 0)
						continue;
					circle(result, Point(x, y), blobR, Scalar(255), FILLED);
				}
			}
		}
	}

	int nPx = countNonZero(result);
	if (nPx > 0)
	{
		cout << "  径向对比度检测: " << nPx << " px" << endl;
	}

	return result;
}

// detect大致分为四个阶段：
// 1.用形态学操作在多个尺度下找局部暗区域，对大blob用暗点定位法拆分，
// 2.对这些候选blob做几何和亮度上的过滤，
// 3.用物理特征综合评分，
// 4.在前面完全没有候选时做暴力搜索最暗点。
// 这样可以在尽量不漏检的前提下逐步提高精度。
vector<Detection> Detector::detect(
	const Mat& roiGray, Point roiOffset,
	const Mat& toothMask, const Mat& pulpMask,
	float confThresh)
{
	// pulpMask用了效果不佳，不用了。
	(void)pulpMask;

	vector<Detection> detections;
	int h = roiGray.rows;
	int w = roiGray.cols;

	// 视频模式下，缓存的mask和当前帧ROI尺寸可能不一致。
	// 不能resize，尺寸不匹配时直接放弃mask，
	// 先让检测无约束进行，后处理阶段的filterByMask会再过滤。
	Mat toothMaskResized = toothMask;
	if (!toothMaskResized.empty() &&
		(toothMaskResized.rows != h || toothMaskResized.cols != w))
	{
		cout << "  掩膜尺寸不匹配: mask=" << toothMaskResized.size()
			<< " roi=" << roiGray.size() << ", 跳过掩膜约束" << endl;
		toothMaskResized = Mat();
	}

	Mat erodedTooth;
	if (!toothMaskResized.empty())
	{
		Mat ek = getStructuringElement(MORPH_ELLIPSE, Size(21, 21));
		erode(toothMaskResized, erodedTooth, ek);
		cout << "  腐蚀后牙齿掩膜像素: " << countNonZero(erodedTooth) << " px" << endl;
	}

	// 统计牙齿区域的亮度均值和标准差，后面用来判断候选点是不是足够暗。
	// 如果没有有效的掩膜就用默认值，保证流程不会中断。
	double toothMean = 128;
	double toothStd = 30;
	if (!erodedTooth.empty() && countNonZero(erodedTooth) > 0)
	{
		Scalar mu, sd;
		meanStdDev(roiGray, mu, sd, erodedTooth);
		toothMean = mu[0];
		toothStd = sd[0];
	}
	cout << "  牙齿亮度: 均值=" << toothMean
		<< " 标准差=" << toothStd << endl;

	// 计算牙齿掩膜的质心，用于中心距离加权。
	Point2f toothCentroid(w / 2.0f, h / 2.0f);
	if (!toothMaskResized.empty() && countNonZero(toothMaskResized) > 0)
	{
		Moments toothMom = moments(toothMaskResized, true);
		if (toothMom.m00 > 0)
		{
			toothCentroid.x = static_cast<float>(toothMom.m10 / toothMom.m00);
			toothCentroid.y = static_cast<float>(toothMom.m01 / toothMom.m00);
		}
	}
	float maxDist = sqrt(static_cast<float>(w * w + h * h)) / 2.0f;
	cout << "  牙齿质心: (" << toothCentroid.x << ", " << toothCentroid.y
		<< ") 最大距离=" << maxDist << endl;

	// 1.对每个尺度用膨胀操作取局部最大值，然后和原图做差，
	// 差值大的地方说明该点比周围暗很多，就是局部极小值。
	// 用多尺度捕捉不同大小的根管口。
	Mat allMinima = Mat::zeros(roiGray.size(), CV_8UC1);

	// 阈值作为尺度的连续函数自适应计算。
	float contrastFactor = max(0.3f, min(1.5f, static_cast<float>(toothStd) / 25.0f));
	vector<int> scales;
	vector<int> hThresholds;
	for (double s = 7.0; s <= 87.0; s *= 1.18)
	{
		int si = static_cast<int>(round(s));
		if (si % 2 == 0) si++;
		if (!scales.empty() && si <= scales.back()) continue;
		scales.push_back(si);

		double baseH = 1.0 + 0.135 * si;
		int hVal = max(1, static_cast<int>(baseH * contrastFactor));
		hThresholds.push_back(hVal);
	}
	int nScales = static_cast<int>(scales.size());

	cout << "  自适应阈值系数=" << contrastFactor
		<< " 尺度数=" << nScales << " 尺度(阈值)=[";
	for (int i = 0; i < nScales; i++)
	{
		if (i > 0) cout << ",";
		cout << scales[i] << "(" << hThresholds[i] << ")";
	}
	cout << "]" << endl;

	for (int si = 0; si < nScales; si++)
	{
		Mat dilated;
		Mat dk = getStructuringElement(MORPH_ELLIPSE,
									   Size(scales[si], scales[si]));
		dilate(roiGray, dilated, dk);

		Mat darkness;
		subtract(dilated, roiGray, darkness);

		Mat minMask;
		threshold(darkness, minMask, hThresholds[si], 255, THRESH_BINARY);

		bitwise_or(allMinima, minMask, allMinima);
	}

	// 只保留牙齿区域内的局部极小值，排除背景干扰。
	if (!toothMaskResized.empty())
	{
		bitwise_and(allMinima, toothMaskResized, allMinima);
	}

	// 闭运算填补小间隙，开运算去除噪点。
	Mat mck = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
	morphologyEx(allMinima, allMinima, MORPH_CLOSE, mck);
	Mat mok = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	morphologyEx(allMinima, allMinima, MORPH_OPEN, mok);

	// 有些根管口虽然不是严格的局部极小值，但绝对亮度很低，用一个基于牙齿均值的阈值把这些也捞出来。
	Mat absDark;
	double absThresh = max(30.0, toothMean - 1.5 * toothStd);
	threshold(roiGray, absDark, absThresh, 255, THRESH_BINARY_INV);
	if (!toothMaskResized.empty())
	{
		bitwise_and(absDark, toothMaskResized, absDark);
	}

	// 做连通域分析，过滤掉面积太小的噪点和太大的整片阴影。
	{
		vector<vector<Point>> absBlobs;
		findContours(absDark.clone(), absBlobs, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		Mat filteredAbsDark = Mat::zeros(roiGray.size(), CV_8UC1);
		for (const auto& b : absBlobs)
		{
			double area = contourArea(b);
			if (area > 10 && area < 5000)
			{
				vector<vector<Point>> bv = { b };
				drawContours(filteredAbsDark, bv, 0, Scalar(255), FILLED);
			}
		}
		bitwise_or(allMinima, filteredAbsDark, allMinima);
	}

	int morphPx = countNonZero(allMinima);

	// 找出所有候选blob的轮廓，每个轮廓就是一个可能的根管口区域。
	// 多尺度形态学检测容易把多个根管口的暗区域合并成一整片巨大blob，
	// 因此对面积过大的blob，用局部最暗点定位法在其内部逐个找暗斑中心进行拆分。
	vector<vector<Point>> rawBlobs;
	findContours(allMinima.clone(), rawBlobs, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	vector<vector<Point>> blobs;
	int splitThresh = 8000;

	for (const auto& blob : rawBlobs)
	{
		double area = contourArea(blob);

		if (area < 10)
		{
			continue;
		}

		if (area <= splitThresh)
		{
			blobs.push_back(blob);
			continue;
		}

		// 大blob需要拆分，在blob内部逐个找最暗点作为候选中心。
		cout << "  拆分大区域: 面积=" << static_cast<int>(area) << endl;

		Mat blobMask = Mat::zeros(roiGray.size(), CV_8UC1);
		vector<vector<Point>> bv = { blob };
		drawContours(blobMask, bv, 0, Scalar(255), FILLED);

		// 叠加中心约束。
		Mat centerZone = Mat::zeros(roiGray.size(), CV_8UC1);
		int zoneR = min(w, h) * 3 / 10;
		circle(centerZone, Point(static_cast<int>(toothCentroid.x),
								 static_cast<int>(toothCentroid.y)),
			   zoneR, Scalar(255), FILLED);
		Mat searchArea;
		bitwise_and(blobMask, centerZone, searchArea);

		// 进一步限制在腐蚀后的牙齿掩膜内，排除牙齿边界区域
		if (!erodedTooth.empty())
		{
			bitwise_and(searchArea, erodedTooth, searchArea);
		}

		if (countNonZero(searchArea) < 100)
		{
			searchArea = blobMask.clone();
		}

		Mat bSmoothed;
		GaussianBlur(roiGray, bSmoothed, Size(15, 15), 4);

		int nFound = 0;
		for (int si = 0; si < 10 && nFound < 8; si++)
		{
			double minVal;
			Point minLoc;
			minMaxLoc(bSmoothed, &minVal, nullptr, &minLoc, nullptr, searchArea);

			if (minVal > toothMean + 5)
			{
				break;
			}

			int candR = 15;
			Mat smallBlob = Mat::zeros(roiGray.size(), CV_8UC1);
			circle(smallBlob, minLoc, candR, Scalar(255), FILLED);
			bitwise_and(smallBlob, blobMask, smallBlob);

			vector<vector<Point>> sc;
			findContours(smallBlob.clone(), sc, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
			for (const auto& s : sc)
			{
				if (contourArea(s) >= 5)
				{
					blobs.push_back(s);
					nFound++;
				}
			}

			circle(searchArea, minLoc, 40, Scalar(0), FILLED);
		}

		cout << "    拆分为 " << nFound << " 个候选点" << endl;
	}

	cout << "  阶段1: " << blobs.size() << " 个局部极小值区域"
		<< " (拆分前" << rawBlobs.size() << "个)" << endl;

	// 2.对每个blob进行几何和亮度过滤。
	// 先不追求精确，只是把明显不合理的候选去掉。
	// 通过的候选会带上一系列特征值，供后面的评分使用。
	struct Candidate
	{
		Point2f center;
		float area;
		float circularity;
		float aspectRatio;
		float innerMean;
		float surroundMean;
		float darkContrast;
		float localDepth;
		Rect bbox;
	};

	vector<Candidate> candidates;

	int rejectArea = 0;
	int rejectBounds = 0;
	int rejectMask = 0;
	int rejectAR = 0;
	int rejectBright = 0;
	int rejectContrast = 0;

	for (const auto& blob : blobs)
	{
		double area = contourArea(blob);
		if (area < 10 || area > 12000)
		{
			rejectArea++;
			continue;
		}

		Moments mom = moments(blob);
		if (mom.m00 < 1)
		{
			continue;
		}
		Point2f ctr(static_cast<float>(mom.m10 / mom.m00),
					static_cast<float>(mom.m01 / mom.m00));

		if (ctr.x < 3 || ctr.x > w - 3 || ctr.y < 3 || ctr.y > h - 3)
		{
			rejectBounds++;
			continue;
		}

		// 检查中心是否落在牙齿掩膜内，不在牙齿上的候选直接排除。
		// 但如果候选非常靠近图像中心且自身足够暗，给一次机会，因为预处理偶尔会把牙齿中心的暗区域误判为背景而从掩膜中去掉。
		if (!toothMaskResized.empty())
		{
			int iy = static_cast<int>(ctr.y);
			int ix = static_cast<int>(ctr.x);
			if (iy < 0 || iy >= toothMaskResized.rows ||
				ix < 0 || ix >= toothMaskResized.cols)
			{
				rejectMask++;
				continue;
			}
			if (toothMaskResized.at<uchar>(iy, ix) == 0)
			{
				// 条件是距离图像中心非常近且blob本身足够暗。
				float dxC = ctr.x - w / 2.0f;
				float dyC = ctr.y - h / 2.0f;
				float distRatio = sqrt(dxC * dxC + dyC * dyC) /
					(sqrt(static_cast<float>(w * w + h * h)) / 2.0f);
				if (distRatio > 0.10f)
				{
					rejectMask++;
					continue;
				}
			}
		}

		// 衡量形状接近圆形的程度。
		double perimeter = arcLength(blob, true);
		float circularity = (perimeter > 0) ?
			static_cast<float>(4.0 * CV_PI * area / (perimeter * perimeter)) : 0.0f;

		// 长宽比用来排除细长的沟壑，有好几张图都长这样。
		float aspectRatio = 1.0f;
		if (static_cast<int>(blob.size()) >= 5)
		{
			RotatedRect ell = fitEllipse(blob);
			float major = max(ell.size.width, ell.size.height);
			float minor = min(ell.size.width, ell.size.height);
			aspectRatio = (minor > 1) ? (major / minor) : 99.0f;
		}
		else
		{
			Rect bb = boundingRect(blob);
			float major = static_cast<float>(max(bb.width, bb.height));
			float minor = static_cast<float>(max(1, min(bb.width, bb.height)));
			aspectRatio = major / minor;
		}

		if (aspectRatio > 8.0f)
		{
			rejectAR++;
			continue;
		}

		// 计算blob内部的亮度统计，太亮的不可能是根管口。
		Mat blobMask = Mat::zeros(roiGray.size(), CV_8UC1);
		vector<vector<Point>> blobVec = { blob };
		drawContours(blobMask, blobVec, 0, Scalar(255), FILLED);
		double innerMean = mean(roiGray, blobMask)[0];

		double blobMin = 255;
		{
			double tmpMax;
			Point tmpMinLoc, tmpMaxLoc;
			minMaxLoc(roiGray, &blobMin, &tmpMax, &tmpMinLoc, &tmpMaxLoc, blobMask);
		}

		// 亮度过滤要有三个条件：
		// 1.绝对亮度很高
		// 2.最暗像素也不够暗
		// 3.和周围对比度很低
		// 三个条件同时满足才排除。
		{
			int ringCheck = max(20, static_cast<int>(sqrt(area) * 2.0));
			Mat dkCheck = getStructuringElement(MORPH_ELLIPSE,
												Size(ringCheck, ringCheck));
			Mat dilCheck;
			dilate(blobMask, dilCheck, dkCheck);
			Mat surCheck;
			subtract(dilCheck, blobMask, surCheck);
			if (!toothMaskResized.empty())
			{
				bitwise_and(surCheck, toothMaskResized, surCheck);
			}
			double quickSurround = innerMean;
			if (countNonZero(surCheck) > 10)
			{
				quickSurround = mean(roiGray, surCheck)[0];
			}
			double quickContrast = quickSurround - innerMean;

			if (innerMean > 200 && blobMin > 100 && quickContrast < 5)
			{
				rejectBright++;
				continue;
			}
		}

		// 计算blob周围环形区域的平均亮度，和内部做差得到暗度对比。
		int ringOuter = max(20, static_cast<int>(sqrt(area) * 2.5));
		Mat dilatedBlob;
		Mat dk = getStructuringElement(MORPH_ELLIPSE,
									   Size(ringOuter, ringOuter));
		dilate(blobMask, dilatedBlob, dk);
		Mat surroundMask;
		subtract(dilatedBlob, blobMask, surroundMask);
		if (!toothMaskResized.empty())
		{
			bitwise_and(surroundMask, toothMaskResized, surroundMask);
		}

		double surroundMean = innerMean;
		if (countNonZero(surroundMask) > 10)
		{
			surroundMean = mean(roiGray, surroundMask)[0];
		}

		float darkContrast = static_cast<float>(surroundMean - innerMean);

		if (darkContrast < 0)
		{
			rejectContrast++;
			continue;
		}

		// 局部深度是在候选点附近小区域内最亮值和候选内部均值的差，
		// 反映了这个暗斑在局部范围内有多突出。
		int lx = max(0, static_cast<int>(ctr.x) - 40);
		int ly = max(0, static_cast<int>(ctr.y) - 40);
		int lw = min(80, w - lx);
		int lh = min(80, h - ly);
		Rect localRect(lx, ly, lw, lh);
		double localMax = 0;
		minMaxLoc(roiGray(localRect), nullptr, &localMax);
		float localDepth = static_cast<float>(localMax - innerMean);

		Candidate c;
		c.center = ctr;
		c.area = static_cast<float>(area);
		c.circularity = circularity;
		c.aspectRatio = aspectRatio;
		c.innerMean = static_cast<float>(innerMean);
		c.surroundMean = static_cast<float>(surroundMean);
		c.darkContrast = darkContrast;
		c.localDepth = localDepth;
		c.bbox = boundingRect(blob);
		candidates.push_back(c);
	}

	cout << "  阶段2 拒绝统计: 面积=" << rejectArea
		<< " 边界=" << rejectBounds
		<< " 掩膜=" << rejectMask
		<< " 长宽比=" << rejectAR
		<< " 过亮=" << rejectBright
		<< " 对比度=" << rejectContrast << endl;
	cout << "  阶段2: " << candidates.size()
		<< " 个候选通过" << endl;

	for (int ci = 0; ci < static_cast<int>(candidates.size()); ci++)
	{
		Candidate& c = candidates[ci];
		cout << "    候选[" << ci << "] 位置=(" << static_cast<int>(c.center.x)
			<< "," << static_cast<int>(c.center.y) << ")"
			<< " 面积=" << c.area
			<< " 圆度=" << c.circularity
			<< " 长宽比=" << c.aspectRatio
			<< " 内部均值=" << c.innerMean
			<< " 周围均值=" << c.surroundMean
			<< " 对比度=" << c.darkContrast
			<< " 局部深度=" << c.localDepth << endl;
	}

	// 3. 综合物理特征评分。
	//
	// 原设计其实是对每个候选在多个patch尺寸下提取HOG特征送入SVM，
	// 取最小决策值作为SVM置信度，低于阈值的候选通过验证。
	// 但实际上SVM在当前数据上所有分数均为正，导致每张图其实都是暴力搜索得到的结果。
	// 我的水平实在有限，剩下的时间也不多了，确实不太会修正，也不太敢大改，于是把以下的代码注释掉，至少说明我曾经做过这样的尝试，虽然效果不尽如人意。
	//
	// int patchSizes[] = { 120, 160, 200, 240 };
	// int nPatchSizes = 4;
	// vector<float> allSvmScores;
	// vector<float> bestScorePerCand(candidates.size(), 999.0f);
	//
	// for (int ci = 0; ci < static_cast<int>(candidates.size()); ci++)
	// {
	//     Candidate& cand = candidates[ci];
	//     float bestDec = 999.0f;
	//     for (int pi = 0; pi < nPatchSizes; pi++)
	//     {
	//         float d = svmScoreAt(roiGray, cand.center, patchSizes[pi]);
	//         if (d != 999.0f)
	//         {
	//             allSvmScores.push_back(d);
	//             if (d < bestDec) bestDec = d;
	//         }
	//     }
	//     bestScorePerCand[ci] = bestDec;
	// }
	//
	// // 用SVM决策值做判断。
	// float threshold = 0.5f;
	// if (cand.darkContrast > 15 && cand.circularity > 0.15f) threshold = 0.8f;
	// else if (cand.darkContrast > 8) threshold = 0.6f;
	// if (bestDec < threshold) { svmOk = true; svmConf = threshold - bestDec; }
	//

	int svmPass = 0;

	for (int ci = 0; ci < static_cast<int>(candidates.size()); ci++)
	{
		Candidate& cand = candidates[ci];

		// 计算SVM分数用于日志输出，便于调试和后续模型改进参考
		float bestDec = 999.0f;
		{
			int patchSizes[] = { 120, 160, 200, 240 };
			for (int pi = 0; pi < 4; pi++)
			{
				float d = svmScoreAt(roiGray, cand.center, patchSizes[pi]);
				if (d != 999.0f && d < bestDec)
				{
					bestDec = d;
				}
			}
		}

		cout << "    候选[" << ci << "] SVM参考分数=" << bestDec << endl;

		// 综合物理特征评分
		float score = 1.0f;

		// 暗度对比
		if (cand.darkContrast > 50)
		{
			score *= 3.0f;
		}
		else if (cand.darkContrast > 30)
		{
			score *= 2.5f;
		}
		else if (cand.darkContrast > 15)
		{
			score *= 2.0f;
		}
		else if (cand.darkContrast > 8)
		{
			score *= 1.5f;
		}
		else if (cand.darkContrast > 3)
		{
			score *= 1.2f;
		}
		else
		{
			score *= 0.5f;
		}

		// 局部深度
		if (cand.localDepth > 80)
		{
			score *= 1.5f;
		}
		else if (cand.localDepth > 40)
		{
			score *= 1.3f;
		}
		else if (cand.localDepth > 20)
		{
			score *= 1.1f;
		}
		else
		{
			score *= 0.7f;
		}

		// 圆度
		if (cand.circularity > 0.6f)
		{
			score *= 1.5f;
		}
		else if (cand.circularity > 0.4f)
		{
			score *= 1.3f;
		}
		else if (cand.circularity > 0.2f)
		{
			score *= 1.0f;
		}
		else
		{
			score *= 0.7f;
		}

		// 长宽比惩罚
		if (cand.aspectRatio > 5.0f)
		{
			score *= 0.3f;
		}
		else if (cand.aspectRatio > 3.5f)
		{
			score *= 0.5f;
		}
		else if (cand.aspectRatio > 2.5f)
		{
			score *= 0.8f;
		}

		// 亮度惩罚
		float candMinBright = cand.innerMean;
		{
			int bx = max(0, cand.bbox.x);
			int by = max(0, cand.bbox.y);
			int bw2 = min(cand.bbox.width, w - bx);
			int bh2 = min(cand.bbox.height, h - by);
			if (bw2 > 0 && bh2 > 0)
			{
				double pMin;
				minMaxLoc(roiGray(Rect(bx, by, bw2, bh2)), &pMin);
				candMinBright = static_cast<float>(pMin);
			}
		}
		bool hasDeepDark = (candMinBright < toothMean - 20);
		bool hasGoodContrast = (cand.darkContrast > 8);

		if (cand.innerMean > toothMean + 20)
		{
			if (hasDeepDark || hasGoodContrast)
				score *= 0.6f;
			else
				score *= 0.15f;
		}
		else if (cand.innerMean > toothMean + 5)
		{
			if (hasDeepDark || hasGoodContrast)
				score *= 0.8f;
			else
				score *= 0.4f;
		}
		else if (cand.innerMean > toothMean)
		{
			score *= hasDeepDark ? 0.95f : 0.7f;
		}

		// 腐蚀掩膜内的候选加分
		if (!erodedTooth.empty())
		{
			int iy = static_cast<int>(cand.center.y);
			int ix = static_cast<int>(cand.center.x);
			if (iy >= 0 && iy < erodedTooth.rows &&
				ix >= 0 && ix < erodedTooth.cols &&
				erodedTooth.at<uchar>(iy, ix) > 0)
			{
				score *= 1.3f;
			}
			else
			{
				// 不在腐蚀掩膜内 = 靠近牙齿边缘或牙齿外，大幅惩罚
				score *= 0.3f;
			}
		}

		// 中心距离加权
		float dx = cand.center.x - toothCentroid.x;
		float dy = cand.center.y - toothCentroid.y;
		float distToCenter = sqrt(dx * dx + dy * dy);
		float distRatio = distToCenter / maxDist;

		if (distRatio < 0.08f)
		{
			score *= 5.0f;
		}
		else if (distRatio < 0.12f)
		{
			score *= 4.0f;
		}
		else if (distRatio < 0.18f)
		{
			score *= 3.0f;
		}
		else if (distRatio < 0.25f)
		{
			score *= 2.0f;
		}
		else if (distRatio < 0.35f)
		{
			score *= 1.2f;
		}
		else if (distRatio < 0.45f)
		{
			score *= 0.5f;
		}
		else if (distRatio < 0.55f)
		{
			score *= 0.15f;
		}
		else
		{
			score *= 0.05f;
		}

		// 面积惩罚
		if (cand.area < 30)
		{
			score *= 0.5f;
		}
		else if (cand.area > 5000)
		{
			score *= 0.5f;
		}

		if (bestDec != 999.0f)
		{
			if (bestDec < 0.3f)
			{
				score *= 2.0f;
			}
			else if (bestDec < 0.5f)
			{
				score *= 1.6f;
			}
			else if (bestDec < 0.8f)
			{
				score *= 1.2f;
			}
			else if (bestDec > 1.2f)
			{
				score *= 0.5f;
			}
			cout << "    SVM评分调整: dec=" << bestDec << endl;
		}


		if (score < confThresh)
		{
			continue;
		}
		svmPass++;

		int bw = max(cand.bbox.width, 10);
		int bh = max(cand.bbox.height, 10);
		bw = max(20, min(bw + 10, 100));
		bh = max(20, min(bh + 10, 100));

		Detection det;
		det.box = Rect(
			static_cast<int>(cand.center.x + roiOffset.x - bw / 2),
			static_cast<int>(cand.center.y + roiOffset.y - bh / 2),
			bw, bh);
		det.confidence = score;
		detections.push_back(det);

		cout << "    已检出: (" << static_cast<int>(cand.center.x) << ","
			<< static_cast<int>(cand.center.y) << ")"
			<< " SVM(参考)=" << bestDec
			<< " 对比度=" << cand.darkContrast
			<< " 深度=" << cand.localDepth
			<< " 圆度=" << cand.circularity
			<< " 得分=" << score << endl;
	}

	// 4.在牙齿中心区域逐个搜索最暗点。
	if (detections.empty())
	{
		cout << "  完全没有候选. 暴力搜索最暗点." << endl;

		Mat blurred;
		GaussianBlur(roiGray, blurred, Size(21, 21), 5);

		Mat searchMask = erodedTooth.empty() ?
			(toothMaskResized.empty() ? Mat() : toothMaskResized.clone()) :
			erodedTooth.clone();

		// 叠加一个中心圆形约束。
		{
			Mat centerZone = Mat::zeros(roiGray.size(), CV_8UC1);
			int zoneR = min(w, h) * 3 / 10;
			circle(centerZone, Point(w / 2, h / 2), zoneR, Scalar(255), FILLED);
			if (searchMask.empty())
			{
				searchMask = centerZone;
			}
			else
			{
				bitwise_and(searchMask, centerZone, searchMask);
			}
		}

		for (int iter = 0; iter < 8; iter++)
		{
			double minVal, maxVal;
			Point minLoc, maxLoc;

			if (searchMask.empty())
			{
				minMaxLoc(blurred, &minVal, &maxVal, &minLoc, &maxLoc);
			}
			else
			{
				minMaxLoc(blurred, &minVal, &maxVal, &minLoc, &maxLoc, searchMask);
			}

			if (minVal > toothMean + 5)
			{
				break;
			}

			int r = 30;
			int lx = max(0, minLoc.x - r);
			int ly = max(0, minLoc.y - r);
			int lw = min(2 * r, w - lx);
			int lh = min(2 * r, h - ly);
			double localMean = mean(roiGray(Rect(lx, ly, lw, lh)))[0];
			float contrast = static_cast<float>(localMean - minVal);

			if (contrast < 5)
			{
				if (!searchMask.empty())
				{
					circle(searchMask, minLoc, 30, Scalar(0), -1);
				}
				continue;
			}

			float bfDx = static_cast<float>(minLoc.x) - toothCentroid.x;
			float bfDy = static_cast<float>(minLoc.y) - toothCentroid.y;
			float bfDistRatio = sqrt(bfDx * bfDx + bfDy * bfDy) / maxDist;
			float centerBonus = 1.0f;
			if (bfDistRatio < 0.15f)
			{
				centerBonus = 3.0f;
			}
			else if (bfDistRatio < 0.30f)
			{
				centerBonus = 2.0f;
			}
			else if (bfDistRatio < 0.50f)
			{
				centerBonus = 1.0f;
			}
			else
			{
				centerBonus = 0.2f;
			}

			Detection det;
			det.box = Rect(
				minLoc.x + roiOffset.x - 25,
				minLoc.y + roiOffset.y - 25,
				50, 50);
			det.confidence = (contrast / 50.0f) * centerBonus;
			detections.push_back(det);

			cout << "    暴力搜索: (" << minLoc.x << "," << minLoc.y
				<< ") 灰度值=" << minVal
				<< " 对比度=" << contrast << endl;

			if (!searchMask.empty())
			{
				circle(searchMask, minLoc, 40, Scalar(0), -1);
			}
		}
	}

	cout << "检测诊断信息" << endl;
	cout << "  局部极小值区域数:   " << blobs.size() << endl;
	cout << "  形状过滤后候选数:   " << candidates.size() << endl;
	cout << "  物理评分通过数:     " << svmPass << endl;
	cout << "  最终检测数:         " << detections.size() << endl;
	cout << "  牙齿平均亮度:       " << toothMean << endl;
	cout << "  绝对暗度阈值:       " << absThresh << endl;

	return detections;
}
