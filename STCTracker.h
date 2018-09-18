#pragma once

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class STCTracker{
public:
	STCTracker();
	~STCTracker();
	void init(const Mat frame, const Rect box,Rect &boxRegion);
	void tracking(const Mat frame, Rect &trackBox,Rect &boxRegion,int FrameNum);

private:
	void createHammingWin();
	void complexOperation(const Mat src1, const Mat src2, Mat &dst, int flag = 0);
	void getCxtPriorPosteriorModel(const Mat image);
	void learnSTCModel(const Mat image);

private:
	double sigma;			// scale parameter (方差)
	double alpha;			// scale parameter
	double beta;			// shape parameter
	double rho;				// learning parameter
	double scale;			//	scale ratio
	double lambda;		//	scale learning parameter
	int num;					//	the number of frames for updating the scale
	vector<double> maxValue;
	Point center;			//目标位置
	Rect cxtRegion;		//上下文区域
	int padding;

	Mat cxtPriorPro;		// 上文概率
	Mat cxtPosteriorPro;	// 下文概率
	Mat STModel;			// 当前概率
	Mat STCModel;			// 时空上下文模型
	Mat hammingWin;			// Hamming窗
};
