#include "STCTracker.h"

STCTracker::STCTracker(){}
STCTracker::~STCTracker(){}

/************创建一个Hamming窗********************/
void STCTracker::createHammingWin(){
	for (int i = 0; i < hammingWin.rows; i++){
		for (int j = 0; j < hammingWin.cols; j++){
			hammingWin.at<double>(i, j) = (0.54 - 0.46 * cos( 2 * CV_PI * i / hammingWin.rows))*(0.54 - 0.46 * cos( 2 * CV_PI * j / hammingWin.cols));
		}
	}
}

/************复数操作 *****************/
void STCTracker::complexOperation(const Mat src1, const Mat src2, Mat &dst, int flag){
	CV_Assert(src1.size == src2.size);
	CV_Assert(src1.channels() == 2);

	Mat A_Real, A_Imag, B_Real, B_Imag, R_Real, R_Imag;
	vector<Mat> planes;
	split(src1, planes);
	planes[0].copyTo(A_Real);
	planes[1].copyTo(A_Imag);

	split(src2, planes);
	planes[0].copyTo(B_Real);
	planes[1].copyTo(B_Imag);

	dst.create(src1.rows, src1.cols, CV_64FC2);
	split(dst, planes);
	R_Real = planes[0];
	R_Imag = planes[1];

	for (int i = 0; i < A_Real.rows; i++){
		for (int j = 0; j < A_Real.cols; j++){
			double a = A_Real.at<double>(i, j);
			double b = A_Imag.at<double>(i, j);
			double c = B_Real.at<double>(i, j);
			double d = B_Imag.at<double>(i, j);

			if (flag){
				// division: (a+bj) / (c+dj)
				R_Real.at<double>(i, j) = (a * c + b * d) / (c * c + d * d + 0.000001);
				R_Imag.at<double>(i, j) = (b * c - a * d) / (c * c + d * d + 0.000001);
			}else{
				// multiplication: (a+bj) * (c+dj)
				R_Real.at<double>(i, j) = a * c - b * d;
				R_Imag.at<double>(i, j) = b * c + a * d;
			}
		}
	}
	merge(planes, dst);
}

/************获取上下文的上文和最大概率的下文***********/
void STCTracker::getCxtPriorPosteriorModel(const Mat image){
	CV_Assert(image.size == cxtPriorPro.size);

	double sum_prior(0), sum_post(0);
	for (int i = 0; i < cxtRegion.height; i++){
		for (int j = 0; j < cxtRegion.width; j++){
			double x = j + cxtRegion.x;
			double y = i + cxtRegion.y;
			double dist = sqrt((center.x - x) * (center.x - x) + (center.y - y) * (center.y - y));

			//论文的等式（5）
			cxtPriorPro.at<double>(i, j) = exp(- dist * dist / (2 * sigma * sigma));
			sum_prior += cxtPriorPro.at<double>(i, j);

			//论文的等式（6）
			cxtPosteriorPro.at<double>(i, j) = exp(- pow(dist / sqrt(alpha), beta));
			sum_post += cxtPosteriorPro.at<double>(i, j);
		}
	}
	cxtPriorPro.convertTo(cxtPriorPro, -1, 1.0/sum_prior);
	cxtPriorPro = cxtPriorPro.mul(image);
	cxtPosteriorPro.convertTo(cxtPosteriorPro, -1, 1.0/sum_post);
}

/************学习时空上下文模型 ***********/
void STCTracker::learnSTCModel(const Mat image){
	//获取上文和下文的概率
	getCxtPriorPosteriorModel(image);

	//提取上文概率的离散傅立叶变换(DFT)
	Mat priorFourier;
	Mat planes1[] = {cxtPriorPro, Mat::zeros(cxtPriorPro.size(), CV_64F)};
    merge(planes1, 2, priorFourier);
	dft(priorFourier, priorFourier);

	//提取下文概率的离散快速傅立叶变换(DTF)
	Mat postFourier;
	Mat planes2[] = {cxtPosteriorPro, Mat::zeros(cxtPosteriorPro.size(), CV_64F)};
    merge(planes2, 2, postFourier);
	dft(postFourier, postFourier);

	//相除
	Mat conditionalFourier;
	complexOperation(postFourier, priorFourier, conditionalFourier, 1);

	//提取当前概率的离散逆傅立叶变换(IDFT)得到STC模型
	dft(conditionalFourier, STModel, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);

	//利用学习到的空间上下文模型更新时空上下文模型
	addWeighted(STCModel, 1.0 - rho, STModel, rho, 0.0, STCModel);
}

/************ 初始化超参和模型 ***********/
void STCTracker::init(const Mat frame, const Rect box,Rect &boxRegion){
	//初始化一些模型
	padding = 1;
	num = 5;         //连续帧数量
	alpha = 2.25; //公式6中alpha的值
	beta = 1;		 //公式(6)
	rho = 0.075; //公式12中的学习参数\rho
	lambda = 0.25;
	sigma = 0.5 * (box.width + box.height);
	scale = 1.0;
	sigma = sigma*scale;

	//目标位置
	center.x = box.x + 0.5 * box.width;
	center.y = box.y + 0.5 * box.height;

	//上下文区域
	cxtRegion.width = (1+padding) * box.width;
	cxtRegion.height = (1+padding) * box.height;
	cxtRegion.x = center.x - cxtRegion.width * 0.5;
	cxtRegion.y = center.y - cxtRegion.height * 0.5;
	cxtRegion &= Rect(0, 0, frame.cols, frame.rows);
	boxRegion=cxtRegion;//输出box区域

	//上文，下文，当前的概率和时空上下文模型
	cxtPriorPro = Mat::zeros(cxtRegion.height, cxtRegion.width, CV_64FC1);
	cxtPosteriorPro = Mat::zeros(cxtRegion.height, cxtRegion.width, CV_64FC1);
	STModel = Mat::zeros(cxtRegion.height, cxtRegion.width, CV_64FC1);
	STCModel = Mat::zeros(cxtRegion.height, cxtRegion.width, CV_64FC1);

	//创建Hamming窗
	hammingWin = Mat::zeros(cxtRegion.height, cxtRegion.width, CV_64FC1);
	createHammingWin();

	Mat gray;
	cvtColor(frame, gray, CV_RGB2GRAY);

	//减去区域内的平均强度进行归一化
	Scalar average = mean(gray(cxtRegion));
	Mat context;
	gray(cxtRegion).convertTo(context, CV_64FC1, 1.0, - average[0]);

	//乘以一个Hamming窗(点乘点)来抑制图像边界效应
	context = context.mul(hammingWin);

	//从第一帧学习时空上下文模型
	learnSTCModel(context);
}

/******** STCTracker:计算置信图获取最大位置*******/
void STCTracker::tracking(const Mat frame, Rect &trackBox,Rect &boxRegion,int FrameNum){
	Mat gray;
	cvtColor(frame, gray, CV_RGB2GRAY);

	//减去区域的平均强度进行归一化
	Scalar average = mean(gray(cxtRegion));
	Mat context;
	gray(cxtRegion).convertTo(context, CV_64FC1, 1.0, - average[0]);

	//乘以一个Hamming窗来削弱边界效应的影响
	context = context.mul(hammingWin);

	//获取上文概率
	getCxtPriorPosteriorModel(context);

	//提取上文概率的离散傅立叶变换
	Mat priorFourier;
	Mat planes1[] = {cxtPriorPro, Mat::zeros(cxtPriorPro.size(), CV_64F)};
    merge(planes1, 2, priorFourier);
	dft(priorFourier, priorFourier);

	//提取当前概率的离散逆傅立叶变换
	Mat STCModelFourier;
	Mat planes2[] = {STCModel, Mat::zeros(STCModel.size(), CV_64F)};
    merge(planes2, 2, STCModelFourier);
	dft(STCModelFourier, STCModelFourier);

	//计算乘积
	Mat postFourier;
	complexOperation(STCModelFourier, priorFourier, postFourier, 0);

	//提取下文概率的离散快速傅立叶逆变换作为Confidence Map(置信图)
	Mat confidenceMap;
	dft(postFourier, confidenceMap, DFT_INVERSE | DFT_REAL_OUTPUT| DFT_SCALE);

	//找到最大的位置
	Point point;
	double  maxVal;
	minMaxLoc(confidenceMap, 0, &maxVal, 0, &point);
	maxValue.push_back(maxVal);

		/***********使用公式（15）更新Scale(尺度)**********/
	if (FrameNum%(num+2)==0){
		double scale_curr=0.0;

		for (int k=0;k<num;k++){
			scale_curr+=sqrt(maxValue[FrameNum-k-2]/maxValue[FrameNum-k-3]);
		}

		scale=(1-lambda)*scale+lambda*(scale_curr/num);

		sigma=sigma*scale;

	}
	//更新center, trackBox和上下文区域
	center.x = cxtRegion.x + point.x;
	center.y = cxtRegion.y + point.y;
	trackBox.x = center.x - 0.5 * trackBox.width;
	trackBox.y = center.y - 0.5 * trackBox.height;
	trackBox &= Rect(0, 0, frame.cols, frame.rows);
	//boundary
	cxtRegion.x = center.x - cxtRegion.width * 0.5;
	if (cxtRegion.x<0){
		cxtRegion.x=0;
	}
	cxtRegion.y = center.y - cxtRegion.height * 0.5;
	if (cxtRegion.y<0){
		cxtRegion.y=0;
	}
	if (cxtRegion.x+cxtRegion.width>frame.cols){
		cxtRegion.x=frame.cols-cxtRegion.width;
	}
	if (cxtRegion.y+cxtRegion.height>frame.rows){
		cxtRegion.y=frame.rows-cxtRegion.height;
	}


	boxRegion=cxtRegion;
	//从当前帧学习STC模型来跟踪下一帧
	average = mean(gray(cxtRegion));

	gray(cxtRegion).convertTo(context, CV_64FC1, 1.0, - average[0]);

	context = context.mul(hammingWin);
	learnSTCModel(context);
}
