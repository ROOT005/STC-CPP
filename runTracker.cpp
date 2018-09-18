#include "STCTracker.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>

#define STC 1

//全局变量
Rect box;
Rect STCbox;
bool drawing_box = false;
bool gotBB = false;
Rect boxRegion;//STC算法显示的上下文区域
bool fromfile =false;
string video;

static double TestTime=0.f;
int frameCount=1;

void readBB(char* file){
	ifstream tb_file (file);
	string line;
	getline(tb_file, line);
	istringstream linestream(line);
	string x1, y1, w1, h1;
	getline(linestream, x1, ',');
	getline(linestream, y1, ',');
	getline(linestream, w1, ',');
	getline(linestream, h1, ',');
	int x = atoi(x1.c_str());
	int y = atoi(y1.c_str());
	int w = atoi(w1.c_str());
	int h = atoi(h1.c_str());
	box = Rect(x, y, w, h);
}

void print_help(void){
	printf("-v    source video\n-b        tracking box file\n");
}

void read_options(int argc, char** argv, VideoCapture& capture){
	for (int i=0; i<argc; i++){
		if (strcmp(argv[i], "-b") == 0){	// read tracking box from file
			//printf("-b%d\n",i);
			if (argc>i){
				readBB(argv[i+1]);
				gotBB = true;
			}else{
				print_help();
			}
		}
	}
}

// bounding box鼠标回调
void mouseHandler(int event, int x, int y, int flags, void *param){
  switch( event ){
	  case CV_EVENT_MOUSEMOVE:
	    if (drawing_box){
	        box.width = x-box.x;
	        box.height = y-box.y;
	    }
	    break;
	  case CV_EVENT_LBUTTONDOWN:
	    drawing_box = true;
	    box = Rect( x, y, 0, 0 );
	    break;
	  case CV_EVENT_LBUTTONUP:
	    drawing_box = false;
	    if( box.width < 0 ){
	        box.x += box.width;
	        box.width *= -1;
	    }
	    if( box.height < 0 ){
	        box.y += box.height;
	        box.height *= -1;
	    }
	    gotBB = true;
	    break;
	  }
}

int main(int argc, char * argv[]){
	String source = "rtsp://admin:12345@192.168.1.113:554/h264/ch1/main/av_stream";
    VideoCapture capture(source);

	Mat frame;
	Mat first;

	//显示帧率
	float rate=capture.get(CV_CAP_PROP_FPS);
	cout<<"rate="<<rate<<endl;

	//注册鼠标画正方形的回调函数
	cvNamedWindow("Tracker", 0);
	cvResizeWindow("Tracker", 640, 480);
	cvSetMouseCallback("Tracker", mouseHandler, NULL );
	//save img path
	string imgFormat="%05d.png";
	char image_name[256];

	capture.set(CV_CAP_PROP_FRAME_WIDTH, 320);
	capture.set(CV_CAP_PROP_FRAME_HEIGHT, 240);

	//初始化
	while(!gotBB){
		if (!fromfile){
			capture >> frame;
		}
		rectangle(frame, box, Scalar(0,0,255));
		imshow("Tracker", frame);
		if (cvWaitKey(33) == 'q'){	return 0; }
	}

	//删除鼠标的回调
	cvSetMouseCallback("Tracker", NULL, NULL );
	//显示pos的box信息
	cout<<"init x="<<box.x<<" int y="<<box.y<<endl;

//条件编译

#ifdef STC
	STCbox=box;
	//STC初始化
	STCTracker stcTracker;
	stcTracker.init(frame, STCbox,boxRegion);
#endif
	Mat last_gray;
	cvtColor(frame, last_gray, CV_RGB2GRAY);

	// Run-time
	Mat current_gray;

	while (1){
		capture >> frame;

		if (frame.empty())
			break;

#ifdef STC
		double t = (double)cvGetTickCount();
		// tracking
		stcTracker.tracking(frame, STCbox,boxRegion,frameCount);
		t = (double)cvGetTickCount() - t;
		cout << "cost time: " << t / ((double)cvGetTickFrequency()*1000000.) <<" s,";
		cout<<cvRound(((double)cvGetTickFrequency()*1000000.)/t)<<"FPS"<<endl;
		cout<<"object size:="<<STCbox.width<<"*"<<STCbox.height<<endl;
#endif
		frameCount++;

		//显示帧信息
		stringstream buf;
		buf << frameCount;
		string num = buf.str();
		putText(frame, num, Point(15, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(100, 0, 255), 3);

#ifdef STC
		//putText(frame, "      STC", Point(80, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 3);
		rectangle(frame, STCbox, Scalar(0, 0, 255), 3);
		//rectangle(frame, boxRegion, Scalar(255, 200, 255), 3);
#endif

		imshow("Tracker", frame);
		sprintf(image_name, imgFormat.c_str(), frameCount);
		imwrite(image_name,frame);
		if ( cvWaitKey(1) == 27 )
			break;
	}
	return 0;
}
