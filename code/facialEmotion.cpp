//g++ facialEmotion.cpp -o fe -I/usr/local/include/opencv -I/usr/local/include  -L/usr/local/lib -lopencv_core -lopencv_imageproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -lopencv_flann
#include <stdio.h>
// OpenCV
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <cvaux.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imageproc/imageproc.hpp"

#define M 2 // number of training images
#define E 5 //emotions count

static int N = 0;

using namespace cv;
using namespace std;

int happyTrainCount = 1;
int sadTrainCount = 1;
int neutralTrainCount = 1;
int angryTrainCount = 1;
int ecstaticTrainCount = 1;

/*
Function    : usage
Input       : None
Return      : None
Description : The usage guide for the program execution
*/
void usage()
{
	cout<<"==========================================================================================="<<endl;
	cout<<"The program reads a continuous image from the camera and detects emotions on subject's face"<<endl;
	cout<<"Give atleast 5 second gap between changing the emotions on the face for better results"<<endl;
	cout<<"Please run the program in ambient lighting conditions, as it is inconvenient to read images"<<endl;
	cout<<"in poor light. Reflection of light on the face helps the program in clear expression detection"<<endl;
	cout<<"The expression of the person is immediately shown in output and there might be a slight delay in"<<endl;
	cout<<"the smiley image for the corresponding expression of the face."<<endl;
	cout<<"The expressions of the person can be captured and saved using the following keys from the keyboard"<<endl;
	cout<<"------------------------"<<endl;
	cout<<" Key    -    Expression "<<endl;
	cout<<"------------------------"<<endl;
	cout<<"  h     -    Happy"<<endl;
	cout<<"  s     -    Sad"<<endl;
	cout<<"  n     -    Neutral"<<endl;
	cout<<"  e     -    Ecstatic"<<endl;
	cout<<"  a     -    Angry"<<endl;
	cout<<"------------------------"<<endl;
	cout<<"Press 'p' for help"<<endl;
}

/*
Function    : numTrainingImages
Input       : None
Return      : None
Description : Get the count of images present in the training images set
*/
void numTrainingImages()
{
	string line;
	ifstream readFile("../data/trainingImagesCount.txt");
	string h ("happy");
	string s ("sad");
	string n ("neutral");
	string a ("angry");
	string e ("ecstatic");

	if(readFile.is_open())
	{
		while(getline(readFile,line))
		{
			std::string::size_type pos = line.find('=');
			int count = (line[line.length()-1]);
			//get the string till the position of =
			if(pos != std::string::npos)
			{
				if(!(h.compare(0,pos,line)) && count != 1)
					happyTrainCount = count;
				else if(!(s.compare(0,pos,line)) && count != 1)
					sadTrainCount = count; 
				else if(!(n.compare(0,pos,line)) && count != 1)
					neutralTrainCount = count;
				else if(!(a.compare(0,pos,line)) && count != 1)
					angryTrainCount = count;
				else if(!(e.compare(0,pos,line)) && count != 1)
					ecstaticTrainCount = count;
			}
		}

		readFile.close();
	}
}

/*
Function    : writeNumTrainingImages
Input       : None
Return      : None
Description : Set the count of images present in the training images set
*/
void writeNumTrainingImages()
{
	string line;
	fstream writeFile;
	writeFile.open("../data/trainingImagesCount.txt",ios::in | ios::out);
	if(writeFile.is_open())
	{
		
		writeFile <<"happy="<<happyTrainCount<<endl;
		writeFile <<"sad="<<sadTrainCount<<endl;
		writeFile <<"neutral="<<neutralTrainCount<<endl;
		writeFile <<"angry="<<angryTrainCount<<endl;
		writeFile <<"ecstatic="<<ecstaticTrainCount<<endl;
		writeFile.close();
	}
}

/*
Function    : main
Input       : None
Return      : None
Description : 1. Read the training images from the data folder and load to program
              2. Calculate the Eigen Vector Values for the training images and get a Mean value
              3. Load the haarcascade xml file for face detection and start detecting the face
              4. Crop the face image to primarily capture the regions around the lips.
              5. Compare the image read from the camera with that of the training set of images
              6. A match occurs for the corresponding expression captured in the image.
              7. Images of smileys corresponding to different images are loaded on the screen based
                 on what is captured.
              8. Functionality to save your current expression is provided, so that next time 
                 the program can be run with better trained images
*/
int main (int argc, const char * argv[])
{
	//load the user guide once
	usage();

	//Smileys are loaded here to display corresponding to a given expression
	IplImage *image_happy = cvLoadImage("../data/smileys/happy.jpg");
	IplImage *image_sad = cvLoadImage("../data/smileys/sad.jpg");
	IplImage *image_neutral = cvLoadImage("../data/smileys/neutral.jpg");
	IplImage *image_angry = cvLoadImage("../data/smileys/angry.jpg");
	IplImage *image_ecstatic = cvLoadImage("../data/smileys/ecstatic.jpg");

	// read the images size
	CvSize sz = cvGetSize(cvLoadImage("../data/training/happy00.png"));
	N = sz.width * sz.height; // compute the vector image length
	
	// read the training set images
	char file[64];
	Mat I = Mat(N, E, CV_32FC1); //replace 1 : replaced here to add 2 more emotions
	Mat S = Mat::zeros(N, 1, CV_32FC1);
	
	//Load all the training images to program.
	for(int i = 0; i < happyTrainCount; i++)
	{
		sprintf(file, "../data/training/happy%02d.png", i);
		Mat m = imread(file, CV_LOAD_IMAGE_GRAYSCALE);
		m = m.t();
		m = m.reshape(1, N);
		m.convertTo(m, CV_32FC1);
		m.copyTo(I.col(i));
		S = S + m;
	}
	for(int i = 0; i < sadTrainCount; i++)
	{
		sprintf(file, "../data/training/sad%02d.png", i);
		Mat m = imread(file, CV_LOAD_IMAGE_GRAYSCALE);
		m = m.t();
		m = m.reshape(1, N);
		m.convertTo(m, CV_32FC1);
		m.copyTo(I.col(i + E/5));
		S = S + m;
	}
	
	for(int i = 0; i < neutralTrainCount; i++)
	{
		sprintf(file, "../data/training/neutral%02d.png", i);
		Mat m = imread(file, CV_LOAD_IMAGE_GRAYSCALE);
		m = m.t();
		m = m.reshape(1, N);
		m.convertTo(m, CV_32FC1);
		m.copyTo(I.col(i + (int)E/2));
		S = S + m;
	}
	
	for(int i = 0; i < angryTrainCount; i++)
	{
		sprintf(file, "../data/training/angry%02d.png", i);
		Mat m = imread(file, CV_LOAD_IMAGE_GRAYSCALE);
		m = m.t();
		m = m.reshape(1, N);
		m.convertTo(m, CV_32FC1);
		m.copyTo(I.col(i + (int)E/2 + 1)); //account for angry emotions by adding E/2 + 1
		S = S + m;
	}

	for(int i = 0; i < ecstaticTrainCount; i++)
	{
		sprintf(file, "../data/training/ecstatic%02d.png", i);
		Mat m = imread(file, CV_LOAD_IMAGE_GRAYSCALE);
		m = m.t();
		m = m.reshape(1, N);
		m.convertTo(m, CV_32FC1);
		m.copyTo(I.col(i + (int)E/2 + 2)); //account for angry emotions by adding E/2 + 1
		S = S + m;
	}

	// calculate eigenvectors
	Mat Mean = S / (float)E; //replace 2 : 
	Mat A = Mat(N, E, CV_32FC1); //replace 3 : 
	for(int i = 0; i < E; i++) A.col(i) = I.col(i) - Mean;
	Mat C = A.t() * A;
	Mat V, L;
	eigen(C, L, V);

	// compute projection matrix Ut
	Mat U = A * V;
	Mat Ut = U.t();
	
	// project the training set to the faces space
	Mat trainset = Ut * A;

	// prepare for face recognition
	CvMemStorage *storage = cvCreateMemStorage(0);
	CvHaarClassifierCascade *cascade = (CvHaarClassifierCascade*)cvLoad("../data/haarcascade_frontalface_default.xml");
	
	cout << "Starting Camera Capture\n";
	CvCapture* capture = cvCaptureFromCAM(CV_CAP_ANY);
	if(capture == 0)
	{
		cout << "Web Cam not found!\n";
		cout << "Please ensure there is webcam attached to your system\n";
		return 1;
	}
	
	CvScalar red = CV_RGB(250,0,0);
	
	
	cout << "Starting Emotion Recognition\n";
	char c;
	while(1) 
	{
		// Get one frame
		IplImage *image = cvQueryFrame(capture);

		// read the keyboard
		c = waitKey(100);
		if(c == 27) break; // quit the program when ESC is pressed
		else if(c == 'p') usage();

		// find the face
		CvSeq *rects = cvHaarDetectObjects(image, cascade, storage, 1.3, 4, CV_HAAR_DO_CANNY_PRUNING, cvSize(50, 50));
		if(rects->total == 0)
		{
			cvShowImage("result", image);
			cvMoveWindow("result",500,500);
			continue;
		}
		CvRect *roi = 0;
		for(int i = 0; i < rects->total; i++)
		{
			CvRect *r = (CvRect*) cvGetSeqElem(rects, i);
			
			// draw rect
			CvPoint p1 = cvPoint(r->x, r->y);
			CvPoint p2 = cvPoint(r->x + r->width, r->y + r->height);
			cvRectangle(image, p1, p2, red, 3);
			
			// biggest rect
			if(roi == 0 || (roi->width * roi->height < r->width * r->height)) roi = r;
		}
		
		// copy the face in the biggest rect
		int suby = roi->height * 0.6; //changing from previous value of 0.6 just to cover complete face
		roi->height -= suby;
		roi->y += suby;
		int subx = (roi->width - roi->height) / 2 * 0.7;
		roi->width -= subx * 2;
		roi->x += subx;
		cvSetImageROI(image, *roi);
		IplImage *mouthRegion = cvCreateImage(cvSize(100, 100 * 0.7), 8, 3);
		IplImage *smallImageGray = cvCreateImage(cvGetSize(mouthRegion), IPL_DEPTH_8U, 1);
		cvResize(image, mouthRegion);
		cvCvtColor(mouthRegion, smallImageGray, CV_RGB2GRAY);
		cvEqualizeHist(smallImageGray, smallImageGray);
		cvResetImageROI(image);
		
		//Switch statement to capture the currently read images from the camera
		//This helps in training the algorithm in a better way for further processing
		switch(c) // capture a frame when H (happy) or S (sad) is pressed
		{
			char file[32];
			case 'h':
				sprintf(file, "../data/training/happy%02d.png", happyTrainCount);
				cvSaveImage(file, smallImageGray);
				happyTrainCount++;
				cvWaitKey(1000);
				break;
			case 's':
				sprintf(file, "../data/training/sad%02d.png", sadTrainCount);
				cvSaveImage(file, smallImageGray);
				sadTrainCount++;
				cvWaitKey(1000);
				break;
			case 'n':
				sprintf(file, "../data/training/neutral%02d.png", neutralTrainCount);
				cvSaveImage(file, smallImageGray);
				neutralTrainCount++;
				cvWaitKey(1000);
				break;
			case 'a':
				sprintf(file, "../data/training/angry%02d.png", angryTrainCount);
				cvSaveImage(file, smallImageGray);
				angryTrainCount++;
				cvWaitKey(1000);
				break;
			case 'e':
				sprintf(file, "../data/training/ecstatic%02d.png", ecstaticTrainCount);
				cvSaveImage(file, smallImageGray);
				ecstaticTrainCount++;
				cvWaitKey(1000);
				break;
		}

		//Write the count of training images present currently to a file
		writeNumTrainingImages();

		// recognize emotion on the face
		double min = 1000000000000000.0;
		int mini;
		Mat mouthRegionmat = cvarrToMat(smallImageGray);
		mouthRegionmat = mouthRegionmat.t();
		mouthRegionmat = mouthRegionmat.reshape(1, N);
		mouthRegionmat.convertTo(mouthRegionmat, CV_32FC1);
		Mat proj = Ut * mouthRegionmat;

		// find the minimum distance vector
		for(int i = 0; i < E; i++)  
		{
			double n = norm(proj - trainset.col(i));
			if(min > n)
			{
				min = n;
				mini = i;
			}
		}
		
		//select the smiley preloaded corresponding to the emotion on the face
		namedWindow("logo",CV_WINDOW_NORMAL);
		cvMoveWindow("logo", 400, 0);
		if(mini == 1)
		{
			cvShowImage("logo", image_sad);
			cout << "Subject is Happy!!\n";
		}
		else if(mini == 0) 
		{
			cvShowImage("logo", image_happy);
			cout << "Subject is Sad!!\n";
		}
		else if(mini == 2) 
		{
			cvShowImage("logo", image_neutral);
			cout << "Subject is Neutral!!\n";
		}
		else if(mini == 3) 
		{
			cvShowImage("logo", image_angry);
			cout << "Subject is Angry!!\n";
		}
		else if(mini == 4) 
		{
			cvShowImage("logo", image_ecstatic);
			cout << "Subject is Ecstatic!!\n";
		}
		// show results
		Mat graySmallOutputImage = cvarrToMat(smallImageGray);
		cvMoveWindow("face", 0, 0);
		imshow("face", graySmallOutputImage);
		namedWindow("result", CV_WINDOW_NORMAL);
		cvMoveWindow("result", 500, 500);
		Mat image = cvarrToMat(image);
		imshow("result", image);
		//allow some sleep time for better results
		usleep(100000);
	}
	
	// cleanup
	cvReleaseCapture(&capture);
	cvDestroyWindow("result");
	cvDestroyWindow("logo");
	cvDestroyWindow("face");
	cvReleaseHaarClassifierCascade(&cascade);
	cvReleaseMemStorage(&storage);
	
	cout <<"Task complete!"<<endl;
	return 0;
}
