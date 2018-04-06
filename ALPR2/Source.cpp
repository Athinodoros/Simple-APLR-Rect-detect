#include "Source.h"
#include <opencv2\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sys/types.h>
#include <windows.h>

#include <sstream>
#include <vector>
#include <iterator>
#include <fstream>

using namespace cv;
using namespace std;


Mat src; Mat src_gray;
Mat new_image;
int thresh = 100;
int max_thresh = 255;
RNG rng(12345);

/// Function header
void thresh_callback(int, void*);
void read_directory(const std::string& name, vector<std::string>& v);
bool verifySizes(RotatedRect candidate);
void GammaCorrection(Mat& src, Mat& dst, float fGamma);
void findPossiblePlates(Mat& in , Mat& out);
void findPossiblePlate(String out);

vector<string> get_all_files_names_within_folder(string folder)
{
	vector<string> names;
	string search_path = folder + "/plates/*.*";
	WIN32_FIND_DATA fd;
	HANDLE hFind = ::FindFirstFile(search_path.c_str(), &fd);
	if (hFind != INVALID_HANDLE_VALUE) {
		do {
			// read all (real) files in current folder
			// , delete '!' read other 2 default folder . and ..
			if (!(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)) {
				names.push_back(fd.cFileName);
			}
		} while (::FindNextFile(hFind, &fd));
		::FindClose(hFind);
	}
	return names;
}

int main(int argv, char** argc) {
	//findPossiblePlates() is for a camera feed
	
	
	/*
	
	VideoCapture cap(0); // open the default camera
	if (!cap.isOpened())  // check if we succeeded
		return -1;

	Mat edges;
	while (true)
	{
		Mat frame;
		bool read = cap.read(frame);// get a new frame from camera
		findPossiblePlates(frame, edges);
		imshow("original", frame);
		if (waitKey(500) == 27)
		{
			cout << "esc key is pressed by user" << endl;
			break;
		}
	}
	*/
	// the camera will be deinitialized automatically in VideoCapture destructor

	

	// this is for files in a folder 
	vector<cv::String> fn;
	cv::glob("./plates/", fn, true); // recurse
	for (size_t k = 0; k<fn.size(); ++k)
	{
		findPossiblePlate(fn[k]);
		waitKey(300);
		continue;
	}
	
	/*
	findPossiblePlate("1test.jpg");
	
	 *
	 *
	 *
	 *
	 *
	 *
	 *
	 */
	//findPossiblePlates("testt.jpg");

	waitKey(0);

	////////////////////////
	/*
	
	new_image = Mat::zeros(input.size(), input.type());

	for (int y = 0; y < input.rows; y++) {
		for (int x = 0; x < input.cols; x++) {
			for (int c = 0; c < 3; c++) {
				new_image.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(2.0*(input.at<Vec3b>(y, x)[c]) + 0);
			}
		}
	}



	//cvtColor(input, gray, CV_RGB2BGR555,0);

	// compute mask (you could use a simple threshold if the image is always as good as the one you provided)
	cv::Mat mask;
	cv::threshold(gray, mask, 0, 255, CV_THRESH_BINARY_INV | CV_THRESH_OTSU);

	// find contours (if always so easy to segment as your image, you could just add the black/rect pixels to a vector)
	std::vector<std::vector<cv::Point>> contours;
	//std::vector<cv::Vec4i> hierarchy;
	cv::findContours(mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	/// Draw contours and find biggest contour (if there are other contours in the image, we assume the biggest one is the desired rect)
	// drawing here is only for demonstration!
	int biggestContourIdx = -1;
	float biggestContourArea = 0;
	cv::Mat drawing = cv::Mat::zeros(mask.size(), CV_8UC3);
	for (int i = 0; i< contours.size(); i++)
	{
		cv::Scalar color = cv::Scalar(200, 100, 0);
		drawContours(drawing, contours, i, color, 1, 8, hierarchy, 0, cv::Point());
		cv::RotatedRect boundingBox = cv::minAreaRect(contours[i]);
		cv::Point2f corners[4];
		boundingBox.points(corners);
		cv::line(drawing, corners[0], corners[1], cv::Scalar(150, 255, 255));
		cv::line(drawing, corners[1], corners[2], cv::Scalar(150, 255, 255));
		cv::line(drawing, corners[2], corners[3], cv::Scalar(150, 255, 255));
		cv::line(drawing, corners[3], corners[0], cv::Scalar(150, 255, 255));

		float ctArea = (float)cv::contourArea(contours[i]);
		if (ctArea > biggestContourArea)
		{
			biggestContourArea = ctArea;
			biggestContourIdx = i;
		}
	}


	// if no contour found
	if (biggestContourIdx < 0)
	{
		cout << "no contour found" << std::endl;
		return 1;
	}
	else {
		cv::Scalar color = cv::Scalar(200, 100, 250);
		drawContours(drawing, contours, biggestContourIdx, color, 1, 8, hierarchy, 0, cv::Point());
	}

	// compute the rotated bounding rect of the biggest contour! (this is the part that does what you want/need)
	cv::RotatedRect boundingBox = cv::minAreaRect(contours[biggestContourIdx]);
	// one thing to remark: this will compute the OUTER boundary box, so maybe you have to erode/dilate if you want something between the ragged lines



	// draw the rotated rect
	cv::Point2f corners[4];
	boundingBox.points(corners);
	cv::line(drawing, corners[0], corners[1], cv::Scalar(255, 255, 255));
	cv::line(drawing, corners[1], corners[2], cv::Scalar(255, 255, 255));
	cv::line(drawing, corners[2], corners[3], cv::Scalar(255, 255, 255));
	cv::line(drawing, corners[3], corners[0], cv::Scalar(255, 255, 255));
	//cout << corners[0] << endl;
	//Rect dim2Crop = Rect(corners[0], corners[1], corners[2], corners[3]);
	//input = input(dim2Crop);

	// display
	cv::imshow("input", input);
	cv::imshow("drawing", drawing);

	//cv::imshow("grayOr", gray);
	cv::imshow("src", src);
	//cv::imshow("gray",  new_image);
	cv::waitKey(0);

	cv::imwrite("rotatedRect.png", drawing);

	destroyAllWindows();
	*/
	return 0;

}

void findPossiblePlates(Mat& in , Mat& out) {
	Mat input = in;
	//Mat gray = imread("testttt.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	Mat src =  input;
	Mat contFixed;

	cvtColor(src, src, CV_BGR2GRAY);
	//imshow("bw", src);


	equalizeHist(src, contFixed);
	equalizeHist(src, src);

	//imshow("contr", src);
	blur(src, src, Size(20, 1));
	threshold(src, src, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
	Sobel(src, src, CV_8U, 1, 0, 7, 1, 0);
	//GammaCorrection(src, src, 2.5);
	Mat element = getStructuringElement(MORPH_RECT, Size(50, 5));
	morphologyEx(src, src, CV_MOP_CLOSE, element);

	imshow("morph", src);
	vector< vector< Point> > contours2;
	findContours(src, contours2, // a vector of contours
		CV_RETR_EXTERNAL, // retrieve the external contours
		CV_CHAIN_APPROX_TC89_L1); // all pixels of each contour

							   //Start to iterate to each contour found
	vector<vector<Point> >::iterator itc = contours2.begin();
	vector<RotatedRect> rects;
	//Remove patch that has no inside limits of aspect ratio and area.
	while (itc != contours2.end()) {
		//Create bounding rect of object
		RotatedRect mr = minAreaRect(Mat(*itc));
		if (!verifySizes(mr)) { // || !isContourConvex(contours2)
			itc = contours2.erase(itc);
		}
		else {
			++itc;
			rects.push_back(mr);
		}
	}
	std::vector<cv::Vec4i> hierarchy;
	Mat test = Mat::zeros(src.size(), CV_8UC3);
	for (int i = 0; i < contours2.size(); i++)
	{
		

		cv::Scalar color = cv::Scalar(200, 100, 0);
		drawContours(test, contours2, i, color, 1, 8, hierarchy, 0, cv::Point());
	}
	imshow("contours", test);

	for (int j = 0; j < rects.size(); j++) {
		cv::RotatedRect boundingBox = rects[j];
		cv::Point2f corners[4];
		boundingBox.points(corners);
		cv::line(in, corners[0], corners[1], cv::Scalar(0, 255, 255));
		cv::line(in, corners[1], corners[2], cv::Scalar(0, 255, 255));
		cv::line(in, corners[2], corners[3], cv::Scalar(0, 255, 255));
		cv::line(in, corners[3], corners[0], cv::Scalar(0, 255, 255));

		// cvDrawRect(input,corners[0], (50, 50), (0, 255, 0), 2)

	}

	if (rects.size() >= 1) {
		for (int i = 0; i < rects.size(); i++) {
			Point2f pts[4];

			rects[i].points(pts);

			// Does the order of the points matter? I assume they do NOT.
			// But if it does, is there an easy way to identify and order 
			// them as topLeft, topRight, bottomRight, bottomLeft?

			cv::Point2f src_vertices[3];
			src_vertices[0] = pts[0];
			src_vertices[1] = pts[1];
			src_vertices[2] = pts[3];
			if (pts[0].x > pts[2].x) {
				src_vertices[0] = pts[0];
				src_vertices[1] = pts[1];
				src_vertices[2] = pts[3];
			}
			else {
				src_vertices[0] = pts[3];
				src_vertices[1] = pts[0];
				src_vertices[2] = pts[2];
			}
			//src_vertices[3] = not_a_rect_shape[3];

			Point2f dst_vertices[3];
			dst_vertices[0] = Point(0, 0);
			dst_vertices[1] = Point(rects[i].boundingRect().width - 1, 0);
			dst_vertices[2] = Point(0, rects[i].boundingRect().height - 1);

			Mat warpAffineMatrix = getAffineTransform(src_vertices, dst_vertices);

			cv::Mat rotated;
			cv::Size size(rects[i].boundingRect().width, rects[i].boundingRect().height);
			warpAffine(contFixed, rotated, warpAffineMatrix, size, INTER_LINEAR, BORDER_CONSTANT);
			//flip by 180
			if (rects[i].boundingRect().width < rects[i].boundingRect().height) {
				Point2f src_center(rotated.cols / 2.0F, rotated.rows / 2.0F);
				Mat rot_mat = getRotationMatrix2D(src_center, 180, 1.0);
				warpAffine(rotated, rotated, rot_mat, rotated.size());
			}
			out = src;
			imshow("rotated", rotated);
		}
	}
	/*

	if (contours2.size() >0) {

		
	}
	*/
}



void findPossiblePlate(String fn) {
	Mat input = imread(fn, CV_LOAD_IMAGE_UNCHANGED);
	Mat gray = imread(fn , CV_LOAD_IMAGE_GRAYSCALE);

	Mat src = input;
	Mat contFixed;

	cvtColor(src, src, CV_BGR2GRAY);
	//imshow("bw", src);


	equalizeHist(src, contFixed);
	equalizeHist(src, src);

	imshow("contr", src);
	blur(src, src, Size(20, 1));
	threshold(src, src, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);
	Sobel(src, src, CV_8U, 1, 0, 7, 1, 0);
	//GammaCorrection(src, src, 2.5);
	Mat element = getStructuringElement(MORPH_RECT, Size(50, 5));
	morphologyEx(src, src, CV_MOP_CLOSE, element);

	//imshow("morph", src);
	vector< vector< Point> > contours2;
	findContours(src, contours2, // a vector of contours
		CV_RETR_EXTERNAL, // retrieve the external contours
		CV_CHAIN_APPROX_TC89_L1); // all pixels of each contour

								  //Start to iterate to each contour found
	vector<vector<Point> >::iterator itc = contours2.begin();
	vector<RotatedRect> rects;
	//Remove patch that has no inside limits of aspect ratio and area.
	while (itc != contours2.end()) {
		//Create bounding rect of object
		RotatedRect mr = minAreaRect(Mat(*itc));
		if (!verifySizes(mr)) { // || !isContourConvex(contours2)
			itc = contours2.erase(itc);
		}
		else {
			++itc;
			rects.push_back(mr);
		}
	}
	std::vector<cv::Vec4i> hierarchy;
	Mat test = Mat::zeros(src.size(), CV_8UC3);
	for (int i = 0; i < contours2.size(); i++)
	{


		cv::Scalar color = cv::Scalar(200, 100, 0);
		drawContours(test, contours2, i, color, 1, 8, hierarchy, 0, cv::Point());
	}
	//imshow("contours", test);

	float mostStraight;
	RotatedRect msr;
	if (rects.size() > 0) {
		msr = rects[0];
		(rects[0].angle <0) ? mostStraight = rects[0].angle *(-1) : mostStraight = rects[0].angle;

	}

	for (int j = 0; j < rects.size(); j++) {
		cv::RotatedRect boundingBox = rects[j];
		cv::Point2f corners[4];
		boundingBox.points(corners);
		cv::line(input, corners[0], corners[1], cv::Scalar(0, 255, 255));
		cv::line(input, corners[1], corners[2], cv::Scalar(0, 255, 255));
		cv::line(input, corners[2], corners[3], cv::Scalar(0, 255, 255));
		cv::line(input, corners[3], corners[0], cv::Scalar(0, 255, 255));
		if (boundingBox.angle < 0) {
			if (mostStraight >(boundingBox.angle + 90)) {
				mostStraight = boundingBox.angle;
				msr = boundingBox;
			}
		}
		else {
			if (mostStraight > boundingBox.angle) {
				mostStraight = boundingBox.angle;
				msr = boundingBox;
			}
		}
		cout << to_string(mostStraight) << endl;
		// cvDrawRect(input,corners[0], (50, 50), (0, 255, 0), 2)
	}

	cv::Point2f corners[4];
	msr.points(corners);
	cv::line(input, corners[0], corners[1], cv::Scalar(100, 50, 255));
	cv::line(input, corners[1], corners[2], cv::Scalar(100, 50, 255));
	cv::line(input, corners[2], corners[3], cv::Scalar(100, 50, 255));
	cv::line(input, corners[3], corners[0], cv::Scalar(100, 50, 255));

	if (rects.size() >= 1) {
		for (int i = 0; i < rects.size(); i++) {
			Point2f pts[4];
			rects[i].points(pts);

			// Does the order of the points matter? I assume they do NOT.
			// But if it does, is there an easy way to identify and order 
			// them as topLeft, topRight, bottomRight, bottomLeft?

			cv::Point2f src_vertices[3];
			//putText(in, to_string(mostStraight), pts[0],FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 200, 250), 1, CV_AA);
			putText(input, "0-" + to_string(pts[0].x), pts[0], FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 200, 250), 1, CV_AA);
			putText(input, "1-" + to_string(pts[1].x), pts[1], FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 200, 250), 1, CV_AA);
			putText(input, "2-" + to_string(pts[2].x), pts[2], FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 200, 250), 1, CV_AA);
			putText(input, "3-" + to_string(pts[3].x), pts[3], FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(200, 200, 250), 1, CV_AA);
			if (pts[0].x > pts[2].x) {
				src_vertices[0] = pts[0];
				src_vertices[1] = pts[1];
				src_vertices[2] = pts[3];
			}
			else {
				src_vertices[0] = pts[3];
				src_vertices[1] = pts[0];
				src_vertices[2] = pts[2];
			}
			//src_vertices[3] = not_a_rect_shape[3];

			Point2f dst_vertices[3];
			dst_vertices[0] = Point(0, 0);
			dst_vertices[1] = Point(rects[i].boundingRect().width - 1, 0);
			dst_vertices[2] = Point(0, rects[i].boundingRect().height - 1);

			Mat warpAffineMatrix = getAffineTransform(src_vertices, dst_vertices);

			cv::Mat rotated;
			cv::Size size(rects[i].boundingRect().width, rects[i].boundingRect().height);
			warpAffine(gray, rotated, warpAffineMatrix, size, INTER_LINEAR, BORDER_CONSTANT);
			Point2f src_center(rotated.cols / 2.0F, rotated.rows / 2.0F);
			Mat rot_mat = getRotationMatrix2D(src_center, 180, 1.0);
			warpAffine(rotated, rotated, rot_mat, size);
			imshow("rotated", rotated);

			imwrite(fn.substr(0,fn.size()-4) + to_string(i)+".jpg" , rotated);
			
		}
	}


	
}


void read_directory(const std::string& name, vector<std::string>& v)
{
	std::string pattern(name);
	pattern.append("\\*");
	WIN32_FIND_DATA data;
	HANDLE hFind;
	if ((hFind = FindFirstFile(pattern.c_str(), &data)) != INVALID_HANDLE_VALUE) {
		do {
			v.push_back(data.cFileName);
		} while (FindNextFile(hFind, &data) != 0);
		FindClose(hFind);
	}
}

void GammaCorrection(Mat& src, Mat& dst, float fGamma)

{

	unsigned char lut[256];

	for (int i = 0; i < 256; i++)

	{

		lut[i] = saturate_cast<uchar>(pow((float)(i / 255.0), fGamma) * 255.0f);

	}

	dst = src.clone();

	const int channels = dst.channels();

	switch (channels)

	{

	case 1:

	{

		MatIterator_<uchar> it, end;

		for (it = dst.begin<uchar>(), end = dst.end<uchar>(); it != end; it++)

			*it = lut[(*it)];

		break;

	}

	case 3:

	{

		MatIterator_<Vec3b> it, end;

		for (it = dst.begin<Vec3b>(), end = dst.end<Vec3b>(); it != end; it++)

		{

			(*it)[0] = lut[((*it)[0])];

			(*it)[1] = lut[((*it)[1])];

			(*it)[2] = lut[((*it)[2])];

		}

		break;

	}

	}

}



bool verifySizes(RotatedRect candidate) {
	float error = 0.4f;
	//Spain car plate size: 52x11 aspect 4,7272
	const float aspect = 4.7272f;
	//Set a min and max area. All other patches are discarded
	int min = (int )(15 * aspect * 15); // minimum area
	int max = (int)(125 * aspect * 125); // maximum area
								  //Get only patches that match to a respect ratio.
	float rmin = aspect - aspect * error;
	float rmax = aspect + aspect * error;
	int area =(int) (candidate.size.height * candidate.size.width);
	float r = (float)candidate.size.width / (float)candidate.size.height;
	if (r < 1)
		r = 1 / r;
	if ((area < min || area > max) || (r < rmin || r > rmax)) {
		return false;
	}
	else {
		return true;
	}
}
Source::Source()
{

}


Source::~Source()
{
}
