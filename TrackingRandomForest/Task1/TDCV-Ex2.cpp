// TDCV-Ex2.cpp :

#include<opencv2/opencv.hpp>
#include<iostream>
#include "hog_visualization.h"
#include "img_functions.h"

using namespace std;
using namespace cv;

int main()
{
	// Reading a picture
	Mat img = imread("D:/data/Travail/Samuel/1819-TUM/IN2356-TrackingAndDetectionInCV/repository/ExerciceRanfo/task1/data/obj1000.jpg");
	
	// Showing picture
	namedWindow("image", WINDOW_NORMAL);
	imshow("image", img);

	// Writing a picture : See details bout parameters in https://docs.opencv.org/3.0-beta/modules/imgcodecs/doc/reading_and_writing_images.html
	vector<int> compression_params;

	try {
		imwrite("D:/data/Travail/Samuel/1819-TUM/IN2356-TrackingAndDetectionInCV/repository/ExerciceRanfo/task1/data/copy.png", img, compression_params);
	}
	catch (cv::Exception& ex) {
		fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
		return 1;
	}

	// Convert in black and white and show
	Mat bwImg;
	cvtColor(img, bwImg, cv::COLOR_RGB2GRAY);
	namedWindow("imageBW", WINDOW_NORMAL);
	imshow("imageBW", bwImg);

	// Doing a 180 degrees rotation
	Mat rotatedImg;
	rotate(img, rotatedImg, ROTATE_180);
	namedWindow("imageRotated", WINDOW_NORMAL);
	imshow("imageRotated", rotatedImg);

	// Flipping a picture
	Mat flippedImg;
	flip(img, flippedImg, 0); // =0 : vertical, >0 : horizontal, <0 : both
	namedWindow("imageFlipped", WINDOW_NORMAL);
	imshow("imageFlipped", flippedImg);

	// Padding a picture with 100 pixels in each direction 
	Mat paddedImg;
	int border = 100;
	copyMakeBorder(img, paddedImg, border, border, border, border, BORDER_REPLICATE);
	namedWindow("imagePadded", WINDOW_NORMAL);
	imshow("imagePadded", paddedImg);

	// Resizing a picture to a square first (padding) and then resizing 2 dimensions to the next multiple of 16px.
	Mat resizedImg;
	Size newSize;
	paddingToNextShape(img, resizedImg, 16, 1, newSize);
	namedWindow("imgResized", WINDOW_NORMAL);
	imshow("imgResized", resizedImg);

	// Sources : HoG features implementation
	// (1) wikipedia
	// (2) https://www.learnopencv.com/histogram-of-oriented-gradients/
	// (3) https://stackoverflow.com/questions/44972099/opencv-hog-features-explanation

	// Answer from (3) taken as model

	// Size of the cell (in px)
	int cellSide = 16;
	Size cellSize = cv::Size(cellSide, cellSide);

	// Size of the block (in px)
	Size blockSize = cv::Size(2* cellSide, 2* cellSide);

	// Stride (in px)
	Size blockStride = cv::Size(cellSide, cellSide);

	// Setting the window size to the size of the resized picture
	Size winSize = newSize;

	// Number of bins to consider for HoG
	int nbins = 9;


	HOGDescriptor descr = cv::HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins);

	// Initializing vector to get the full descriptors
	std::vector<float> descriptors = std::vector<float>();
	descr.compute(resizedImg, descriptors);

	// Visualizing HoG features
	visualizeHOG(resizedImg, descriptors, descr, 5);

	waitKey(0);

	return 0;
}