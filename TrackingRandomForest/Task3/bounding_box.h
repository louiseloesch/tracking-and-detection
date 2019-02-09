#pragma once
#include <string>
#include<opencv2/opencv.hpp>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <cstdlib>
#include <opencv2/ml.hpp>
#include <unordered_set>
#include "randomForest.h"

using namespace std;
using namespace cv;


vector<Rect> get_multi_sliding_windows(Mat& image, int track, int * windowsizeArray, int Nwindows);
cv::Mat create_features_from_rects(Mat& image, int track, vector<Rect> rects, int pic_resize, int cells_size, int blocks_size);
void classify_windows( Mat& image, int track, vector<Rect> rects,Mat y_pred, Mat confidence, int threshold, string resultsFolder, vector<Rect> * resultsRect, vector<int> * resultsLabels, vector<float> * resultsConfidence);
int NMS(Mat& image, int track, int nb_pred, string resultsFolder, vector<Rect> * resultsRect, vector<int> * resultsLabels, vector<float> * resultsConfidence, float threshold_IOU);
int int_over_union(vector<Rect> resultsRect, vector<int> resultsLabels,int correct_pred, std::map<int, Rect> ground_truth, Mat& image, int track, string resultsFolder);
float IOU(Rect rect1, Rect rect2);
std::tuple<Mat, Mat> prediction(RandomForest * randomforest, Mat& image, int track, vector<Rect> rects, int pic_resize, int cells_size, int blocks_size);