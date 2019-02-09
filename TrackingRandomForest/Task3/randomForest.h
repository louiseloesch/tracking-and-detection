#pragma once
#ifndef DEF_RANDOMFOREST
#define DEF_RANDOMFOREST
#include <string>
#include<opencv2/opencv.hpp>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <cstdlib>
#include <opencv2/ml.hpp>
#include <unordered_set>
#include "dataset_functions.h"

class RandomForest {

	public:

	int nbDTrees;
	int MaxCategories;
	int MaxDepth;
	int MinSampleCount;
	int CVFolds;
	std::vector<cv::Ptr<cv::ml::StatModel>> forest;

	void getNRand(std::unordered_set<int> * index, int max_number, int number_samples);
	void getNDistinctRand(std::unordered_set<int> * index, int max_number, int number_samples);
	void subsample(cv::Mat * sublabels, cv::Mat * subfeatures, cv::Mat labels, cv::Mat features, float ratio);
	void create();
	void train(Mat features, Mat labels);
	void predict(cv::InputArray samples, Mat test_labels, cv::Mat * resp, cv::Ptr<cv::ml::TrainData> testData, cv::Mat * confidence, bool task2);

};

#endif