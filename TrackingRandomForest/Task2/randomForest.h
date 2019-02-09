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

	// getting N random points among "max_number"
	void getNRand(std::unordered_set<int> * index, int max_number, int number_samples);

	// getting N distinct random points among "max_number"
	void getNDistinctRand(std::unordered_set<int> * index, int max_number, int number_samples);

	// getting subsampling fo initial data (number defined by ratio)
	void subsample(cv::Mat * sublabels, cv::Mat * subfeatures, cv::Mat labels, cv::Mat features, float ratio);
	void create();

	// training of the random forest
	void train(Mat features, Mat labels);

	// predicting with the random forest
	void predict(cv::InputArray samples, Mat test_labels, cv::Mat * resp, cv::Ptr<cv::ml::TrainData> testData);

};

#endif