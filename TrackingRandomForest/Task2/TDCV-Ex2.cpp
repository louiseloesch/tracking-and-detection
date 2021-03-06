// TDCV-Ex2.cpp : 

#include<opencv2/opencv.hpp>
#include<iostream>
#include "hog_visualization.h"
#include "img_functions.h"
#include "dataset_functions.h"
#include "randomForest.h"

#include <boost/filesystem.hpp>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <opencv2/ml.hpp>

//#define ABS_PATH "D:/LouiseLoesch/telecom_paristech/3A/tracking and detection for CV/Projet 2/Projet2/"
#define ABS_PATH "D:/data/Travail/Samuel/1819-TUM/IN2356-TrackingAndDetectionInCV/repository/ExerciceRanfo/task2/"

using namespace std;
using namespace cv;
namespace fs = boost::filesystem;

int main()
{

	/* initialize random seed: */
	srand(time(NULL));

	// Folders train/test Louise
	//std::string trainFolder("data/task2/train/0");
	//std::string testFolder("data/task2/test/0");

	// Folders train/test Sam
	std::string trainFolder("data/train/0");
	std::string testFolder("data/test/0");

	// Parameters of the experiment
	int num_classes = 6;
	int resize_px = 128;
	int cells_px = 16;
	int blocks_px = 32;
	int blocks_stride = 16;

	//Train dataset constitution
	Mat labels, features;
	std::string absoluteTrainPath(ABS_PATH + trainFolder);
	std::cout << "creation of train dataset" << endl;

	// Constitution of the features matrix - training set
	create_dataset(num_classes, absoluteTrainPath, &labels, &features, resize_px, cells_px, blocks_px, blocks_stride,1);

	//Test dataset constitution
	Mat test_labels, test_features;
	std::string absoluteTestPath(ABS_PATH + testFolder);
	std::cout << "creation of test dataset" << endl;

	// Constitution of the features matrix - test set
	create_dataset(num_classes, absoluteTestPath, &test_labels, &test_features, resize_px, cells_px, blocks_px,blocks_stride,0);

	// Creation of a simple decision binary tree and training
	cv::Ptr<cv::ml::DTrees> decision_tree = cv::ml::DTrees::create();
	decision_tree->setMaxCategories(6);
	decision_tree->setMaxDepth(20);
	decision_tree->setMinSampleCount(2);
	decision_tree->setCVFolds(1);
	cv::Ptr<cv::ml::TrainData> trainData = cv::ml::TrainData::create(features, cv::ml::ROW_SAMPLE, labels);
	decision_tree->train(trainData);
	std::cout << "###########" << endl;
	std::cout << "Binary Tree results" << endl;
	std::cout << "###########" << endl;
	std::cout << "Training completed" << endl;
	
	// Error on the training set
	auto error = decision_tree->calcError(trainData, true, noArray());
	std::cout << "Training set error : " << error << "%" << endl;

	// Creation of the test dataset and making predictions
	cv::Ptr<cv::ml::TrainData> testData = cv::ml::TrainData::create(test_features, cv::ml::ROW_SAMPLE, test_labels);
	decision_tree->predict(test_features);
	Mat y_predict;
	auto test_error = decision_tree->calcError(testData, false, y_predict);
	std::cout << "Test set error : " << test_error << "%" << endl;
	std::cout << "Prediction completed" << endl;

	// Initializing Random Forest
	RandomForest randomForest;
	randomForest.nbDTrees=15;
	randomForest.MaxCategories=6;
	randomForest.MaxDepth=15;
	randomForest.MinSampleCount=2;
	randomForest.CVFolds=1;
	randomForest.create();
	std::cout << "###########" << endl;
	std::cout << "Random Forest results" << endl;
	std::cout << "###########" << endl;

	// Training the random forest
	randomForest.train(features, labels);

	// Predictions of the random forest
	Mat y_pred;
	randomForest.predict(test_features, test_labels, &y_pred, testData);

	// Compute the error of the random forest
	float err = 0;
	for (int i = 0; i < y_pred.size().height; i++) {
		if (y_pred.at<float>(i) != test_labels.at<float>(i))
			err++;
	}
	err = (err / y_pred.size().height)*100;
	std::cout << "error of the random forest : "<<err << "%" << endl;

	waitKey(0);
	return 0;
}