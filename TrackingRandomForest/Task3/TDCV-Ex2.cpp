// TDCV-Ex2.cpp : Ce fichier contient la fonction 'main'. L'exécution du programme commence et se termine à cet endroit.
// the following line should ABSOLUTELY be the first one

#include<opencv2/opencv.hpp>
#include<iostream>
#include "hog_visualization.h"
#include "img_functions.h"
#include "dataset_functions.h"
#include "randomForest.h"
#include "bounding_box.h"
//#include "plot.hpp"

#include <boost/filesystem.hpp>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <opencv2/ml.hpp>
#include <string>

#define ABS_PATH "D:/LouiseLoesch/telecom_paristech/3A/tracking and detection for CV/Projet 2/Projet2/"
//#define ABS_PATH "D:/data/Travail/Samuel/1819-TUM/IN2356-TrackingAndDetectionInCV/repository/ExerciceRanfo/task3/"

using namespace std;
using namespace cv;
namespace fs = boost::filesystem;

int main()
{
	/* initialize random seed: */
	srand(time(NULL));

	// Task 3
	// Folders train/test Louise
	std::string trainFolder3("data/task3/train/0");
	std::string testFolder3("data/task3/test/");
	std::string gtfolder("/data/task3/gt/");
	std::string resultsfolder("/data/task3/results/");

	// Folders train/test Sam
	//std::string trainFolder3("data/train/0");
	//std::string testFolder3("data/test/");
	//std::string gtfolder("/data/gt/");
	//std::string resultsfolder("/data/results/");


	// Parameters of the experiment
	int num_classes = 4;
	int resize_px = 64;
	int cells_px = 16;
	int blocks_px = 32;
	int blocks_stride = 16;

	//Train dataset constitution
	Mat labels3, features3;
	std::string absoluteTrainPath3(ABS_PATH + trainFolder3);
	std::cout << "creation of train dataset" << endl;

	// Constitution of the features matrix - training set
	create_dataset(num_classes, absoluteTrainPath3, &labels3, &features3, resize_px, cells_px, blocks_px, blocks_stride, 1);

	//Test dataset constitution
	//Mat test_labels, test_features;
	std::string absoluteTestPath3(ABS_PATH + testFolder3);
	std::cout << "creation of test dataset" << endl;
	
	//initializing random forest
	RandomForest randomforest;
	randomforest.nbDTrees = 15;
	randomforest.MaxCategories = 4;
	randomforest.MaxDepth = 15;
	randomforest.MinSampleCount = 2;
	randomforest.CVFolds = 1;

	randomforest.create();
	randomforest.train(features3, labels3);

	//variables for the test
	Mat image; //test image
	vector<string> vectorImg; //vector of file names
	string imgName; //name of test file
	int track = 0; //keep track of the nb of test images we read
	vector<int> correct_pred (7 ,0); // nb of correct predicted box for each threshold
	int nb_gt=0;
	vector<int> nb_pred(7, 0); // nb of predicted box for each threshold
	vector<int> threshold{ 50, 55, 60, 65, 70, 75, 80}; //thresholds for the NMS
	vector<float> precision;
	vector<float> recall;
	string resultsFolderAbs = ABS_PATH + resultsfolder;
	float threshold_IOU = 0.2;

	int windowsSizeArray[3] = { 80,112,144 };
	get_files_folder(absoluteTestPath3, &vectorImg);

	for (std::vector<string>::iterator it = vectorImg.begin(); it != vectorImg.end(); ++it) {// go through the test pictures

		imgName = *it;
		std::cout << imgName.substr(0, 4) + ".gt" << endl;
		image = imread(absoluteTestPath3 + imgName);
		imshow(imgName,image);

		//create the ground truth boxes 
		std::map<int, Rect> ground_truth;
		tie(ground_truth, nb_gt) = create_ground_truth(ABS_PATH + gtfolder + imgName.substr(0, 4) + ".gt.txt", nb_gt);
		
		vector<Rect> rects = get_multi_sliding_windows(image, track, windowsSizeArray, 3);
		//predict the class of each window
		Mat y_pred;
		Mat confidence;
		tie(y_pred, confidence)= prediction(&randomforest, image, track, rects, resize_px, cells_px, blocks_px);
		std::cout << "sliding windows done" << endl;

		for (int i = 0; i < threshold.size(); i++) {
			image = imread(absoluteTestPath3 + imgName);
			vector<int> resultsLabels;
			vector<float> resultsConfidence;
			vector<Rect> resultsRect;
			//keep windows that detects an object with a confidence above the threshold
			classify_windows(image, track, rects, y_pred, confidence, threshold[i] , resultsFolderAbs + std::to_string(threshold[i]) + "/",&resultsRect, &resultsLabels, &resultsConfidence);
			// compute NMS and update the number of predicted boxes
			nb_pred[i] = NMS(image, track, nb_pred[i], resultsFolderAbs + std::to_string(threshold[i]) + "/", &resultsRect, &resultsLabels, &resultsConfidence, threshold_IOU);
			// Evaluate the detection result
			correct_pred[i] = int_over_union(resultsRect, resultsLabels, correct_pred[i], ground_truth, image, track, resultsFolderAbs + std::to_string(threshold[i]) + "/");
		}
		
		track++;
	}
	//compute the precision and recall
	for (int i = 0; i < threshold.size(); i++) {
		std::cout << "the number of predicted bounding boxes " << nb_pred[i] << endl;
		std::cout << " the number of ground truth bounding boxes" << nb_gt << endl;
		precision.push_back( (float)correct_pred[i] / (float)nb_pred[i]);
		recall.push_back( (float)correct_pred[i] / (float)nb_gt);
		std::cout << "for threshold "<< threshold[i]<<" precision " << precision[i] << " recall " << recall[i] << endl;
	}
	visualize_vector(precision, "precision", 7);
	visualize_vector(recall, "recall", 7);
	
	waitKey(0);
	return 0;
}