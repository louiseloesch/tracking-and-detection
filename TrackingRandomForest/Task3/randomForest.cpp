#include "randomForest.h"

using namespace std;
using namespace cv;

void RandomForest::getNDistinctRand(std::unordered_set<int> * index, int max_number, int number_samples) {
	int num;
	std::unordered_set<int>::iterator search;

	for (int i = 0; i < number_samples; i++) {

		num = -1;
		while ((num == -1) || (search != (*index).end())) {
			num = rand() % (max_number + 1);
			search = (*index).find(num);
		}
		(*index).insert(num);

	}
}

void RandomForest::subsample(Mat * sublabels, Mat * subfeatures, Mat labels, Mat features, float ratio) {

	cv::Size s = features.size();
	int lengthDataset = s.height;
	int Nsample = round(lengthDataset * ratio);
	Mat rowFeatTmp, rowLabelTmp;

	unordered_set<int> indexSamples;
	getNDistinctRand(&indexSamples, lengthDataset - 1, Nsample);

	int index;
	for (std::unordered_set<int>::iterator it = indexSamples.begin(); it != indexSamples.end(); ++it) {

		index = *it;
		rowFeatTmp = features.row(index);
		rowLabelTmp = labels.row(index);

		(*subfeatures).push_back(rowFeatTmp);
		(*sublabels).push_back(rowLabelTmp);

	}

}

void RandomForest::create() {

	for (int i = 0; i < nbDTrees; i++) {
		cv::Ptr<cv::ml::DTrees> decision_tree = cv::ml::DTrees::create();

		decision_tree->setMaxCategories(MaxCategories);
		decision_tree->setMaxDepth(MaxDepth);
		decision_tree->setMinSampleCount(MinSampleCount);
		decision_tree->setCVFolds(CVFolds);

		forest.push_back(decision_tree);
	}

}

void RandomForest::train(Mat features, Mat labels) {

	float ratio = 0.6;

	for (int i = 0; i < nbDTrees; i++) {
		Mat subfeatures;
		Mat sublabels;

		//creation of the random subset of the training data
		subsample(&sublabels, &subfeatures, labels, features, ratio);

		cv::Ptr<cv::ml::TrainData> sub_trainData = cv::ml::TrainData::create(subfeatures, cv::ml::ROW_SAMPLE, sublabels);

		// train tree
		forest[i]->train(sub_trainData);
		auto y_pred = cv::OutputArray(sublabels);
		y_pred.clear();
		auto error = forest[i]->calcError(sub_trainData, true,y_pred);

		std::cout << "training done and error " << error << endl;
	}
	
}

void RandomForest::predict(cv::InputArray samples, Mat test_labels, cv::Mat * resp, cv::Ptr<cv::ml::TrainData> testData, cv::Mat * confidence,bool task2) {

	cv::Size s = samples.size();

	cv::Mat pred = Mat(s.height,0, CV_32S);
	for (int i = 0; i < nbDTrees; i++) {
		forest[i]->predict(samples,*resp);
		(*resp).convertTo(*resp, CV_32S);
		cv::hconcat(pred, *resp, pred);
		if (task2) {
			auto test_error = forest[i]->calcError(testData, false, noArray());
			std::cout << "predict error " << test_error << endl;
		}
	}
	//visualize_whole_matrix_int(pred, "Labels per tree");

	cv::Mat labelsCount = Mat(s.height, MaxCategories, CV_32S);
	for (int idy = 0; idy < pred.size().height; idy++) {
		for (int x = 0; x < MaxCategories; x++) {
			labelsCount.at<int>(idy, x) = 0;
		}
	}

	for (int idy = 0; idy < pred.size().height; idy++) {
		for (int idx = 0; idx < pred.size().width; idx++) {
			//printf("Feature : %d \n", idy);
			//printf("Class : %d \n", (int)pred.at<float>(idy, idx));
			labelsCount.at<int>(idy, pred.at<int>(idy, idx))++;
		}
	}
	//visualize_whole_matrix_int(labelsCount, "Matrix confusion");

	//visualize_matrix(labelsCount, "my visu", 6);

	int value;
	int index;
	int countR = 0;
	int countO = 0;
	int countJ = 0;
	for (int idy = 0; idy < pred.size().height; idy++) {
		get_max_index(labelsCount.row(idy), &index, &value);
		if (task2)
			printf("Current picture : %d -- Real : %d -- Confidence : %f %% \n", index, test_labels.at<int>(idy), (float) (100.0*value)/nbDTrees);
		(*resp).at<int>(idy) = index;
		float confidence_tmp = (float)(100.0*value) / nbDTrees;
		(*confidence).push_back(confidence_tmp);
		if (confidence_tmp > 50) {
			if (index == 0) {
				countO++;
			}
			if (index == 1) {
				countJ++;
			}
			if (index == 2) {
				countR++;
			}
		}
	}

	std::cout << "Predict -- Rose:" << countR << " Orange:" << countO << " Jaune:" << countJ << endl;

}