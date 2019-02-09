#include "bounding_box.h"
#include "randomForest.h"

vector<Rect> get_multi_sliding_windows(Mat& image, int track, int * windowsizeArray, int Nwindows)
{
	vector<Rect> rects;
	int step = 8;
	int windowsSize;
	for (int index = 0; index < Nwindows; index++) {
		windowsSize = windowsizeArray[index];
		for (int i = 0; i < image.rows; i += step)
		{
			if ((i + windowsSize) > image.rows) { break; }
			for (int j = 0; j < image.cols; j += step)
			{
				if ((j + windowsSize) > image.cols) { break; }
				Rect rect(j, i, windowsSize, windowsSize);
				rects.push_back(rect);
				//rectangle(image, rect, Scalar(0, 255, 255));
			}
		}
	}
	//imwrite("myImageWithRect.jpg", image);
	return rects;
}

Mat create_features_from_rects(Mat& image, int track, vector<Rect> rects, int pic_resize, int cells_size, int blocks_size) {
	Mat features;
	Size newSize = Size(pic_resize, pic_resize);
	int cellSide = cells_size;
	Size cellSize = cv::Size(cellSide, cellSide);
	Size blockSize = cv::Size(blocks_size, blocks_size);
	Size blockStride = cv::Size(cellSide, cellSide);
	Size winSize = newSize;
	int nbins = 9;
	HOGDescriptor descr = cv::HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins);


	for (auto iter = rects.begin(); iter != rects.end(); iter++) {
		Mat img = image(*iter);
		// Resizing the picture
		Mat resizedImg;
		paddingToNextShape(img, resizedImg, cellSide, SQUARE_RESIZED, newSize);

		// Compute HOG and add to features
		std::vector<float> descriptors = std::vector<float>();
		descr.compute(resizedImg, descriptors);
		Mat descriptorsToAdd = ((Mat)descriptors).reshape(1, 1);
		descriptorsToAdd.convertTo(descriptorsToAdd, CV_32F);
		features.push_back(descriptorsToAdd);
	}
	return features;
}

std::tuple<Mat, Mat> prediction(RandomForest * randomforest, Mat& image, int track, vector<Rect> rects, int pic_resize, int cells_size, int blocks_size) {
	Mat features = create_features_from_rects(image, track, rects, pic_resize, cells_size, blocks_size);
	std::cout << "features created" << endl;
	Mat test_labels;
	cv::Ptr<cv::ml::TrainData> testData;
	Mat y_pred;
	Mat confidence;
	(*randomforest).predict(features, test_labels, &y_pred, testData, &confidence, false);
	return std::make_tuple(y_pred, confidence);
}

void classify_windows( Mat& image, int track, vector<Rect> rects,Mat y_pred,Mat confidence, int threshold, string resultsFolder, vector<Rect> * resultsRect, vector<int> * resultsLabels, vector<float> * resultsConfidence)
{
	vector<int> resultsLabelsTmp;
	vector<float> resultsConfidenceTmp;
	vector<Rect> resultsRectTmp;
	int countR = 0;
	int countO = 0;
	int countJ = 0;
	for (int idy = 0; idy < y_pred.size().height; idy++) {
		if (y_pred.at<int>(idy,0) != 3 && confidence.at<float>(idy,0) > threshold) {
			std::cout << "image " << idy<<" of classe "<< y_pred.at<int>(idy, 0) << "confidence " << confidence.at<float>(idy,0) << endl;
			
			if (y_pred.at<int>(idy, 0) == 0) {
				countO++;
				rectangle(image, rects[idy], Scalar(0, 120, 255));
			}
			if (y_pred.at<int>(idy, 0) == 1) {
				countJ++;
				rectangle(image, rects[idy], Scalar(0, 255, 255));
			}
			if (y_pred.at<int>(idy, 0) == 2) {
				countR++;
				rectangle(image, rects[idy], Scalar(130, 120, 255));
			}
			resultsRectTmp.push_back(rects[idy]);
			resultsLabelsTmp.push_back(y_pred.at<int>(idy, 0));
			resultsConfidenceTmp.push_back(confidence.at<float>(idy, 0));
		}
	}
	//imshow("rect" + to_string(track), image);
	imwrite(resultsFolder + "rect" + to_string(track) + ".jpg", image);

	// Sorting in increasing confidence
	vector<int> perm(resultsLabelsTmp.size(), 0);
	for (int i = 0; i != resultsLabelsTmp.size(); i++) {
		perm[i] = i;
	}

	sort(perm.begin(), perm.end(),
		[&](const float& a, const float& b) {
		return (resultsConfidenceTmp[a] < resultsConfidenceTmp[b]);
		}
	);

	for (int i = 0; i != resultsLabelsTmp.size(); i++) {
		(*resultsLabels).push_back(resultsLabelsTmp[perm[i]]);
		(*resultsRect).push_back(resultsRectTmp[perm[i]]);
		(*resultsConfidence).push_back(resultsConfidenceTmp[perm[i]]);
	}

	return;
}

// Felzenszwalb et al.
int NMS(Mat& image, int track, int nb_pred, string resultsFolder, vector<Rect> * resultsRect, vector<int> * resultsLabels, vector<float> * resultsConfidence, float threshold_IOU) {
	
	int i = (*resultsLabels).size() - 1;
	int k = 1;
	Rect testRect, currentRect;

	while (i >= 0) {

		currentRect = (*resultsRect)[i];
		putText(image, "Class " + to_string((*resultsLabels)[i]) + ": " + to_string((*resultsConfidence)[i]), cv::Point(currentRect.x, currentRect.y), cv::FONT_HERSHEY_COMPLEX_SMALL, 0.5, cv::Scalar(255, 0, 0));
		rectangle(image, currentRect, Scalar(255, 0, 0));

		for (int j = i-1; j >= 0; j--) {

			testRect = (*resultsRect)[j];
			//std::cout << "overlap NMS: "<<IOU(currentRect, testRect) << endl;
			if (IOU(currentRect, testRect) > threshold_IOU) {
				(*resultsLabels).erase((*resultsLabels).begin() + j);
				(*resultsRect).erase((*resultsRect).begin() + j);
				(*resultsConfidence).erase((*resultsConfidence).begin() + j);
			}

		}

		k++;
		i = (*resultsLabels).size() - k;

	}

	imwrite(resultsFolder + "NMS" + to_string(track) + ".jpg", image);
	return nb_pred+k;
}

float IOU(Rect rect1, Rect rect2) {

	Point inter_up;
	Point inter_down;
	inter_up.x = max(rect1.x, rect2.x);
	inter_up.y = max(rect1.y, rect2.y);
	inter_down.x = min(rect1.x + rect1.width, rect2.x + rect2.width);
	inter_down.y = min(rect1.y + rect1.height, rect2.y + rect2.height);

	float intersection_rect = max(0,inter_down.x - inter_up.x)*max(0,inter_down.y - inter_up.y);
	float union_rect = (rect1.width*rect1.height) + (rect2.width*rect2.height) - intersection_rect;
	
	return intersection_rect / union_rect;
	
}

int int_over_union(vector<Rect> resultsRect, vector<int> resultsLabels, int correct_pred, std::map<int, Rect> ground_truth, Mat& image, int track, string resultsFolder) {
	
	int currentLabel;
	Rect currentRect;
	Rect currentGroundTruth;
	float IOU_ratio;

	for (int i = 0; i<resultsLabels.size(); i++) {

		currentLabel = resultsLabels[i];
		currentRect = resultsRect[i];
		currentGroundTruth = ground_truth[currentLabel];
		rectangle(image, Rect(currentGroundTruth.x, currentGroundTruth.y, currentGroundTruth.width, currentGroundTruth.height), Scalar(0, 255, 0));
		IOU_ratio = IOU(currentRect, currentGroundTruth);
		std::cout << " ratio : " << IOU_ratio << endl;

		if (IOU_ratio > 0.5) {
			//prediction is correct
			correct_pred++;
		}

	}

	imwrite(resultsFolder + "ground_truth" + to_string(track) + ".jpg", image);

	return correct_pred;
}
