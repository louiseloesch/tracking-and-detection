#include "dataset_functions.h"

int get_size_dataset(int num_classes, std::string absolutePath) {

	int count = 0;

	for (int classe = 0; classe < num_classes; classe++) {
		fs::path p(absolutePath + std::to_string(classe));
		for (auto i = fs::directory_iterator(p); i != fs::directory_iterator(); i++)
		{
			if (!fs::is_directory(i->path())) //we eliminate directories
			{
				count++;
			}
			else
				continue;
		}
	}

	return count;
}

void add_augmented_picture(HOGDescriptor descr, Mat * labels, Mat * features, Mat resizedImg, int classe) {

	int Nimages = 7;

	// saving Label of the current picture
	for (int i = 0; i < Nimages; i++) {
		(*labels).push_back(classe);
	}

	// Normal image : Compute HOG and add to features
	std::vector<float> descriptors = std::vector<float>();
	descr.compute(resizedImg, descriptors);
	Mat descriptorsToAdd = ((Mat)descriptors).reshape(1, 1);
	descriptorsToAdd.convertTo(descriptorsToAdd, CV_32F);
	(*features).push_back(descriptorsToAdd);

	// Rotated 90 image : Compute HOG and add to features
	Mat rotated90;
	rotate(resizedImg, rotated90, ROTATE_90_CLOCKWISE);
	descriptors = std::vector<float>();
	descr.compute(rotated90, descriptors);
	descriptorsToAdd = ((Mat)descriptors).reshape(1, 1);
	descriptorsToAdd.convertTo(descriptorsToAdd, CV_32F);
	(*features).push_back(descriptorsToAdd);

	// Rotated -90 image : Compute HOG and add to features
	Mat rotatedn90;
	rotate(resizedImg, rotatedn90, ROTATE_90_COUNTERCLOCKWISE);
	descriptors = std::vector<float>();
	descr.compute(rotatedn90, descriptors);
	descriptorsToAdd = ((Mat)descriptors).reshape(1, 1);
	descriptorsToAdd.convertTo(descriptorsToAdd, CV_32F);
	(*features).push_back(descriptorsToAdd);

	// Rotated 180 image : Compute HOG and add to features
	Mat rotated180;
	rotate(resizedImg, rotated180, ROTATE_180);
	descriptors = std::vector<float>();
	descr.compute(rotated180, descriptors);
	descriptorsToAdd = ((Mat)descriptors).reshape(1, 1);
	descriptorsToAdd.convertTo(descriptorsToAdd, CV_32F);
	(*features).push_back(descriptorsToAdd);

	// Flipped 1 : Compute HOG and add to features
	Mat flipped1;
	flip(resizedImg, flipped1, 1);
	descriptors = std::vector<float>();
	descr.compute(flipped1, descriptors);
	descriptorsToAdd = ((Mat)descriptors).reshape(1, 1);
	descriptorsToAdd.convertTo(descriptorsToAdd, CV_32F);
	(*features).push_back(descriptorsToAdd);

	// Flipped 0 : Compute HOG and add to features
	Mat flipped0;
	flip(resizedImg, flipped0, 0);
	descriptors = std::vector<float>();
	descr.compute(flipped0, descriptors);
	descriptorsToAdd = ((Mat)descriptors).reshape(1, 1);
	descriptorsToAdd.convertTo(descriptorsToAdd, CV_32F);
	(*features).push_back(descriptorsToAdd);

	// Flipped -1 : Compute HOG and add to features
	Mat flippedn1;
	flip(resizedImg, flippedn1, -1);
	descriptors = std::vector<float>();
	descr.compute(flippedn1, descriptors);
	descriptorsToAdd = ((Mat)descriptors).reshape(1, 1);
	descriptorsToAdd.convertTo(descriptorsToAdd, CV_32F);
	(*features).push_back(descriptorsToAdd);

}

void get_files_folder(std::string absolutePath, vector<string> * vector) {

	fs::path p(absolutePath);
	for (auto i = fs::directory_iterator(p); i != fs::directory_iterator(); i++)
	{
		if (!fs::is_directory(i->path())) //we eliminate directories
		{
			(*vector).push_back(i->path().filename().string());
		}
		else {
			continue;
		}
	}

}

std::tuple< std::map<int, Rect>,int> create_ground_truth(std::string path, int nb_gt) {
	std::ifstream file(path);
	std::string line;
	std::map<int, Rect> ground_truth;
	while (std::getline(file, line)) {
		std::istringstream stream(line);
		int cpt = 0;
		int classe;
		Rect rectangle;
		while (stream)
		{
			string s;
			if (!std::getline(stream, s, ' ')) break;
			if (cpt == 0)
				classe = std::stoi(s);
			if (cpt == 1)
				rectangle.x = std::stoi(s);
			if (cpt == 2)
				rectangle.y = std::stoi(s);
			if (cpt == 3)
				rectangle.width = abs(std::stoi(s) - rectangle.x);
			if (cpt == 4)
				rectangle.height = abs(std::stoi(s) - rectangle.y);
			cpt++;
			//std::cout << s << endl;
		}
		ground_truth[classe] = rectangle;
		nb_gt++;
	}
	return std::make_tuple(ground_truth, nb_gt);
}


void create_dataset(int num_classes, std::string absolutePath, Mat * labels, Mat * features, int pic_resize, int cells_size, int blocks_size, int blocks_stride, int training) {

	// Resizing the picture so that the whole dataset has the same size
	Size newSize = Size(pic_resize, pic_resize);
	int cellSide = cells_size;
	Size cellSize = cv::Size(cellSide, cellSide);
	Size blockSize = cv::Size(blocks_size, blocks_size);
	Size blockStride = cv::Size(blocks_stride, blocks_stride);
	Size winSize = newSize;
	int nbins = 9;
	HOGDescriptor descr = cv::HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins);

	for (int classe = 0; classe < num_classes; classe++) {
		fs::path p(absolutePath + std::to_string(classe));
		for (auto i = fs::directory_iterator(p); i != fs::directory_iterator(); i++)
		{
			if (!fs::is_directory(i->path())) //we eliminate directories
			{
				cout << absolutePath + std::to_string(classe) + "/" + i->path().filename().string() << endl;
				Mat img = imread(absolutePath + std::to_string(classe) + "/" + i->path().filename().string());

				// Resizing the picture
				Mat resizedImg;
				paddingToNextShape(img, resizedImg, cellSide, SQUARE_RESIZED, newSize);

				if (training==1) {
					add_augmented_picture(descr, labels, features, resizedImg, classe);
				}
				else {
					(*labels).push_back(classe);
					std::vector<float> descriptors = std::vector<float>();
					descr.compute(resizedImg, descriptors);
					Mat descriptorsToAdd = ((Mat)descriptors).reshape(1, 1);
					descriptorsToAdd.convertTo(descriptorsToAdd, CV_32F);
					(*features).push_back(descriptorsToAdd);
				}

			}
			else
				continue;
		}
	}
	/*
	if (training != 1) {
		visualize_whole_matrix(*labels, "first test labels");
	}
	*/
	(*labels).convertTo(*labels, CV_32S);	
	(*features).convertTo(*features, CV_32F);


	//visualize_matrix(features, "Features", 10);

}

void get_max_index(cv::Mat matrix, int * index, int * value) {

	*index = 0;
	*value = 0;

	for (int i = 0; i < matrix.size().height; i++) {
		for (int j = 0; j < matrix.size().width; j++) {
			if (matrix.at<int>(i, j) >= *value) {
				*value = matrix.at<int>(i, j);
				*index = j * matrix.size().height + i;
			}
		}
	}

}

void visualize_vector(std::vector<string> labels, std::string label, int num_elements) {

	printf("Visualization : %s \n", label);
	printf("[");
	for (std::vector<string>::size_type i = 0; i != labels.size() && i!=num_elements; i++) {
		printf("%s,",labels[i]);
	}
	printf("...]\n");

}

void visualize_vector(std::vector<float> values, std::string label, int num_elements) {

	printf("Visualization : %s \n", label);
	printf("[");
	for (std::vector<string>::size_type i = 0; i != values.size() && i != num_elements; i++) {
		printf("%f,", values[i]);
	}
	printf("...]\n");

}

void visualize_matrix(cv::Mat matrix, std::string label, int num_elements) {

	printf("Visualization : %s \n", label);
	printf("[");
	for (int i = 0; i < num_elements; i++) {
		printf("[");
		for (int j = 0; j < num_elements; j++) {
			printf("%f,", matrix.at<float>(i,j));
		}
		printf("...],\n");
	}
	printf("...]\n");

}

void visualize_whole_matrix(cv::Mat matrix, std::string label) {

	std::cout << "Visualization: " << label;
	printf("[");
	for (int i = 0; i < matrix.size().height; i++) {
		printf("[");
		for (int j = 0; j < matrix.size().width; j++) {
			printf("%f,", matrix.at<float>(i, j));
		}
		printf("...],\n");
	}
	printf("...]\n");

}

void visualize_whole_matrix_int(cv::Mat matrix, std::string label) {

	std::cout << "Visualization: " << label;
	printf("[");
	for (int i = 0; i < matrix.size().height; i++) {
		printf("[");
		for (int j = 0; j < matrix.size().width; j++) {
			printf("%d,", matrix.at<int>(i, j));
		}
		printf("...],\n");
	}
	printf("...]\n");

}