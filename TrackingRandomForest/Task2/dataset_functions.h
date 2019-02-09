#include <string>
#include <filesystem>
#include<opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <iostream>
#include <iterator>
#include <algorithm>
#include <opencv2/ml.hpp>
#include "img_functions.h"

using namespace std;
using namespace cv;
namespace fs = boost::filesystem;

int get_size_dataset(int num_classes, std::string absolutePath);
void add_augmented_picture(HOGDescriptor descr, Mat * labels, Mat * features, Mat resizedImg, int classe);
void create_dataset(int num_classes, std::string absolutePath, Mat * labels, Mat * features, int pic_resize, int cells_size, int blocks_size, int blocks_stride, int training);
void get_max_index(cv::Mat matrix, int * index, int * value);
void visualize_vector(std::vector<string> labels, std::string label, int num_elements);
void visualize_vector(std::vector<float> values, std::string label, int num_elements);
void visualize_matrix(cv::Mat matrix, std::string label, int num_elements);
void visualize_whole_matrix(cv::Mat matrix, std::string label);
void visualize_whole_matrix_int(cv::Mat matrix, std::string label);

