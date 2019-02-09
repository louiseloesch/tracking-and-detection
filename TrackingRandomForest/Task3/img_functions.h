#include<opencv2/opencv.hpp>

using namespace cv;

#define UNDEFINED 0
#define SQUARE 1
#define SQUARE_DEFINED 2
#define SMART_RESIZING 3
#define SQUARE_RESIZED 4

int findNextMultiple(int number, int multiple);
void paddingToNextShape(Mat src, Mat &dst, int multiple, int mode, Size &newSize);