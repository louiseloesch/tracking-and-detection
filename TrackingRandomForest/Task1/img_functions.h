#include<opencv2/opencv.hpp>

using namespace cv;

#define UNDEFINED 0
#define SQUARE 1

// Finding the next multiple of "multiple" that is higher than "number"
int findNextMultiple(int number, int multiple);

// Performing the padding : multiple mode possible for resizing.
void paddingToNextShape(Mat src, Mat &dst, int multiple, int mode, Size &newSize);