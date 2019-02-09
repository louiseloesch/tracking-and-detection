#include<opencv2/opencv.hpp>

using namespace cv;

#define UNDEFINED 0
#define SQUARE 1
#define SQUARE_DEFINED 2
#define SMART_RESIZING 3
#define SQUARE_RESIZED 4

// Finding next multiple of "multiple" above "number"
int findNextMultiple(int number, int multiple);

// Resizing picture : multiple modes available
// The one we usually take : SQUARE_RESIZED (padding to a square and then resize all the dataset to the same size newSize)
void paddingToNextShape(Mat src, Mat &dst, int multiple, int mode, Size &newSize);