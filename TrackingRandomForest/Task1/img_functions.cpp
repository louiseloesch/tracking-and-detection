#include "img_functions.h"

int findNextMultiple(int number, int multiple) {

	if (number % multiple == 0) {
		return number;
	}
	else {
		return ((number / multiple) + 1) * multiple;
	}

}

void paddingToNextShape(cv::Mat src, cv::Mat &dst, int multiple, int mode, Size &newSize) {
	
	int newCols, newRows, rows, cols;
	Size s;

	s = src.size();
	rows = s.height;
	cols = s.width;

	if (mode == UNDEFINED) { // padding to next multiple pixels number of "multiple" each dimension independently

		newRows = findNextMultiple(rows, multiple);
		newCols = findNextMultiple(cols, multiple);

	}
	else if (mode == SQUARE) { // padding to a square and then resize

		int maxRowsCols;

		if (rows > cols) {
			maxRowsCols = rows;
		}
		else {
			maxRowsCols = cols;
		}

		newRows = findNextMultiple(maxRowsCols, multiple);
		newCols = newRows;

	} // padding to a square and next resize
	else {
		printf("Undefined usage \n");
		return;
	}

	newSize = cv::Size(newCols, newRows);

	// computing padding parameters
	int borderTop = (int)(newSize.height - rows) / 2;
	int borderLeft = (int)(newSize.width - cols) / 2;
	int borderBottom = newSize.height - rows - borderTop;
	int borderRight = newSize.width - cols - borderLeft;

	copyMakeBorder(src, dst, borderTop, borderBottom, borderLeft, borderRight, BORDER_REPLICATE); // performs the padding

}