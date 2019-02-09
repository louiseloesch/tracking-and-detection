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

	if (mode == UNDEFINED) {

		newRows = findNextMultiple(rows, multiple);
		newCols = findNextMultiple(cols, multiple);

		newSize = cv::Size(newCols, newRows);

		int borderTop = (int)(newSize.height - rows) / 2;
		int borderLeft = (int)(newSize.width - cols) / 2;
		int borderBottom = newSize.height - rows - borderTop;
		int borderRight = newSize.width - cols - borderLeft;

		copyMakeBorder(src, dst, borderTop, borderBottom, borderLeft, borderRight, BORDER_REPLICATE);

	}
	else if (mode == SQUARE) {
		// just resize to the samllest size multiple of "multiple". 


		int maxRowsCols;

		if (rows > cols) {
			maxRowsCols = rows;
		}
		else {
			maxRowsCols = cols;
		}

		newRows = findNextMultiple(maxRowsCols, multiple);
		newCols = newRows;

		newSize = cv::Size(newCols, newRows);

		int borderTop = (int)(newSize.height - rows) / 2;
		int borderLeft = (int)(newSize.width - cols) / 2;
		int borderBottom = newSize.height - rows - borderTop;
		int borderRight = newSize.width - cols - borderLeft;

		copyMakeBorder(src, dst, borderTop, borderBottom, borderLeft, borderRight, BORDER_REPLICATE);

	}
	else if (mode == SQUARE_DEFINED) { // valid only if size entered is upper than the one of pictures.
		// directly resize the picture to a square (valid only if the size requested is higher than the size of the pictures from the dataset).

		newRows = newSize.height;
		newCols = newSize.width;

		newSize = cv::Size(newCols, newRows);

		int borderTop = (int)(newSize.height - rows) / 2;
		int borderLeft = (int)(newSize.width - cols) / 2;
		int borderBottom = newSize.height - rows - borderTop;
		int borderRight = newSize.width - cols - borderLeft;

		copyMakeBorder(src, dst, borderTop, borderBottom, borderLeft, borderRight, BORDER_REPLICATE);

	}
	else if (mode == SMART_RESIZING) {
		// resize such that the temporary image upper dimension is 128 and then padding

		newRows = newSize.height;
		newCols = newSize.width;

		float factorRows = (float) rows / newRows;
		float factorCols = (float) cols / newCols;
		int newColsTmp, newRowsTmp;
		Size sizeTmp;
		Mat imgTmp;
		if (factorRows >= 1 || factorCols >= 1) {

			if (factorRows < factorCols) {
				newColsTmp = newSize.width;
				newRowsTmp = (int)((float)rows / factorCols);
				sizeTmp = Size(newColsTmp, newRowsTmp);
				resize(src, imgTmp, sizeTmp);
			}
			else {
				newRowsTmp = newSize.height;
				newColsTmp = (int)((float)cols / factorRows);
				sizeTmp = Size(newColsTmp, newRowsTmp);
				resize(src, imgTmp, sizeTmp);
			}

		}
		else {
			newRowsTmp = rows;
			newColsTmp = cols;
			sizeTmp = Size(newColsTmp, newRowsTmp);
			imgTmp = src.clone();
		}
		
		newSize = cv::Size(newCols, newRows);

		int borderTop = (int)(newSize.height - newRowsTmp) / 2;
		int borderLeft = (int)(newSize.width - newColsTmp) / 2;
		int borderBottom = newSize.height - newRowsTmp - borderTop;
		int borderRight = newSize.width - newColsTmp - borderLeft;

		copyMakeBorder(imgTmp, dst, borderTop, borderBottom, borderLeft, borderRight, BORDER_REPLICATE);

	}
	else if (mode == SQUARE_RESIZED) {
		// convert to square and then to desired size

		int newColsTmp, newRowsTmp;
		Size sizeTmp;
		Mat imgTmp;

		if (rows > cols) {
			newColsTmp = rows;
			newRowsTmp = rows;
		}
		else {
			newColsTmp = cols;
			newRowsTmp = cols;
		}

		int borderTop = (int)(newRowsTmp - rows) / 2;
		int borderLeft = (int)(newColsTmp - cols) / 2;
		int borderBottom = newRowsTmp - rows - borderTop;
		int borderRight = newColsTmp - cols - borderLeft;

		copyMakeBorder(src, imgTmp, borderTop, borderBottom, borderLeft, borderRight, BORDER_REPLICATE);

		newRows = newSize.height;
		newCols = newSize.width;
		newSize = cv::Size(newCols, newRows);

		resize(imgTmp, dst, newSize);

	}

	else {
		printf("Undefined usage \n");
		return;
	}

}