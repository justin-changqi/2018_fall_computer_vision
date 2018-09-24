#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char** argv )
{
	int width = 256;
	int height = 256;
	int size = width*height;
	char OriFile[] = "../images/lena_256.raw";    //Input Image name
	char OutFile[] = "../images/lena_256out.raw"; //Output Raw Image name

	//-----------------------1. Read File-----------------------//
	FILE *lenaFile, *resultFile;
	lenaFile = fopen(OriFile, "rb");
	resultFile = fopen(OutFile, "wb");
	if (lenaFile == NULL) {
		puts("Loading File Error!");
		system("PAUSE");
		exit(0);
	}

	// char *lenai = new char[size];
  char lenai[width][height];

	fread(lenai, sizeof(char), size, lenaFile);

	// -----------------------2. Transfer data type to OpenCV-----------------------//
	// Mat type
	cv::Mat lenaMat = cv::Mat(height, width, CV_8UC1);
  for (int i = 0; i < lenaMat.rows; i++)
  {
    for (int j = 0; j < lenaMat.cols; j++)
    {
      lenaMat.at<char>(i, j) = lenai[i][j];
    }
  }

	// -----------------------3. Create window and show your Image-----------------------//
	cv::namedWindow("lenaFile", 0);
	cv::resizeWindow("lenaFile", 256, 256);
	cv::moveWindow("lenaFile", 100, 100);
	cv::imshow("lenaFile", lenaMat);//display Image	
	cv::waitKey(0);

	// -----------------------4. Save Image as raw format-----------------------//
	fwrite(lenai, 1, size, resultFile);

	// -----------------------5. Release memory-----------------------//
	fclose(lenaFile);
	fclose(resultFile);
	cv::destroyWindow("lenaFile");
  lenaMat.release();
  return 0;
}