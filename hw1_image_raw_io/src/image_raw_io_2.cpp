#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char** argv )
{
	int width = 512;
	int height = 512;
	int size = width*height;
	char OriFile[] = "../images/lena_512.raw";    // Input Image name
  char OutFile[] = "../images/lena_512_draw.png";    // Output Image name

  FILE *lenaFile;
  lenaFile = fopen(OriFile, "rb");
	if (lenaFile == NULL) {
		puts("Loading File Error!");
		system("PAUSE");
		exit(0);
	}
  unsigned char lenai[width][height];
  fread(lenai, sizeof(char), size, lenaFile);
  cv::Mat lenaMat = cv::Mat(height, width, CV_8UC1);
  for (int i = 0; i < lenaMat.rows; i++)
  {
    for (int j = 0; j < lenaMat.cols; j++)
    {
      lenaMat.at<char>(i, j) = lenai[i][j];
    }
  }
  // Draw Point and ellipse
  cv::Point draw_pt =  cv::Point(width/2, height/2);
  for (int i = 2; i < 15; i = i + 3)
  {
    cv::ellipse(lenaMat, cv::Point(width/2, height/2),
                         cv::Size( width/i, width/(2 * i) ), 
                         45 + 20 *i,
                         0,
                         360,
                         cv::Scalar( 255, 0, 0 ),
                         3);
  }
  cv::putText(lenaMat, "106368002", cvPoint(50,400), 
              cv::FONT_HERSHEY_PLAIN, 5, 
              cvScalar(255,0,0), 3, CV_AA);
  

  cv::imwrite( OutFile, lenaMat );

  // Show result
  cv::namedWindow("lenaFile", 0);
	cv::resizeWindow("lenaFile", width, height);
	cv::moveWindow("lenaFile", 100, 100);
	cv::imshow("lenaFile", lenaMat);//display Image

  fclose(lenaFile);
  cv::waitKey(0);
  delete lenai;
  lenaMat.release();
  return 0;
}