#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char** argv )
{
	int width = 256;
	int height = 256;
	int size = width*height;
	char OriFile[] = "../images/lena_256.raw";    // Input Image name
	char OutFileRotFace[] = "../images/lena_256_rot_face.raw"; // Output Raw Image name
  char OutFileBright[] = "../images/lena_256_bright.raw"; // Output Raw Image name

	//-----------------------1. Read File-----------------------//
	FILE *lenaFile, *resultRotFace, *resultBright;
	lenaFile = fopen(OriFile, "rb");
	resultRotFace = fopen(OutFileRotFace, "wb");
  resultBright = fopen(OutFileBright, "wb");
	if (lenaFile == NULL) {
		puts("Loading File Error!");
		system("PAUSE");
		exit(0);
	}

	// char *lenai = new char[size];
  unsigned char lenai[width][height];

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

  // ***** slove 1.2.b.1 *****
  std::cout << "1.2.b.1: " << std::endl; 
  std::cout << "  Instensity of 168th row, 99th column: "; 
  std::cout << (unsigned int)lenaMat.at<unsigned char>(168, 99) << std::endl;

  // ***** slove 1.2.b.2 *****
  std::cout << "1.2.b.2: " << std::endl; 
  std::cout << "  Instensity of 18083th pixel: "; 
  std::cout << (unsigned int)lenaMat.at<unsigned char>(18083/width, 18083%265) << std::endl;
  
  // ***** slove 1.2.d *****
  cv::Mat lenaMat_rotated = cv::Mat(width, height, CV_8UC1);
  for (int i = 0; i < lenaMat.rows; i++)
  {
    for (int j = 0; j < lenaMat.cols; j++)
    {
      lenaMat_rotated.at<char>(lenaMat_rotated.rows-1-j, lenaMat_rotated.cols-1-i) = lenaMat.at<char>(i, j);
    }
  }

  // ***** slove 1.2.e *****
  cv::Mat lenaMat_rotated_face = cv::Mat(height, width, CV_8UC1);
  int size_face_h = height/2;
  int size_face_w = width/2;
  cv::Mat lenaMat_face = cv::Mat(size_face_h, size_face_w, CV_8UC1);
  for (int i = 0; i < lenaMat.rows; i++)
  {
    for (int j = 0; j < lenaMat.cols; j++)
    {
      lenaMat_rotated_face.at<char>(lenaMat_rotated.rows-1-j, i) = lenaMat.at<char>(i, j);
    }
  }
  // get rotated face ROI
  int roi_origin_x = lenaMat_rotated.cols/2 - size_face_w/2;
  int roi_origin_y = lenaMat_rotated.rows/2 - size_face_h/2;
  for (int i = 0; i < lenaMat_face.rows; i++)
  {
    for (int j = 0; j < lenaMat_face.cols; j++)
    {
      lenaMat_face.at<char>(j, lenaMat_face.cols - 1 - i) = lenaMat.at<char>(roi_origin_y + i, roi_origin_x + j);
    }
  }
  // paste rotated face ROI to rotated image
  unsigned char lena_face_rot_raw[lenaMat_rotated_face.cols][lenaMat_rotated_face.rows];
  for (int i = roi_origin_y; i < roi_origin_y + lenaMat_face.rows; i++)
  {
    for (int j = roi_origin_x; j < roi_origin_x + lenaMat_face.cols; j++)
    {
      int face_x = j - roi_origin_x;
      int face_y = i - roi_origin_y;
      lenaMat_rotated_face.at<char>(i, j) = lenaMat_face.at<char>(face_y, face_x);
    }
  }

  // ***** slove 1.2.f *****
  // mat to fwrite array
  for (int i = 0; i < lenaMat.rows; i++)
  {
    for (int j = 0; j < lenaMat.cols; j++)
    {
      lena_face_rot_raw[i][j] = lenaMat_rotated_face.at<char>(i, j);
    }
  }

  // ***** slove 1.3.a *****
  unsigned char lena_bright_raw[width][height];
  for (int i = 0; i < width; i++)
  {
    for (int j = 0; j < height; j++)
    {
      lena_bright_raw[i][j] = lenai[i][j] + 99;
    }
  }

	// -----------------------3. Create window and show your Image-----------------------//
  // show 1.2.c
	cv::namedWindow("lenaFile", 0);
	cv::resizeWindow("lenaFile", 256, 256);
	cv::moveWindow("lenaFile", 100, 100);
	cv::imshow("lenaFile", lenaMat);//display Image

  // show 1.2.d	
  cv::namedWindow("lenaRotated", 0);
	cv::resizeWindow("lenaRotated", 256, 256);
	cv::moveWindow("lenaRotated", 100, 100);
	cv::imshow("lenaRotated", lenaMat_rotated);//display Image	

  // show 1.2.e
  cv::namedWindow("lenaRotatedFace", 0);
	cv::resizeWindow("lenaRotatedFace", 256, 256);
	cv::moveWindow("lenaRotatedFace", 100, 100);
	cv::imshow("lenaRotatedFace", lenaMat_rotated_face);//display Image

	// -----------------------4. Save Image as raw format-----------------------//

  // Save 1.2.f
  fwrite(lena_face_rot_raw, 1, size, resultRotFace);

  // Save 1.3.a
  fwrite(lena_bright_raw, 1, size, resultBright);

	// -----------------------5. Release memory-----------------------//
	fclose(lenaFile);
	fclose(resultRotFace);
  fclose(resultBright);
  cv::waitKey(0);
	cv::destroyWindow("lenaFile");
  cv::destroyWindow("lenaRotated");
  cv::destroyWindow("lenaRotatedFace");
  delete lenai;
  delete lena_face_rot_raw;
  delete lena_bright_raw;
  lenaMat.release();
  lenaMat_rotated.release();
  lenaMat_rotated_face.release();
  return 0;
}