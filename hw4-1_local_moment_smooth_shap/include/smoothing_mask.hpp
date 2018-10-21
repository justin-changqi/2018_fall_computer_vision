#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

void loadRawFile(cv::Mat &dst_img, std::string file_path, int width, int height);
void showImage(std::string win_name, cv::Mat &show_img);
void saveImage(cv::Mat &img, std::string folder, std::string file_name);
cv::Mat averagingFilter(cv::Mat &src, cv::Mat &mask);
void setBoxMask(cv::Mat &mask);
cv::Mat medianFilter(cv::Mat &src, cv::Mat &mask);
cv::Mat thresholding(cv::Mat &src, double th);
cv::Mat applyMasks(cv::Mat &src, std::vector<cv::Mat> &masks);