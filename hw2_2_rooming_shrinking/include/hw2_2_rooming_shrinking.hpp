#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

void loadRawFile(cv::Mat &dst_img, std::string file_path, int width, int height);
void showImage(std::string win_name, cv::Mat &show_img);
void rowColReplication(cv::Mat &src_img, cv::Mat &dst_img);
void rowColDeletion(cv::Mat &src_img, cv::Mat &dst_img);
void gaussionBlur(cv::Mat &src_img, cv::Mat &dst_img, int kernel_size);
void nearestNeighboring(cv::Mat &src_img, cv::Mat &dst_img);
void bilinearInterpolation(cv::Mat &src_img, cv::Mat &dst_img);