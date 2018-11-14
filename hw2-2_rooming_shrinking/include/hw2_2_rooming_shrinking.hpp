#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

const std::string SAVE_IMG_FOLDER = "../result_img/";

void loadRawFile(cv::Mat &dst_img, std::string file_path, int width, int height);
void showImage(std::string win_name, cv::Mat &show_img);
void rowColReplication(cv::Mat &src_img, cv::Mat &dst_img);
void rowColDeletion(cv::Mat &src_img, cv::Mat &dst_img);
void gaussionBlur(cv::Mat &src_img, cv::Mat &dst_img, int kernel_size);
void nearestNeighboring(cv::Mat &src_img, cv::Mat &dst_img);
void bilinearInterpolation(cv::Mat &src_img, cv::Mat &dst_img);
void saveImage(cv::Mat &img, std::string prefix);
double getMSE(cv::Mat &src, cv::Mat &target);
double getPSNR(double mse, int num_bits);
