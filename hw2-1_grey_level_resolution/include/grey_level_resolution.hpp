#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

void loadRawFile(cv::Mat &dst_img, std::string file_path, int width, int height);
cv::Mat getQuantizeImage(cv::Mat &src, int num_bit);
void showImage(std::string win_name, cv::Mat &show_img);
void showAllImages(std::vector<cv::Mat> &list, std::string prefix);