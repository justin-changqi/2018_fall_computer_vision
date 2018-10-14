#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

const std::string SAVE_IMG_FOLDER = "../result_img/";
const double GAMMAS[] = {0.04, 0.1, 0.2, 0.4, 0.67, 1.0, 1.5, 2.5, 5.0, 10.0, 25.0};

void loadRawFile(cv::Mat &dst_img, std::string file_path, int width, int height);
void showImage(std::string win_name, cv::Mat &show_img);
void saveImage(cv::Mat &img, std::string prefix);
void PowerLawTransformation(cv::Mat &src, cv::Mat &dst, double gamma);
void showAllImages(std::vector<cv::Mat> &list, std::string prefix);
void saveAllImages(std::vector<cv::Mat> &list, std::string floder, std::string prefix);