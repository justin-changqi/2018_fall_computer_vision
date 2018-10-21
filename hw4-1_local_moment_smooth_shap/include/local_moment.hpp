#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

void loadRawFile(cv::Mat &dst_img, std::string file_path, int width, int height);
void showImage(std::string win_name, cv::Mat &show_img);
void saveImage(cv::Mat &img, std::string folder, std::string file_name);

class LocalMoment
{
  public:
    int mask_size[2];
    cv::Mat src_img;
    cv::Mat pad_img;
    int pad_x;
    int pad_y;
    LocalMoment(cv::Mat &src_img, int mask_size[2] );
    cv::Mat addPadding(cv::Mat &src_img,  int mask_size[2]);
    std::vector<cv::Mat>  getLocalMomtEnh(double E, double k0, double k1, double k2);
    cv::Mat getLocalMoment();
    void getImageMeanSd(cv::Mat &src_img, double *mean_sd);
};