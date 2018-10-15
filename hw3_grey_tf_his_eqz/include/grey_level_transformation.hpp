#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

const std::string SAVE_IMG_FOLDER = "../result_img/";
const double GAMMAS[] = {0.04, 0.1, 0.2, 0.4, 0.67, 1.0, 1.5, 2.5, 5.0, 10.0, 25.0};

void loadRawFile(cv::Mat &dst_img, std::string file_path, int width, int height);
void showImage(std::string win_name, cv::Mat &show_img);
void saveImage(cv::Mat &img, std::string folder, std::string file_name);
double powerLaw(double L, double c, double r, double gamma);
void PowerLawTransformation(cv::Mat &src, cv::Mat &dst, double gamma);
void showAllImages(std::vector<cv::Mat> &list, std::string prefix);
void saveAllImages(std::vector<cv::Mat> &list, std::string floder, std::string prefix);
void plotCurves(cv::Mat &plot, std::vector<std::vector<cv::Point2f> > curvesPoints);
double linearFunc(uint8_t value, cv::Point2f r1s1, cv::Point2f r2s2, double L);
void piecewiseLinearTF(cv::Mat &src_img, 
                       cv::Mat &dst_img, 
                       cv::Point2f r1s1, 
                       cv::Point2f r2s2);
void writeCSV( std::string folder, 
               std::string file_name,  
               std::vector<std::vector<cv::Point2f> > curvesPoints);
void writeCSV( std::string folder, 
               std::string file_name,  
               std::vector<cv::Point2f> curvesPoints);