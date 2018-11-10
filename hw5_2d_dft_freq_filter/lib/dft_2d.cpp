#include <dft_2d.hpp>

Dft2d::Dft2d(cv::Mat &src_img)
{
  this->src_img_ = src_img.clone();
}

cv::Mat Dft2d::getDftImg()
{
  static cv::Mat img_out(src_img_.rows, src_img_.cols, CV_64FC1);
  for (int i = 0; i < src_img_.rows; i++) 
  {
    for (int j = 0; j < src_img_.cols; j++) 
    {
      std::complex<double> Fxy;
      Fxy = this->getDftValue(j, i);
      img_out.at<double>(i, j) = std::abs(Fxy);
      std::cout << "\rCalculating DFT " << std::setprecision(4) 
                << ((i*src_img_.rows+j)*100.0)/(double)(src_img_.rows*src_img_.cols)
                << " %  " << std::flush;
    }
  }
  std::cout << std::endl;
  return img_out;
}

cv::Mat Dft2d::getDftImg(int num_threads)
{
  static cv::Mat img_out(src_img_.rows, src_img_.cols, CV_64FC1);
  dtf_p_count_ = 0;
  std::thread post(&Dft2d::dftProgress, this);
  std::vector<std::thread> threads;
  double trunk = src_img_.rows / num_threads;
  for (int i = 0; i < num_threads; i++)
  {
    threads.push_back(std::thread(&Dft2d::dftTask, this, i*trunk, (i+1)*trunk, std::ref(img_out)));
  }
  for(int i = 0; i < threads.size() ; i++)
  {
      threads.at(i).join();
  }
  post.join();
  return img_out;
}

void Dft2d::dftTask(int min_rows, int max_rows, cv::Mat &dft_out)
{
  for (int i = min_rows; i < max_rows; i++) 
  {
    for (int j = 0; j < src_img_.cols; j++) 
    {
      dft_out.at<double>(i, j) = std::abs(this->getDftValue(j, i));
      dtf_p_count_mtx_.lock();
      dtf_p_count_ += 1;
      dtf_p_count_mtx_.unlock();
    }
  }
}

void Dft2d::dftProgress()
{
  double size_img = src_img_.rows * src_img_.cols;
  while(true)
  {
    dtf_p_count_mtx_.lock();
    double persent = (dtf_p_count_*100) / size_img;
    dtf_p_count_mtx_.unlock();
    std::cout << "\rDFT Progress: " << std::setprecision(4)  
              << persent << "%     " << std::flush;
    if (dtf_p_count_ >= size_img) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  std::cout << std::endl;
}

std::complex<double> Dft2d::getDftValue(int u, int v)
{
  std::complex<double> result(0, 0);
  const double M = src_img_.cols;
  const double N = src_img_.rows;
  const std::complex<double> i(0, 1);
  for (int y = 0; y < N; y++) 
  {
    for (int x = 0; x < M; x++) 
    {
      double fxy = src_img_.at<uint8_t>(y, x)*pow(-1, x+y);
      result += fxy*exp(-2.0*i*M_PI*(u*x/M+v*y/M));
    }
  }
  return result/(M*N);
}

IDft2d::IDft2d(cv::Mat &src_img)
{
  this->src_img_ = src_img.clone();
}

cv::Mat IDft2d::getIDftImg()
{
  static cv::Mat img_out(src_img_.rows, src_img_.cols, CV_8UC1);
  for (int i = 0; i < src_img_.rows; i++) 
  {
    for (int j = 0; j < src_img_.cols; j++) 
    {
      std::complex<double> Fxy;
      Fxy = this->getDftValue(j, i);
      img_out.at<uint8_t>(i, j) = std::abs(Fxy); 
      std::cout << "\rCalculating IDFT " << std::setprecision(4) 
                << ((i*src_img_.rows+j)*100.0)/(double)(src_img_.rows*src_img_.cols)
                << " %  " << std::flush;
    }
  }
  std::cout << std::endl;
  return img_out;
}

// void Dft2d::dftTask()
// {
//     cout << a << endl;
// }

std::complex<double> IDft2d::getDftValue(int u, int v)
{
  std::complex<double> result(0, 0);
  const double M = src_img_.cols;
  const double N = src_img_.rows;
  const std::complex<double> i(0, 1);
  for (int y = 0; y < N; y++) 
  {
    for (int x = 0; x < M; x++) 
    {
      double fxy = src_img_.at<double>(y, x)*pow(-1, x+y);
      result += fxy*exp(-2.0*i*M_PI*(u*x/M+v*y/M));
    }
  }
  return result;
}