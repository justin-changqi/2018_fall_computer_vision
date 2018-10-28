#include <filters.hpp>

Filter::Filter (cv::Mat &src_img, Method method)
{
  this->method = method;
  this->setMask();
  this->src_img = src_img.clone();
  cv::Mat mask = this->masks[0];
  this->addPadding(this->src_img, mask); 
}

cv::Mat Filter::setMask()
{
  switch(this->method)
  {
    case Method::LAPLACIAN:
    {
      cv::Mat mask = (cv::Mat_<int>(3,3) << -1, -1, -1, -1, 8, -1, -1, -1, -1);
      this->masks.push_back(mask);
      break;
    }
    case Method::SOBEL:
    {
      cv::Mat mask_x = (cv::Mat_<int>(3,3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
      cv::Mat mask_y = (cv::Mat_<int>(3,3) << -1, 2, 1, 0, 0, 0, -1, -2, -1);
      this->masks.push_back(mask_x);
      this->masks.push_back(mask_y);
      break;
    } 
  }
}

cv::Mat Filter::applyMask(cv::Mat &src_img, cv::Mat &mask)
{
  cv::Mat out_img(src_img.rows, src_img.cols, CV_32SC1);
  cv::Mat padded_img = this->pad_img;
  for (int i = 0; i < src_img.rows; i++)
  {
    for (int j = 0; j < src_img.cols; j++)
    {
      // cv::Mat op_make(mask.rows, mask.cols, CV_32SC1);
      int px = 0;
      for (int k = 0; k < mask.rows; k++)
      {
        for (int l = 0; l < mask.cols; l++)
        {
          int offset_y = i + k;
          int offset_x = j + l;
          // std::cout << "(" << offset_x << ", " << offset_y << ")" << std::endl;
          // op_make.at<int>(k, l) = int(padded_img.at<uint8_t>(offset_y, offset_x) * 
          //                             mask.at<int>(k, l));
          px += padded_img.at<uint8_t>(offset_y, offset_x) * mask.at<int>(k, l);
        }
      }
      out_img.at<int>(i, j) = abs(px);
      // std::cout << op_make << std::endl;
      // std::cout << this->det(op_make) << std::endl;
    }
  }
  return out_img;
}

void Filter::addPadding(cv::Mat &src_img, cv::Mat &mask)
{
  int pad_x = (mask.cols-1)/2;
  int pad_y = (mask.rows-1)/2;
  cv::Mat out_img(src_img.rows+2*pad_y, 
                  src_img.cols+2*pad_x, CV_8UC1);
  // put src to new image
  for (int i = 0; i < src_img.rows; i++)
  {
    for (int j = 0; j < src_img.cols; j++)
    {
      out_img.at<char>(i+pad_y, j+pad_x) = src_img.at<char>(i, j);
    }
  }
  // mirror rows
  for (int i = 0; i <= pad_y; i++)
  {
    out_img.row(pad_y+i).copyTo(out_img.row(pad_y-i));
    out_img.row(out_img.rows-1-pad_y-i).copyTo(out_img.row(out_img.rows-1-pad_y+i));
  }
  // mirror cols
  for (int i = 0; i <= pad_x; i++)
  {
    out_img.col(pad_x+i).copyTo(out_img.col(pad_x-i));
    out_img.col(out_img.cols-1-pad_x-i).copyTo(out_img.col(out_img.cols-1-pad_x+i));
  }
  this->pad_img = out_img.clone();
}

cv::Mat Filter::getFilteredImg()
{
  cv::Mat out_img(this->src_img.rows, this->src_img.cols, CV_8UC1);
  std::vector<cv::Mat> imgs_masked;
  for (int i = 0; i < this->masks.size(); i++)
  {
    cv::Mat w_img = this->applyMask(this->src_img, this->masks[i]);
    imgs_masked.push_back(w_img);
  }
  if (imgs_masked.size() == 1)
  {
    return this->Normalize(imgs_masked[0]);
  } 
  else
  {
    return imgs_masked[0];
  }
}

cv::Mat Filter::sumImgs(std::vector<cv::Mat> img_list)
{ 
  for (int i = 0; i < this->src_img.rows; i++)
  {
    for (int j = 0; j < this->src_img.cols; j++)
    {
    }
  }
}

int Filter::det(int n, cv::Mat &mat)
{
  // std::cout << n << std::endl;
  
  int c, subi, i, j, subj;
  cv::Mat submat(n, n, CV_32SC1);
  
  if (n == 2) 
  {   
    return( (mat.at<int>(0,0) * mat.at<int>(1,1)) - 
            (mat.at<int>(1,0) * mat.at<int>(0,1)));
  }
  else
  {  
    for(c = 0; c < n; c++)
    {  
      subi = 0;  
      for(i = 1; i < n; i++)
      {  
        subj = 0;
        for(j = 0; j < n; j++)
        {    
          if (j == c)
          {
            continue;
          }
          submat.at<int>(subi, subj) = mat.at<int>(i,j);
          subj++;
        }
        subi++;
      }
      this->det_d = this->det_d + (pow(-1 ,c) * mat.at<int>(0,c) * 
                    this->det(n - 1 ,submat));
    }
  }
  return this->det_d;
}

int Filter::det(cv::Mat &mat)
{
  return mat.at<int>(0,0)*mat.at<int>(1,1)*mat.at<int>(2,2)+
         mat.at<int>(0,1)*mat.at<int>(1,2)*mat.at<int>(2,0)+
         mat.at<int>(0,2)*mat.at<int>(1,0)*mat.at<int>(2,1)-
         mat.at<int>(0,2)*mat.at<int>(1,1)*mat.at<int>(2,0)-
         mat.at<int>(0,0)*mat.at<int>(1,2)*mat.at<int>(2,1)-
         mat.at<int>(0,1)*mat.at<int>(1,0)*mat.at<int>(2,2);
}

cv::Mat Filter::Normalize(cv::Mat &src)
{
  cv::Mat nor_mat(src.rows, src.cols, CV_8UC1);
  std::vector<int> px_list;
  for (int i = 0; i < src.rows; i++)
  {
    for (int j = 0; j < src.cols; j++)
    {
      px_list.push_back(src.at<int>(i, j));
    }
  }
  int max_px = *max_element(px_list.begin(), px_list.end());
  int min_px = *min_element(px_list.begin(), px_list.end());
  // std::cout<<"Max value: "<< max_px << std::endl;
  // std::cout<<"Min value: "<< min_px << std::endl;
  double scale = 255.0 / (max_px - min_px);
  // std::cout << scale << std::endl;
  for (int i = 0; i < src.rows; i++)
  {
    for (int j = 0; j < src.cols; j++)
    {
       nor_mat.at<uint8_t>(i, j) = src.at<int>(i, j) * scale;
    }
  }
  // std::cout << nor_mat << std::endl;
  return nor_mat;
}