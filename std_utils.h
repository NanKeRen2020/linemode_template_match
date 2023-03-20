#ifndef STD_UTILS_H
#define STD_UTILS_H

#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp> 
#include <numeric>
#include "line2Dup.h"
#include "icp.h"
#include "edge_scene.h"
#include "scene/kdtree_scene/kdtree_scene.h"


template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    for (int i = 0; i < v.size(); ++i)
    {
        os << v[i];
        if (i != v.size() - 1)
        {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
}

bool check_roi_valid(cv::Rect& roi, int width, int height);


// 按坐标值大小排序找到的轮廓
void filter_string(std::string& filter_string, const std::string& ignore_string);


// 转变为黑底白字
void change_bg_black(cv::Mat& binary_img, const cv::Rect roi = cv::Rect());

// 检测字符行或列的缺失
template<typename T1, typename T2, typename T3>
bool detect_gap(const cv::Mat& to_project, bool horizon_project, T1&& integral_width, 
                T2&& missing_thr1, T3&& missing_thr2, bool update_parameter = false, uchar col_label = 0)
{
    cv::Mat project;
    cv::reduce(to_project, project, horizon_project, cv::REDUCE_SUM, CV_32S);    
    std::vector<int> integral_project = get_local_integral(project, integral_width);
    
    bool result = false;
    // 行字符/全列字符
    if (!horizon_project || (horizon_project && col_label == 9))
    { 
        
        auto min_itr = std::min_element(integral_project.begin() + 0, integral_project.end() - 0);
        result = ( (*min_itr) < missing_thr1 || (*min_itr) > missing_thr2 );
         
        if (update_parameter && !(*min_itr) && (!horizon_project && integral_width < project.cols/3 || 
            horizon_project && integral_width < project.rows/2))
        {
            integral_width += 2;
        }
        if (update_parameter && (*min_itr) < missing_thr1 )
        {
            missing_thr1 = ((*min_itr) <= 0) ? 600 : (*min_itr)*0.8;
        }
        if (update_parameter && (*min_itr) > missing_thr2 )
        {
			
            missing_thr2 = (*min_itr);
        }
    }
    if (horizon_project && col_label != 9)
    {
        int w = (integral_project.size() + integral_width)/3;
        auto min_itr1 = std::min_element(integral_project.begin() + 0, integral_project.begin() + w);
        auto min_itr2 = std::min_element(integral_project.begin() + w, integral_project.begin() + 2*w);
        auto min_itr3 = std::min_element(integral_project.begin() + 2*w, integral_project.end() - 0);
       
        int result1 = 9;
        if ( (*min_itr1) < missing_thr1 ) result1 = result1 - 2;
        if ( (*min_itr2) < missing_thr1 ) result1 = result1 - 3;
        if ( (*min_itr3) < missing_thr1 ) result1 = result1 - 4;
        result = (col_label != result1);
    }
   
    return result;
}


// linemode模板匹配识别
class Linemode_Template_Match
{

public:

Linemode_Template_Match(const std::vector<int>& pyr_stride, const std::vector<float>& angle_range, float angle_step, 
                        const std::vector<float>& scale_range, float scale_step, const std::vector<std::string>& class_ids, 
                        const std::string& template_model_name);

Linemode_Template_Match(int feature_number, const std::vector<int>& pyr_stride, const std::vector<float>& angle_range, float angle_step, 
                        const std::vector<float>& scale_range, float scale_step, const std::vector<std::string>& class_ids, 
                        const std::string& template_model_name, int weak_thresh = 0, int strong_thresh = 80);

void load_model(const std::vector<std::string>& class_ids, const std::string& template_model_name);

void set_labels(const std::map<int, std::string> labels) { index_char = labels; }

void read_labels(const std::string& template_image_path);

bool is_loaded(const std::string& class_id) { return m_detector.isLoaded(class_id); }

bool train_model(const std::string& template_image_path, const std::string& class_id, const std::string& template_model_name);

bool train_model(const cv::Mat& template_image, const std::string& class_id, const std::string& template_model_name);

bool delete_model(const std::vector<std::string>& class_ids) { return m_detector.removeTemplate(class_ids); }

// 形状匹配定位和识别
bool recognize(const cv::Mat& to_match, const std::vector<std::string>& class_ids, int match_similarity_thresh, 
               cv::Rect& match_roi, bool pyramid_acc = true);

// 单字符形状匹配识别
std::string recognize(const cv::Mat& to_match, const std::vector<std::string>& class_ids, int match_similarity_thresh, bool pyramid_acc = false);

// 字符行形状匹配识别
std::string recognize(const cv::Mat& to_match, const std::vector<std::string>& class_ids, int match_similarity_thresh,
                      int nms_similarity_thresh, float num_iou_thresh, std::vector<cv::Rect>& match_rois, bool pyramid_acc = false, 
                      const std::vector<cv::Rect>& mask_roi = std::vector<cv::Rect>() );

private:

// 模板匹配识别标签
std::map<int, std::string> index_char
{
    
    {0, "0"}, {1, "1"}, {2, "2"}, {3, "3"}, {4, "4"}, {5, "5"}, {6, "6"}, {7, "7"},  {8, "8"}, 
    {9, "9"}, {10, "A"}, {11, "B"}, {12, "S"},  {13, "T"}, {14, "Y"} 
   
};

int m_feature_number;
std::vector<int> m_pyr_stride;
int m_weak_thresh;
int m_strong_thresh;


int m_similarity_thresh;
int m_nms_similarity_thresh;
float m_num_iou_thresh;

std::vector<float> m_angle_range;
float m_angle_step;
std::vector<float> m_scale_range;
float m_scale_step;

int m_scale_number;

line2Dup::Detector m_detector;
shape_based_matching::shapeInfo_producer m_shapes; 

};

#endif // STD_UTILS_H