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

// linemode模板匹配识别
class Linemode_Template_Match
{

public:

Linemode_Template_Match(const std::vector<int>& pyr_stride, const std::vector<float>& angle_range, float angle_step, 
                        const std::vector<float>& scale_range, float scale_step, const std::string& model_path, 
                        const std::vector<std::string>& class_ids, const std::string& template_model_name);

Linemode_Template_Match(int feature_number, const std::vector<int>& pyr_stride, const std::vector<float>& angle_range, float angle_step, 
                        const std::vector<float>& scale_range, float scale_step, const std::string& model_path, const std::vector<std::string>& class_ids, 
                        const std::string& template_model_name, int weak_thresh = 0, int strong_thresh = 80);

void load_model(const std::string& model_path, const std::vector<std::string>& class_ids, const std::string& template_model_name);

void set_labels(const std::map<int, std::string> labels) { index_char = labels; }

void read_labels(const std::string& template_image_path);

bool is_loaded(const std::string& class_id) { return m_detector.isLoaded(class_id); }

bool train_model(const std::string& template_image_path, const std::string& model_path, const std::string& class_id, const std::string& template_model_name);

bool train_model(const cv::Mat& template_image, const std::string& model_path, const std::string& class_id, const std::string& template_model_name);

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
