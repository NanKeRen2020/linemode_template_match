
#include <numeric>
#include <vector>
#include <string>

#include <unistd.h>
#include <stdio.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <dirent.h>

#include "std_utils.h"


bool check_roi_valid(cv::Rect& roi, int width, int height)
{

    if (roi.x < 0)
    {
        roi.width += roi.x;
        roi.x = 0;
    }
    if (roi.y < 0)
    {
        roi.height += roi.y;
        roi.y = 0;
    }
    if (roi.x + roi.width > width)
    {
        roi.width = width - roi.x - 1;
    }
    if (roi.y + roi.height > height)
    {
        roi.height = height - roi.y - 1;
    }
    
    return (roi.width > 0) && (roi.height > 0);
}

float reblur(const unsigned char *data, int width, int height)
{
    float blur_val = 0.0f;
    float kernel[9] = { 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f, 1.0f / 9.0f };
    float *BVer = new float[width * height];//垂直方向低通滤波后的结果
    float *BHor = new float[width * height];//水平方向低通滤波后的结果

    float filter_data = 0.0;
    for (int i = 0; i < height; ++i)//均值滤波
    {
        for (int j = 0; j < width; ++j)
        {
            if (i < 4 || i > height - 5)
            {//处理边界 直接赋值原数据
                BVer[i * width + j] = data[i * width + j];
            }
            else
            {
                filter_data = kernel[0] * data[(i - 4) * width + j] + kernel[1] * data[(i - 3) * width + j] + kernel[2] * data[(i - 2) * width + j] +
                    kernel[3] * data[(i - 1) * width + j] + kernel[4] * data[(i)* width + j] + kernel[5] * data[(i + 1) * width + j] +
                    kernel[6] * data[(i + 2) * width + j] + kernel[7] * data[(i + 3) * width + j] + kernel[8] * data[(i + 4) * width + j];
                BVer[i * width + j] = filter_data;
            }

            if (j < 4 || j > width - 5)
            {
                BHor[i * width + j] = data[i * width + j];
            }
            else
            {
                filter_data = kernel[0] * data[i * width + (j - 4)] + kernel[1] * data[i * width + (j - 3)] + kernel[2] * data[i * width + (j - 2)] +
                    kernel[3] * data[i * width + (j - 1)] + kernel[4] * data[i * width + j] + kernel[5] * data[i * width + (j + 1)] +
                    kernel[6] * data[i * width + (j + 2)] + kernel[7] * data[i * width + (j + 3)] + kernel[8] * data[i * width + (j + 4)];
                BHor[i * width + j] = filter_data;
            }

        }
    }

    float D_Fver = 0.0;
    float D_FHor = 0.0;
    float D_BVer = 0.0;
    float D_BHor = 0.0;
    float s_FVer = 0.0;//原始图像数据的垂直差分总和 对应论文中的 s_Fver
    float s_FHor = 0.0;//原始图像数据的水平差分总和 对应论文中的 s_Fhor
    float s_Vver = 0.0;//模糊图像数据的垂直差分总和 s_Vver
    float s_VHor = 0.0;//模糊图像数据的水平差分总和 s_VHor
    for (int i = 1; i < height; ++i)
    {
        for (int j = 1; j < width; ++j)
        {
            D_Fver = std::abs((float)data[i * width + j] - (float)data[(i - 1) * width + j]);
            s_FVer += D_Fver;
            D_BVer = std::abs((float)BVer[i * width + j] - (float)BVer[(i - 1) * width + j]);
            s_Vver += std::max((float)0.0, D_Fver - D_BVer);

            D_FHor = std::abs((float)data[i * width + j] - (float)data[i * width + (j - 1)]);
            s_FHor += D_FHor;
            D_BHor = std::abs((float)BHor[i * width + j] - (float)BHor[i * width + (j - 1)]);
            s_VHor += std::max((float)0.0, D_FHor - D_BHor);
        }
    }
    float b_FVer = (s_FVer - s_Vver) / s_FVer;
    float b_FHor = (s_FHor - s_VHor) / s_FHor;
    blur_val = std::max(b_FVer, b_FHor);

    delete[] BVer;
    delete[] BHor;

    return blur_val;
}

float grid_max_reblur(const cv::Mat &img, int rows, int cols)
{
    int row_height = img.rows / rows;
    int col_width = img.cols / cols;
    float blur_val = FLT_MIN;
    for (int y = 0; y < rows; ++y)
    {
        for (int x = 0; x < cols; ++x)
        {
            cv::Mat img_roi = img(cv::Rect(x * col_width, y * row_height, col_width, row_height));
            auto this_grad_blur_val = reblur(img_roi.data, img_roi.cols, img_roi.rows);
            if (this_grad_blur_val > blur_val) blur_val = this_grad_blur_val;
        }
    }
    return std::max<float>(blur_val, 0);
}


float clarity_estimate(const cv::Mat& image)
{

    // float blur_val = ReBlur(src_data.data(), src_data.width(), src_data.height());
    float blur_val = grid_max_reblur(image, 2, 2);
    float clarity = 1.0f - blur_val;

    float T1 = 0.0f;
    float T2 = 1.0f;
    if (clarity <= T1)
    {
        clarity = 0.0;
    }
    else if (clarity >= T2)
    {
        clarity = 1.0;
    }
    else
    {
        clarity = (clarity - T1) / (T2 - T1);
    }

    return clarity*1000;
}


//----------------------------------------------------------------- limode match --------------------------------------------

Linemode_Template_Match::Linemode_Template_Match(const std::vector<int>& pyr_stride, const std::vector<float>& angle_range, float angle_step, 
                    const std::vector<float>& scale_range, float scale_step, const std::string& model_path, const std::vector<std::string>& class_ids, const std::string& template_model_name): 
                    m_pyr_stride(pyr_stride), m_angle_range(angle_range), m_angle_step(angle_step), m_scale_range(scale_range), m_scale_step(scale_step),
                    m_weak_thresh(0), m_strong_thresh(80), m_detector(line2Dup::Detector(64, pyr_stride, 0, 80))
{
    if (!class_ids.empty() && !template_model_name.empty())
    m_detector.readClasses(model_path, class_ids, template_model_name);

    m_shapes.angle_range = m_angle_range;
    m_shapes.angle_step = m_angle_step;
    m_shapes.scale_range = m_scale_range;
    m_shapes.scale_step = m_scale_step;  
    m_scale_number = m_shapes.produce_infos();

    // 多尺度加速分割顶层和底层模板
    m_detector.split_class_templates(class_ids, m_scale_number, m_shapes.infos.size()/m_scale_number);
    //m_detector.set_produce_dxy = true; 

}

Linemode_Template_Match::Linemode_Template_Match(int feature_number, const std::vector<int>& pyr_stride, const std::vector<float>& angle_range, float angle_step, 
                     const std::vector<float>& scale_range, float scale_step, const std::string& model_path, const std::vector<std::string>& class_ids, const std::string& template_model_name,
                     int weak_thresh, int strong_thresh ): m_pyr_stride(pyr_stride), m_angle_range(angle_range), m_angle_step(angle_step),
                     m_scale_range(scale_range), m_scale_step(scale_step), m_strong_thresh(strong_thresh), m_weak_thresh(weak_thresh), 
                     m_feature_number(feature_number), m_detector(line2Dup::Detector(feature_number, pyr_stride, weak_thresh, strong_thresh)) 
                     
{  
    if (!class_ids.empty() && !template_model_name.empty())
    m_detector.readClasses(model_path, class_ids, template_model_name);

    m_shapes.angle_range = m_angle_range;
    m_shapes.angle_step = m_angle_step;
    m_shapes.scale_range = m_scale_range;
    m_shapes.scale_step = m_scale_step;  
    m_scale_number = m_shapes.produce_infos();

    // 多尺度加速分割顶层和底层模板
    m_detector.split_class_templates(class_ids, m_scale_number, m_shapes.infos.size()/m_scale_number);
}

void Linemode_Template_Match::load_model(const std::string& model_path, const std::vector<std::string>& class_ids, const std::string& template_model_name)
{
    if (!class_ids.empty() && !template_model_name.empty())
    m_detector.readClasses(model_path, class_ids, template_model_name);

    // 多尺度加速分割顶层和底层模板
    m_detector.split_class_templates(class_ids, m_scale_number, m_shapes.infos.size()/m_scale_number);
}


bool Linemode_Template_Match::train_model(const cv::Mat& template_image, const std::string& model_path, const std::string& class_id, const std::string& template_model_name)
{
 
    delete_model({class_id});
    std::vector<shape_based_matching::shapeInfo_producer::Info> infos_have_templ;
    cv::Mat train_img = template_image;

    cv::Mat show_image = template_image.clone();
    if (show_image.channels() == 1)
    cv::cvtColor(show_image, show_image, cv::COLOR_GRAY2BGR); 

    if (train_img.channels() == 3)
    cv::cvtColor(template_image, train_img, cv::COLOR_BGR2GRAY); 

    m_shapes.set_src(train_img);

    if ( m_shapes.infos.size() > 1 )
    {
        cv::Mat padded_img = cv::Mat(train_img.rows * 3, train_img.cols * 3, train_img.type(), cv::Scalar::all(0));
        train_img.copyTo(padded_img(cv::Rect(train_img.cols, train_img.rows, train_img.cols, train_img.rows)));
        train_img = padded_img;
    }

    for(auto& info: m_shapes.infos)
    {
        //feature numbers missing it means using the detector initial num 
        int templ_id = m_detector.addTemplate(m_shapes.src_of(info), class_id, m_shapes.mask_of(info));
        if(templ_id != -1)
        {
            infos_have_templ.push_back(info);
        }

        // 输出模板特征点
        auto templ = m_detector.getTemplates(class_id, templ_id);
        cv::Vec3b randColor;
        randColor[0] = rand()%155 + 100;
        randColor[1] = rand()%155 + 100;
        randColor[2] = rand()%155 + 100;
        for(int i=0; i< templ[0].features.size(); i++){
            auto feat = templ[0].features[i];
            cv::circle(show_image, {feat.x, feat.y}, 3, randColor, -1);
        }
        cv::imwrite("./results/shape_match_points.png", show_image);

    }
    // 多尺度加速分割顶层和底层模板
    m_detector.split_class_templates({class_id}, m_scale_number, m_shapes.infos.size()/m_scale_number);

    // save templates
    m_detector.writeClasses(model_path, template_model_name);
    m_shapes.save_infos(infos_have_templ, "./models/" + class_id  + "_" + template_model_name + "_infos.yaml");

    return true;

}

void Linemode_Template_Match::read_labels(const std::string& template_image_path)
{
    std::vector<cv::String> fn;
    cv::glob(template_image_path.c_str(), fn, false);
    index_char.clear();
    std::string image_name;
    for(int i = 0; i < fn.size(); ++i)
    {
        image_name = boost::filesystem::path(fn[i]).stem().string();
        // 模板图像命令规则 字符名称/字符编号
        index_char.insert({i, image_name.substr(0, 1)});
    }    
}

bool Linemode_Template_Match::train_model(const std::string& template_image_path, const std::string& model_path, const std::string& class_id, const std::string& template_model_name)
{
    // 删除已有模板
    delete_model({class_id});

    std::vector<cv::String> fn;
    cv::glob(template_image_path.c_str(), fn, false);

    std::vector<shape_based_matching::shapeInfo_producer::Info> infos_have_templ;
    cv::Mat train_img;
    boost::filesystem::path path;
    std::string image_name;
    index_char.clear();
    for(int i = 0; i < fn.size(); ++i)
    {
        path = boost::filesystem::path(fn[i]);
        train_img = cv::imread(fn[i]);
        image_name = path.stem().string();
        // 模板图像命令规则 字符名称/字符编号
        index_char.insert({i, image_name.substr(0, 1)});

        if (train_img.channels() == 3)
        cv::cvtColor(train_img, train_img, cv::COLOR_BGR2GRAY); 

        m_shapes.set_src(train_img);

        if ( m_shapes.infos.size() > 1 )
        {
            cv::Mat padded_img = cv::Mat(train_img.rows * 3, train_img.cols * 3, train_img.type(), cv::Scalar::all(0));
            train_img.copyTo(padded_img(cv::Rect(train_img.cols, train_img.rows, train_img.cols, train_img.rows)));
            train_img = padded_img;
        }  

        for(auto& info: m_shapes.infos)
        {
            //feature numbers missing it means using the detector initial num 
            int templ_id = m_detector.addTemplate(m_shapes.src_of(info), class_id, m_shapes.mask_of(info));
            if(templ_id != -1)
            {
                infos_have_templ.push_back(info);

            }
        }

    }
    // 多尺度加速分割顶层和底层模板
    m_detector.split_class_templates({class_id}, m_scale_number, m_shapes.infos.size()/m_scale_number);

    // save templates
    m_detector.writeClasses(model_path, template_model_name);
    m_shapes.save_infos(infos_have_templ, "./models/" + class_id  + "_" + template_model_name + "_infos.yaml");

    return true;
}

// shape match location  
bool Linemode_Template_Match::recognize(const cv::Mat& to_match, const std::vector<std::string>& class_ids, 
                                        int match_similarity_thresh, cv::Rect& match_roi, bool pyramid_acc)
{

    std::vector<line2Dup::Match> matches;
    if (pyramid_acc)
    {
       matches = m_detector.match_fast(to_match, match_similarity_thresh, class_ids, m_scale_number - 1);
    }
    else
    {
       matches = m_detector.match(to_match, match_similarity_thresh, class_ids);
    }
    if (matches.empty())  return false;

    std::sort(matches.begin(), matches.end(), 
              [](const line2Dup::Match match1, const line2Dup::Match match2)
                { return match1.similarity > match2.similarity;});  
    match_roi.x = matches[0].x;
    match_roi.y = matches[0].y;
    auto templ = m_detector.getTemplates(matches[0].class_id, matches[0].template_id);
    match_roi.width = templ[0].width;
    match_roi.height = templ[0].height;

    // 输出模板点
    cv::Mat show_image = to_match.clone();
    if (show_image.channels() == 1)
    cv::cvtColor(show_image, show_image, cv::COLOR_GRAY2BGR); 

    cv::Vec3b randColor;
    randColor[0] = rand()%155 + 100;
    randColor[1] = rand()%155 + 100;
    randColor[2] = rand()%155 + 100;
    for(int i=0; i< templ[0].features.size(); i++){
        auto feat = templ[0].features[i];
        cv::circle(show_image, {feat.x + matches[0].x, feat.y + matches[0].y}, 3, randColor, -1);
    }
    cv::imwrite("./results/shape_match_points.png", show_image);

    return true;
}


// char recognize
std::string Linemode_Template_Match::recognize(const cv::Mat& to_match, const std::vector<std::string>& class_ids, 
                                               int match_similarity_thresh, bool pyramid_acc)
{

    std::vector<line2Dup::Match> matches;
    if (pyramid_acc)
    {
        // 多模板匹配识别，要求分辨率足够大
        matches = m_detector.match_fast(to_match, match_similarity_thresh, class_ids, m_scale_number - 1);
    }
    else
    {
        matches = m_detector.match(to_match, match_similarity_thresh, class_ids);
    }
    if (matches.empty())
    {
        return std::string();
    }

    // 获取最大相似度match的特征点长度
    std::sort(matches.begin(), matches.end(), 
              [](const line2Dup::Match match1, const line2Dup::Match match2)
                { return match1.similarity > match2.similarity;});     
    int max_sim_match_size = m_detector.getTemplates(matches[0].class_id, matches[0].template_id)[0].features.size();

    for (int i = 0; i < matches.size(); ++i)
    {
        std::cout << m_detector.getTemplates(matches[i].class_id, matches[i].template_id)[0].features.size() 
                  << ", " << matches[i].similarity << ", ";
    }
    std::cout << std::endl;

    // 统计前20相似度match的特征点长度
    std::map<int, int> size_counts;
    int max_size = (matches.size() > 20) ? 20 : matches.size();
    for (auto it = matches.begin(); it != matches.begin() + max_size; ++it)
    {
        int match_feature_size = m_detector.getTemplates(it->class_id, it->template_id)[0].features.size();
        if (size_counts.find(match_feature_size) == size_counts.end())
        {
            size_counts.insert({match_feature_size, 1});

        }
        else ++size_counts[match_feature_size];    // 递增某个特征长度的计数
    }

    std::vector<std::pair<int, int>> size_count_vec(size_counts.begin(), size_counts.end());
    std::sort(size_count_vec.begin(), size_count_vec.end(), [](const std::pair<int, int>& sc1, const std::pair<int, int>& sc2)
                                 {  return sc1.second > sc2.second; });
    auto it = matches.begin();
    auto templ = m_detector.getTemplates(it->class_id, it->template_id);
    
    // 输出模板点
    cv::Mat show_image = to_match.clone();
    if (show_image.channels() == 1)
    cv::cvtColor(show_image, show_image, cv::COLOR_GRAY2BGR); 

    cv::Vec3b randColor;
    randColor[0] = rand()%155 + 100;
    randColor[1] = rand()%155 + 100;
    randColor[2] = rand()%155 + 100;
    for(int i=0; i< templ[0].features.size(); i++){
        auto feat = templ[0].features[i];
        cv::circle(show_image, {feat.x +it->x, feat.y + it->y}, 1, randColor, -1);
    }
    cv::imwrite("./results/shape_match_points_char.png", show_image);

    return index_char.find(it->template_id/m_shapes.infos.size())->second;

}

std::string Linemode_Template_Match::recognize(const cv::Mat& to_match, const std::vector<std::string>& class_ids, int match_similarity_thresh, 
                                               int nms_similarity_thresh, float num_iou_thresh, std::vector<cv::Rect>& match_rois, 
                                               bool pyramid_acc, const std::vector<cv::Rect>& mask_rois)
{

    cv::Mat to_match_roi = to_match;
    std::vector<line2Dup::Match> matches;
    if (pyramid_acc)
    {
        // 多模板匹配分辨率需要足够大
        matches = m_detector.match_fast(to_match, match_similarity_thresh, class_ids, m_scale_number - 1);
    }
    else
    {
        matches = m_detector.match(to_match, match_similarity_thresh, class_ids);
    }
    if (matches.empty())   return std::string(); 

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> idxs;
    std::vector<std::pair<int, cv::Rect>> id_boxes;
    cv::Rect box;
    for(auto match: matches){
        box.x = match.x;
        box.y = match.y;

        auto templ = m_detector.getTemplates(match.class_id, match.template_id);

        box.width = templ[0].width;
        box.height = templ[0].height;

        if ((box.x + box.width) > to_match_roi.cols )
        {
            continue;
        }

        boxes.push_back(box);
        scores.push_back(match.similarity);
        
    }

    cv::dnn::NMSBoxes(boxes, scores, nms_similarity_thresh, num_iou_thresh, idxs);
    for (auto id: idxs)
    {
        id_boxes.push_back({id, boxes[id]});
    }
    std::sort(id_boxes.begin(), id_boxes.end(), [](const std::pair<int, cv::Rect> idb1, 
              const std::pair<int, cv::Rect> idb2) { return (idb1.second.x) < (idb2.second.x); });


    cv::Mat show_image = to_match.clone();
    if (show_image.channels() == 1)
    cv::cvtColor(show_image, show_image, cv::COLOR_GRAY2BGR); 


    std::string found_string;
    match_rois.clear();
    for(int i = 0; i < id_boxes.size(); ++i){
        auto match = matches[id_boxes[i].first];
        auto templ = m_detector.getTemplates(match.class_id, match.template_id);
        found_string = found_string + index_char.find(match.template_id/m_shapes.infos.size())->second;
        box.x = match.x;
        box.y = match.y;
        box.width = templ[0].width;
        box.height = templ[0].height;
        match_rois.push_back(box);

        // 输出模板点
        cv::Vec3b randColor;
        randColor[0] = rand()%155 + 100;
        randColor[1] = rand()%155 + 100;
        randColor[2] = rand()%155 + 100;
        for(int i=0; i< templ[0].features.size(); i++){
            auto feat = templ[0].features[i];
            cv::circle(show_image, {feat.x + match.x, feat.y + match.y}, 1, randColor, -1);
        }
        cv::imwrite("./results/shape_match_points.png", show_image);

    }
    cv::Mat save_mat = to_match.clone();
    cv::cvtColor(save_mat, save_mat, cv::COLOR_GRAY2BGR);
    for (auto roi: match_rois)
    {
        cv::rectangle(save_mat, roi, cv::Scalar(0, 255, 0), 3);
        
    }
    cv::imwrite("./results/match_result_rois.png", save_mat);

    return found_string; 
}


