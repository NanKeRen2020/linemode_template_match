#include <boost/filesystem.hpp>
#include <map>
#include <numeric>
#include <future>
#include <thread>
#include "std_utils.h"

int main(int argc, char* argv[])
{
    // linemode 模板匹配 
    cv::Mat to_recognize = cv::imread(argv[1]);
    if (to_recognize.channels() == 3)
        cv::cvtColor(to_recognize, to_recognize, cv::COLOR_BGR2GRAY);     

    // 多尺度模板，匹配定位加速
    Linemode_Template_Match linemode_match_loc(128, {2}, {0, 1}, 2, {0.5, 1.0}, 0.5, {"test_loc"}, "test_loc", 10, 10);

    // 多模板匹配识别，图像分辨率不够时无法开启多尺度加速
    Linemode_Template_Match linemode_match_ocr(128, {2}, {0, 1}, 2, {1.0, 1.5}, 2.5, {"test_ocr"}, "test_ocr", 10, 10);
    
    // 设置与模板图像字符对应的label
    linemode_match_ocr.set_labels({
    
        {0, "0"}, {1, "1"}, {2, "2"}, {3, "3"}, {4, "4"}, {5, "5"}, {6, "6"}, {7, "7"},  {8, "8"}, 
        {9, "9"}, {10, "A"}, {11, "B"}, {12, "S"},  {13, "T"}, {14, "Y"} 
   
    });
    cv::Rect match_roi;
    std::vector<cv::Rect> match_rois;
    std::string result;
    
    if (std::string(argv[2]) == "loc")
    {
        // detect the location 
        if (!linemode_match_loc.is_loaded("test_loc"))
           linemode_match_loc.train_model(argv[3], {"test_loc"}, "test_loc");

        linemode_match_loc.recognize(to_recognize, {"test_loc"}, 80, match_roi, true);
        
        cv::Mat write_mat = to_recognize.clone();
        if (write_mat.channels() == 1)
            cv::cvtColor(write_mat, write_mat, cv::COLOR_GRAY2BGR); 
        cv::rectangle(write_mat, match_roi, cv::Scalar(0, 255, 0), 2, 8, 0);
        cv::imwrite("./datas/locate_text_area.png", write_mat);
               
    }
    else
    {
        // recognize the chars 
        if (!linemode_match_ocr.is_loaded("test_ocr"))
           linemode_match_ocr.train_model(argv[3], {"test_ocr"}, "test_ocr");
        
        result = linemode_match_ocr.recognize(to_recognize, {"test_ocr"}, 70, 0.8, 0, match_rois);
        std::cout << "ocr result: " << result << std::endl;
    }
    
    
}
