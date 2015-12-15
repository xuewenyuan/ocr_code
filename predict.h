
#ifndef _PREDICT_H_
#define _PREDICT_H_
#include<vector>
#include<opencv2\opencv.hpp>


//实现特征提取并预测的函数
/*
*输入 resultpath 结果路径
*输入 vvM 二维字块数组
*返回值 0，代表正常
*/
int predict(const char* resultpath, const std::vector<std::vector<cv::Mat>>& vvM);




/*
*输入 vvM 二维字块数组
*输出 vvi 根据vvM字块预测的结果（汉字在字典的标号，用int 类型表示），二维数组
*返回值 0，代表正常 ，负值代表异常
*/
int predict(const std::vector<std::vector<cv::Mat>>& vvM, std::vector<std::vector<int>>& vvi);






//备用接口,暂时不用管
/*
*输入 charMat 分割好的字符
*
*返回值 汉字在字典的标号，用int 类型表示， 负值代表异常
*/
int predict(const cv::Mat& charMat);
#endif