
#include<vector>
#include<opencv2\opencv.hpp>
using namespace std;
//实现分割的函数
/*
*输入 filepath 化验单路径
*输出 vvM 二维字块数组
* 返回值 0，代表正常
*/
int segment(const char* filepath, vector<vector<cv::Mat>>& vvM);