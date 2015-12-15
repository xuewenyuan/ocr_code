
#ifndef _PREDICT_H_
#define _PREDICT_H_
#include<vector>
#include<opencv2\opencv.hpp>


//ʵ��������ȡ��Ԥ��ĺ���
/*
*���� resultpath ���·��
*���� vvM ��ά�ֿ�����
*����ֵ 0����������
*/
int predict(const char* resultpath, const std::vector<std::vector<cv::Mat>>& vvM);




/*
*���� vvM ��ά�ֿ�����
*��� vvi ����vvM�ֿ�Ԥ��Ľ�����������ֵ�ı�ţ���int ���ͱ�ʾ������ά����
*����ֵ 0���������� ����ֵ�����쳣
*/
int predict(const std::vector<std::vector<cv::Mat>>& vvM, std::vector<std::vector<int>>& vvi);






//���ýӿ�,��ʱ���ù�
/*
*���� charMat �ָ�õ��ַ�
*
*����ֵ �������ֵ�ı�ţ���int ���ͱ�ʾ�� ��ֵ�����쳣
*/
int predict(const cv::Mat& charMat);
#endif