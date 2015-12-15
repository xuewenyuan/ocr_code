#include<iostream>
#include<stdlib.h>
#include<vector>
#include<opencv2\opencv.hpp>
#include<direct.h>
#include"predict.h"
#include"segment.h"
using namespace std;


void help()
{
	cout << "format error" << endl;
	cout << "ocr.exe filepath resultpath" << endl;
}
/*
//�����ڸ���ͷ�ļ����
int segment(const char* filepath, vector<vector<cv::Mat>>& vvM); 
int predict(const char* resultpath, vector<vector<cv::Mat>>& vvM);

*/

int main(int argc, char* argv[])
{
	if (argc != 3)
	{
		help();
		return -1;
	}
	char *filepath = NULL;//���鵥·��
	char *resultpath = NULL;//������ַ��ı���·��
	vector<vector<cv::Mat>> vvM;
	filepath = argv[1];
	resultpath = argv[2];

	segment(filepath, vvM);//���ݷ���ֵ�ж�segment�Ƿ���ȷ
#ifdef _DEBUG
	_mkdir("./tempimage");
	for(int i=0;i<vvM.size();i++)
	{
		for (int j = 0; j < vvM[i].size();j++)
		{
			char imagename[64] = { 0 };			
			sprintf(imagename, "./tempimage/%04d_%04d.png", i, j);
			cv::imwrite(imagename, vvM[i][j]);
		}
	}
#endif
	predict(resultpath, vvM);//���ݷ���ֵ�ж�predict�Ƿ���ȷ

	return 0;


	
}