#include"FeatureExtractor.h"
#include"LinearSVM.h"
#include"predict.h"
#include<iostream>
#include<fstream>
#include<string>
using namespace std;
using namespace visint_ocr;
int predict(const char* resultpath, const std::vector<std::vector<cv::Mat>>& vvM)
{

	//加载字典
	map<int, string> mistr;
	ifstream inputmap("./body.map");
	if (inputmap.fail())
		return -1;
	while (!inputmap.eof())
	{
		int l;
		string hanzi;
		inputmap >> l >> hanzi;

		mistr[l] = hanzi;
	}
	inputmap.close();//加载完毕
	
	ofstream output(resultpath);
	if (output.fail())
		return -1;//创建文件失败
	string chars;
	FeatureExtractor fex(4, 3);
	fex.setPyramidLevel(3);
	LinearSVM svm;
	svm.load_svm_model("./body.model");
	for (size_t i = 0; i < vvM.size(); i++)
	{
		
		for (size_t j = 0; j < vvM[i].size(); j++)
		{
			vector<float> features;//特征向量
			fex.Extract(vvM[i][j], features);
			int label =svm.predict_s(features);
			chars += mistr[label];
		}
		chars+=" ";
	}


	output << chars;
	output.close();
	return 0;
}


int predict(const std::vector<std::vector<cv::Mat>>& vvM, std::vector<std::vector<int>>& vvi)
{
	FeatureExtractor fex(4, 3);
	fex.setPyramidLevel(3);
	LinearSVM svm;
	svm.load_svm_model("./head.model");
	for (size_t i = 0; i < vvM.size(); i++)
	{
		for (size_t j = 0; j < vvM[i].size(); j++)
		{
			vector<float> features;//特征向量
			fex.Extract(vvM[i][j], features);
			int label = svm.predict_s(features);
			vvi[i].push_back(label);
		}
		
	}
	return 0;
}


//备用接口
int predict(const cv::Mat& charMat)
{
	FeatureExtractor fex(4, 3);
	fex.setPyramidLevel(3);
	LinearSVM svm;
	svm.load_svm_model("./body.model");
	vector<float> features;//特征向量
	fex.Extract(charMat, features);
	int label = svm.predict_s(features);
	return label;
}