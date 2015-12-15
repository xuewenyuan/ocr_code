#include "FeatureExtractor.h"
#include<iostream>
#include<fstream>
using namespace visint_ocr;

/*
* 参数：需要用通过实验找到最优参数
* theta：四个方向是确定的
* lambda：波长（尺度）支持1-3个尺度【根据实验做具体设置】
* psi：0
* gamma: 0.5 usually
*/
void FeatureExtractor::Init()
{
	//应该是读配置文件
	param.kenelsize = 11;
	param.theta[0] = 0.0, param.theta[1] = CV_PI / 4, param.theta[2] = CV_PI / 2, param.theta[3] = CV_PI*3.0 / 4.0;
	param.sigma = CV_PI/3;
	param.waves[0] = 2 * CV_PI / 3, param.waves[1] = 2.5* CV_PI / 3, param.waves[2] = 3 * CV_PI / 3;

	//param.sigma = CV_PI;
	//param.waves[0] = 2 * CV_PI / 3, param.waves[1] = 2.5* CV_PI / 6, param.waves[2] = 3 * CV_PI / 12;
	param.gamma = 0.5;
	param.psi = 0;
	this->thetanum = 4;
	this->wavenum = 1;
	this->level = 3;
}

FeatureExtractor::FeatureExtractor()
{
	Init();
	BuildKernelFamily();
}


FeatureExtractor::FeatureExtractor(int thetanum, int wavenum)
{
	this->Init();
	this->thetanum = thetanum;
	this->wavenum = wavenum;
	BuildKernelFamily();
}

FeatureExtractor::FeatureExtractor(const GaborParam& param, int thetanum, int wavenum)
{
	this->param.kenelsize = param.kenelsize;
	this->param.gamma = param.gamma;
	this->param.psi = param.psi;
	this->thetanum = thetanum;
	this->wavenum = wavenum;
	for (int i = 0; i < wavenum; i++)
	{
		this->param.waves[i] = param.waves[i];
	}
	for (int i = 0; i < thetanum; i++)
	{
		this->param.theta[i] = param.theta[i];
	}
	this->level = 3;
	BuildKernelFamily();
}

FeatureExtractor::~FeatureExtractor()
{
}


void FeatureExtractor::BuildKernelFamily()
{
	kernels.clear();
	for (int i = 0; i < wavenum; i++)
	{
		for (int j = 0; j < thetanum; j++)
		{
			cv::Mat kernel = cv::getGaborKernel(cv::Size(param.kenelsize, param.kenelsize), param.sigma, param.theta[j], param.waves[i], param.gamma,param.psi);//, param.psi
			this->kernels.push_back(kernel);
		}
	}
}

void FeatureExtractor::Pyramid(const cv::Mat& dst, int level, std::vector<float>& features)
{

	int H = dst.rows / (int)pow(2, level);
	int W = dst.cols / (int)pow(2, level);
	for (int i = H; i <= dst.rows; i += H)
	{
		for (int j = W; j <= dst.cols; j += W)
		{
			cv::Mat tempdst = dst.rowRange(i - H, i).colRange(j - W, j);
			cv::Scalar meanv, stdDev;
			cv::meanStdDev(tempdst, meanv, stdDev);
			features.push_back(meanv[0]);
			features.push_back(stdDev[0]);
		}
	}
}

void FeatureExtractor::setPyramidLevel(int level)
{
	if (level >= 3||level<=0) 
		this->level = 3;
	else 
		this->level = level;
}

void FeatureExtractor::Normalize(std::vector<float>& features)
{
	float meanmin = features[0], meanmax = features[0], stdmin = features[1], stdmax = features[1];
	for (size_t i = 0; i < features.size(); i += 2)
	{
		//mean
		meanmin = std::min(meanmin, features[i]);
		meanmax = std::max(meanmax, features[i]);
		stdmin = std::min(stdmin, features[i + 1]);
		stdmax = std::max(stdmax, features[i + 1]);
	}
	for (size_t i = 0; i < features.size(); i += 2)
	{
		features[i] = (features[i] - meanmin) / (meanmax - meanmin + 0.00001);
		features[i + 1] = (features[i + 1] - stdmin) / (stdmax - stdmin + 0.00001);
	}
}

void FeatureExtractor::Extract(const cv::Mat& img, std::vector<float>& features)
{
	//features.clear();
	cv::Mat inv = img;// 浅copy
	if (inv.channels() == 3)
	{
		cvtColor(inv, inv, cv::COLOR_BGR2GRAY);
	}
	int height = 16;
	int width = (int)(height*1.0*inv.rows / inv.cols);
	cv::resize(inv, inv, cv::Size(height, width));
	inv = 255 - inv;
	cv::Mat text_f;
	inv.convertTo(text_f, CV_32F, 1.0 / 255.0, 0);

	for (size_t i = 0; i < kernels.size(); i++)
	{

		cv::Mat dst;
		cv::filter2D(text_f, dst, -1, kernels[i]);

		//第 0 层金字塔
		//Pyramid(dst, 0, features);
		//第 1 层金字塔
		//Pyramid(dst, 1, features);
		//第 2 层金字塔
		//Pyramid(dst, 2, features);

		for (int i = 0; i < this->level; i++)
		{
			Pyramid(dst, i, features);
		}
	}

	//归一化操作
	Normalize(features);
}

void visint_ocr::ExtractGaborFeatureWithLabel(std::string basepath, std::string imagespath, std::string featurespath)
{
	std::vector<float>feature;
	FeatureExtractor fex(4,3);
	fex.setPyramidLevel(3);
	std::ifstream input(imagespath);
	std::ofstream output(featurespath,std::ios::out);
	if (input.fail() || output.fail()) //
	{
		std::cout << "paths error";
		system("pause");
		return;
	}
	//读取imagespath 中图片的路径 、 label，并把抽取的特征依次保存在featuresMat
	//while(1)
	//std::string base = "F:\\WYL\\AllFonts\\";
	//int index = 1;
	while (1)
	{
		feature.clear();
		int label = 0;
		std::string imagepath;
		input >> imagepath >> label;
		if (input.eof()) break;
		output << label ;
		cv::Mat image = cv::imread(basepath+imagepath, 0);
		fex.Extract(image, feature);
		for (size_t i = 0; i < feature.size(); i++)
		{
			output << " "<<(i+1)<<":" << feature[i];
		}
		output << std::endl;
		std::cout << ".";
		//std::cout << index++<<std::endl;
		// write featurespath
	}
	output.close();
	input.close();


}
void visint_ocr::ExtractGaborFeatureWithLabel(std::string basepath,std ::string imagespath, cv::Mat& featuresMat)
{
	std::vector<float>feature;
	FeatureExtractor fex(4,3);
	fex.setPyramidLevel(2);
	//读取imagespath 中图片的路径 、 label，并把抽取的特征依次保存在featuresMat
	//while(1)
	std::ifstream input(imagespath);
	if (input.fail())
	{
		std::cout << "error";
		return;
	}
	while (1)
	{
		feature.clear();
		int label = 0;;
		std::string imagepath;
		input >> imagepath >> label;
		if (input.eof())
		{
			break;
		}
		cv::Mat image = cv::imread(basepath + imagepath, 0);
		fex.Extract(image, feature);
		feature.insert(feature.begin(), label);
		cv::Mat featuremat(feature);		
		featuremat = featuremat.t();
		featuresMat.push_back(featuremat);
		//featuresMat.
	}


}

void visint_ocr::ExtractGaborFeature(const cv::Mat& image, cv::Mat& featureMat)
{
	std::vector<float>feature;
	FeatureExtractor fex(4,3);
	fex.setPyramidLevel(2);
	fex.Extract(image, feature);
	cv::Mat featuremat(feature);
	featuremat = featuremat.t();
	featureMat.push_back(featuremat);
}

void visint_ocr::ExtractGaborFeature(std::string basepath, std::string imagespath, cv::Mat& featuresMat)
{
	std::vector<float>feature;
	FeatureExtractor fex;
	fex.setPyramidLevel(2);
	//读取imagespath 中图片的路径 、 label，并把抽取的特征依次保存在featuresMat
	//while(1)
	std::ifstream input(imagespath);
	if (input.fail())
	{
		std::cout << "error";
		return;
	}
	while (1)
	{
		feature.clear();
		std::string imagepath;
		input >> imagepath;
		if (input.eof())
		{
			break;
		}
		cv::Mat image = cv::imread(imagepath, 0);
		fex.Extract(image, feature);
		cv::Mat featuremat(feature);
		featuremat = featuremat.t();
		featuresMat.push_back(featuremat);
	}
}