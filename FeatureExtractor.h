//#pragma once
#ifndef _FEATUREEXTRACTOR_H_
#define _FEATUREEXTRACTOR_H_
#include<opencv2\opencv.hpp>
#include<vector>
namespace visint_ocr{

	/*
	* ksize is the size of the Gabor kernel. 
	* sigma is the standard deviation of the Gaussian function used in the Gabor filter.
	* theta is the orientation of the normal to the parallel stripes of the Gabor function.
	* lambda is the wavelength of the sinusoidal factor in the above equation.
	* gamma is the spatial aspect ratio.
	* psi is the phase offset.
	* ktype indicates the type and range of values that each pixel in the Gabor kernel can hold.
	*/

	struct GaborParam
	{
		int kenelsize;	// 核尺寸		
		float theta[4];	// 方向		
		float sigma;	// 标准差（高斯）		
		float waves[3];	// 波长 lambda 《？——？》尺度		  sinusoidal factor
		float gamma;	// 空间纵横比		
		float psi;		// 相位偏移
	};
	class FeatureExtractor
	{
	public:
		//默认参数初始化
		FeatureExtractor();
		//指定尺度数目，初始化
		FeatureExtractor(int thetanum,int wavenum);
		//自定义GaborParam，初始化
		FeatureExtractor(const GaborParam& param, int thetanum, int wavenum);
		void Extract(const cv::Mat& img, std::vector<float>& features);
		void setPyramidLevel(int level);
		~FeatureExtractor();
	private:
		void Init();
		void BuildKernelFamily();
		void Pyramid(const cv::Mat& dst, int i, std::vector<float>& features);
		void Normalize(std::vector<float>& features);
	private:
		int thetanum;
		int wavenum;
		int level; //金字塔层数
		GaborParam param;
		std::vector<cv::Mat> kernels;
	};
	/*
	* 抽取Gabor特征（+label）,保存在featurepath
	* basepath：图像根地址
	* imagespath：图像列表(相对路径+标签)
	* featurepath: 特征和标签的文件路径
	*/
	void ExtractGaborFeatureWithLabel(std::string basepath, std::string imagespath, std::string featurepath);
	/*
	* 抽取Gabor特征（+label）同时保存在Mat中
	* basepath：图像根地址
	* imagespath：图像列表(相对路径+标签)
	* featuresMat: 存贮特征和标签
	*/
	void ExtractGaborFeatureWithLabel(std::string basepath, std::string imagespath, cv::Mat& featuresMat);
	void ExtractGaborFeature(const cv::Mat& image, cv::Mat& featureMat);
	void ExtractGaborFeature(std::string basepath, std::string imagespath, cv::Mat& featuresMat);
}

#endif
