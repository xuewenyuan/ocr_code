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
		int kenelsize;	// �˳ߴ�		
		float theta[4];	// ����		
		float sigma;	// ��׼���˹��		
		float waves[3];	// ���� lambda �������������߶�		  sinusoidal factor
		float gamma;	// �ռ��ݺ��		
		float psi;		// ��λƫ��
	};
	class FeatureExtractor
	{
	public:
		//Ĭ�ϲ�����ʼ��
		FeatureExtractor();
		//ָ���߶���Ŀ����ʼ��
		FeatureExtractor(int thetanum,int wavenum);
		//�Զ���GaborParam����ʼ��
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
		int level; //����������
		GaborParam param;
		std::vector<cv::Mat> kernels;
	};
	/*
	* ��ȡGabor������+label��,������featurepath
	* basepath��ͼ�����ַ
	* imagespath��ͼ���б�(���·��+��ǩ)
	* featurepath: �����ͱ�ǩ���ļ�·��
	*/
	void ExtractGaborFeatureWithLabel(std::string basepath, std::string imagespath, std::string featurepath);
	/*
	* ��ȡGabor������+label��ͬʱ������Mat��
	* basepath��ͼ�����ַ
	* imagespath��ͼ���б�(���·��+��ǩ)
	* featuresMat: ���������ͱ�ǩ
	*/
	void ExtractGaborFeatureWithLabel(std::string basepath, std::string imagespath, cv::Mat& featuresMat);
	void ExtractGaborFeature(const cv::Mat& image, cv::Mat& featureMat);
	void ExtractGaborFeature(std::string basepath, std::string imagespath, cv::Mat& featuresMat);
}

#endif
