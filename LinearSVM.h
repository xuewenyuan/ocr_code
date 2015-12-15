//#pragma once
#ifndef _LINEARSVM_H_
#define _LINEARSVM_H_
#include<linear.h>
#include<vector>
class LinearSVM
{
public:
	LinearSVM();
	~LinearSVM();

	int predict_s(const std::vector<float>& features);
	int load_svm_model(const char* modelpath);
private:
	//feature_node * pnode;
	model * linearmodel;
 
};

#endif