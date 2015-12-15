#include "LinearSVM.h"


LinearSVM::LinearSVM() : linearmodel(NULL)
{
}


LinearSVM::~LinearSVM()
{
	if (linearmodel!=NULL)
		free_and_destroy_model(&linearmodel);
}


int LinearSVM::load_svm_model(const char* modelpath)
{
	linearmodel = load_model(modelpath);
	if (NULL==linearmodel)
		return -1;
	return 0;
}

int LinearSVM::predict_s(const std::vector<float>& features)
{
	if (linearmodel == NULL)//模型加载失败 
		return -1; 
	feature_node *pnode = NULL;
	pnode = new feature_node[features.size() + 1];
	if (pnode = NULL)
		return -1;//feature_node 内存分配失败

	for (size_t i = 0; i < features.size(); i++)
	{
		pnode[i].index = (int)(i + 1);//linearsvm 索引从1开始
		pnode[i].value = features[i];
	}
	pnode[features.size()].index = -1;//linearsvm 索引结束标志
	int label = -1;
	label = (int)predict(linearmodel, pnode);
	delete[] pnode;
	pnode = NULL;

	return 0;

}