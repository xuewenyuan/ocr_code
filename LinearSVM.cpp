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
	if (linearmodel == NULL)//ģ�ͼ���ʧ�� 
		return -1; 
	feature_node *pnode = NULL;
	pnode = new feature_node[features.size() + 1];
	if (pnode = NULL)
		return -1;//feature_node �ڴ����ʧ��

	for (size_t i = 0; i < features.size(); i++)
	{
		pnode[i].index = (int)(i + 1);//linearsvm ������1��ʼ
		pnode[i].value = features[i];
	}
	pnode[features.size()].index = -1;//linearsvm ����������־
	int label = -1;
	label = (int)predict(linearmodel, pnode);
	delete[] pnode;
	pnode = NULL;

	return 0;

}