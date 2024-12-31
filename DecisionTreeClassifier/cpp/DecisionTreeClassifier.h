#include "TreeNode.h"
#include <vector>

class DecisionTreeClassifier{
	public:
		DecisionTreeClassifier(int depth);
		void fit(float* X, int samples, int* y, int features);
		int* predict(float* X, int samples, int features);
		std::string getDot();
		~DecisionTreeClassifier();
	private:
		int depth;
		int featureCount = -1;
		TreeNode* splittingTree = nullptr;
		TreeNode* bestSplit(float* X, int samples, int* y, int features);
		TreeNode* recurse(float* X, int samples, int* y, int features, int depth);
		int primaryClass(int* y, int labelCount);
		void deleteTree(TreeNode* node);
};
