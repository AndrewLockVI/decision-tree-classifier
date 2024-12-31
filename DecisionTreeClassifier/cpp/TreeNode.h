#include "string"

struct SplitResults{
	float* XLeft;
	float* XRight;
	int* yLeft;
	int* yRight;
	int leftSize;
	int rightSize;
};

class TreeNode{
	public:
		TreeNode(int classification);
		TreeNode(float splittingVal, int featureIndex);
		bool isLeaf();
		void setSplit(float splittingValue, int featureIndex);
		float evalSplit(float* X, int* y, int samples, int features, std::string criterion);
		TreeNode* getLeftChild();
		TreeNode* getRightChild();
		void setLeftChild(TreeNode* child);
		void setRightChild(TreeNode* child);
		float getSplitVal();
		int getIndexSplit();
		SplitResults splitOnNode(float* X, int* y, int samples, int features);
		std::string getDotEdges();
		int getClassification();
		bool lessThan(float* sample, int features);

	private:
		bool leaf;
		float splitValue;
		int index;
		TreeNode* leftChild = nullptr;
		TreeNode* rightChild = nullptr;
		std::string getDotLabel();
		int classification;
};


