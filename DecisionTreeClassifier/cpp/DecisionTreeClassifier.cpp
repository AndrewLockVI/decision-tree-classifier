#include "DecisionTreeClassifier.h"
#include <limits>
#include <iostream>
#include <unordered_map>
#include <thread>
#include <mutex>

using namespace std;

DecisionTreeClassifier::DecisionTreeClassifier(int maxDepth){
	this->depth = maxDepth;
}

void DecisionTreeClassifier::fit(float* X, int samples, int* y, int features){

	if (splittingTree != nullptr){
		deleteTree(splittingTree);
	}

	if(features <= 0){
		throw invalid_argument("Invalid argument, there must be 1 or more features to train on.");
	}

	if(samples <= 0){
		throw invalid_argument("Invalid argument, there must be 1 or more samples to train on.");
	}

	splittingTree = recurse(X, samples, y, features, depth);
	featureCount = features;

}


std::string DecisionTreeClassifier::getDot(){
	if (splittingTree == nullptr){
		throw logic_error("Decision tree must be created prior to generating dot output.");
	}
	std::string edges = splittingTree->getDotEdges();
	std::string dot = "digraph decisionTree {\n" + edges + "}";
	return dot;
}

int DecisionTreeClassifier::primaryClass(int* y, int labelCount){

	unordered_map map = unordered_map<int,int>();

	for(int i = 0; i < labelCount; ++i){
		map[y[i]] += 1;
	}

	int mostElements = 0;
	int label = 0;

	for (auto& item : map){
		if(item.second > mostElements){
			mostElements = item.second;
			label = item.first;
		}
	}

	return label;
}



// add depth
TreeNode* DecisionTreeClassifier::recurse(float* X, int rows, int* y, int columns, int depthRem){

	if(depthRem == 0){
		TreeNode* ret = new TreeNode(primaryClass(y, rows));
		return ret;
	}

	// found minimum node
	if(rows == 1){
		TreeNode* ret = new TreeNode(primaryClass(y, rows));
		return ret; 
	}

	// get best split option 
	TreeNode* chosen = bestSplit(X, rows, y, columns);
	SplitResults split = chosen->splitOnNode(X, y, rows, columns);

	// no valid splits, but we still did create some new arrays.
	if(split.rightSize == rows || split.leftSize == rows){
		TreeNode* ret = new TreeNode(primaryClass(y, rows));
		delete split.XLeft;
		delete split.XRight;
		delete split.yLeft;
		delete split.yRight;
		return ret; 
	}

	// traverse lt tree
	TreeNode* left = recurse(split.XLeft, split.leftSize, split.yLeft, columns, depthRem - 1);
	// traverse gt tree
	TreeNode* right = recurse(split.XRight, split.rightSize, split.yRight, columns, depthRem - 1);

	chosen->setLeftChild(left);
	chosen->setRightChild(right);

	delete split.XLeft;
	delete split.XRight;
	delete split.yLeft;
	delete split.yRight;

	return chosen;
}





//	1	1	0
//	3	3	0
//	2	1	1
//	4	1	3





// consider adding interpolation to this and sorting the list first.
// Also, no reason to consider the 0th split if that is the case.

TreeNode* DecisionTreeClassifier::bestSplit(float* X, int rows, int* y, int columns) {
    TreeNode* bestNode = nullptr;
    float bestGini = std::numeric_limits<float>::max();
    std::mutex mtx; 

    auto evalColumn = [&](int col){
        TreeNode* localBestNode = nullptr;
        float localBestGini = std::numeric_limits<float>::max();

        for (int row = 0; row < rows; ++row){
            float val = X[row * columns + col];
            TreeNode* current = new TreeNode(val, col);
            float gini = current->evalSplit(X, y, rows, columns, "gini");

            if (gini < localBestGini){
                delete localBestNode;
                localBestNode = current;
                localBestGini = gini;
            }
			else{
                delete current;
            }
        }

        std::lock_guard<std::mutex> lock(mtx);
        if (localBestGini < bestGini){
            delete bestNode;
            bestNode = localBestNode;
            bestGini = localBestGini;
        }
		else{
            delete localBestNode;
        }
    };

    std::vector<std::thread> threads;
    for (int col = 0; col < columns; ++col) {
        threads.emplace_back(evalColumn, col);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    return bestNode;
}

int* DecisionTreeClassifier::predict(float* X, int samples, int features) {

	if(featureCount == -1){
		throw logic_error("Unable to predict prior to calling fit().");
	}

	if(features != this->featureCount){
		throw invalid_argument("Incorrect number of features for prediction.");
	}

	int* predictions = new int[samples];

	for(int i = 0; i < samples; ++i){
		TreeNode* current = splittingTree;
		while(!current->isLeaf()){
			float* currentElement = X;
			currentElement += features * i;
			bool lessThan = current->lessThan(currentElement, features);
			if(lessThan){
				current = current->getLeftChild();
			}
			else{
				current = current->getRightChild();
			}
		}
		predictions[i] = current->getClassification();
	}

	return predictions;
}

DecisionTreeClassifier::~DecisionTreeClassifier(){
	deleteTree(splittingTree);

}

void DecisionTreeClassifier::deleteTree(TreeNode* node){

	if(node == nullptr){
		return;
	}

	deleteTree(node->getLeftChild());
	deleteTree(node->getRightChild());
	delete node;
}
