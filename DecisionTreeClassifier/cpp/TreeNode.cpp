#include "TreeNode.h"
#include "stdexcept"
#include "Criterion.h"
#include "math.h"
#include "iostream"
#include <string>
#include <sstream>

TreeNode::TreeNode(int classification){
	leaf = true;
	this->classification = classification;
}

TreeNode::TreeNode(float splittingVal, int featureIndex){
	splitValue = splittingVal;
	index = featureIndex;
	leaf = false;
}

void TreeNode::setSplit(float splittingVal, int featureIndex){
	splitValue = splittingVal;
	index = featureIndex;
	leaf = false;
}

bool TreeNode::isLeaf(){
	return leaf;
}

float TreeNode::evalSplit(float* X, int* y, int samples, int features, std::string criterion){

	if(isLeaf()){
		throw std::logic_error("Cannot evaluate split on leaf node.");
	}

	if(criterion != "gini"){
		throw std::invalid_argument("Gini impurity is the only supported criterion.");
	}

	Criterion evalCriterion= Criterion();

	return evalCriterion.giniImpurity(X, y, samples, features, this->index, this->splitValue);
}


void TreeNode::setLeftChild(TreeNode* child){
	leftChild = child;
}

void TreeNode::setRightChild(TreeNode* child){
	rightChild = child;
}

TreeNode* TreeNode::getLeftChild(){
	return leftChild;
}

TreeNode* TreeNode::getRightChild(){
	return rightChild;
}

float TreeNode::getSplitVal(){
	return splitValue;
}

int TreeNode::getIndexSplit(){
	return index;
}

SplitResults TreeNode::splitOnNode(float* X, int* y, int samples, int features){

	SplitResults result = SplitResults();

	int ltCount = 0;
	int gteqCount = 0;

	for(int i = 0 ; i < samples; ++i){
		if(X[(i*features) + index] < splitValue){
			ltCount += 1;
		}
		else{
			gteqCount += 1;
		}
	}

	// Create X arrays to return

	float* ltArr = new float[ltCount * features];
	float* gteqArr = new float[gteqCount * features];

	// Create array ptr next open

	float* nextLtX = ltArr;
	float* nextGteqX = gteqArr;

	// Create y arrays to return

	int* ltYArr = new int[ltCount];
	int* gteqYArr = new int[gteqCount];

	// Create array ptr next open

	int* nextLtY = ltYArr;
	int* nextGteqY = gteqYArr;

	// Set pointers for return to the new arrays

	result.XLeft = ltArr;
	result.yLeft = ltYArr;

	result.XRight = gteqArr;
	result.yRight = gteqYArr;

	result.leftSize = ltCount;
	result.rightSize = gteqCount;

	// Set arrays with correct values

	for(int i = 0 ; i < samples; ++i){
		if(X[(i*features) + index] < splitValue){
			for(int x = 0; x < features; ++x){
				nextLtX[x] = X[(i*features) + x];
			}

			nextLtX += features;

			nextLtY[0] = y[i];
			nextLtY += 1;
		}
		else{
			for(int x = 0; x < features; ++x){
				nextGteqX[x] = X[(i*features) + x];
			}

			nextGteqX += features;

			nextGteqY[0] = y[i];
			nextGteqY += 1;
		}
	}

	//for(int x = 0 ; x < ltCount; ++x){
	//	for(int i = 0 ; i < features; ++i){
	//		std::cout << ltArr[x*features + i];
	//	}
	//	std::cout << std::endl;
	//}

	//for(int x = 0 ; x < ltCount; ++x){
	//	std::cout << ltYArr[x] << std::endl;
	//}

	return result;
}







std::string TreeNode::getDotEdges(){

	if(isLeaf()){
		return "";
	}

	std::string current = getDotLabel() + "->" + leftChild->getDotLabel() + ";\n";
	current += getDotLabel() + "->" + rightChild->getDotLabel() + ";\n";

	current += rightChild->getDotEdges();
	current += leftChild->getDotEdges();

	return current;
}

std::string TreeNode::getDotLabel(){
	const void * address = static_cast<const void*>(this);
	std::stringstream ss;
	ss << address;  
	std::string name = ss.str(); 
	if (isLeaf()){
		return "\"" + name + "\nCLASSIFICATION: " + std::to_string(classification) + "\"";
	}

	return "\"" + name + "\nINDEX: " +  std::to_string(index) + "\nVALUE:" + std::to_string(splitValue) + "\"";
}

int TreeNode::getClassification(){
	if(isLeaf()){
		return classification;
	}
	throw std::logic_error("Unable to call getClassification() on internal vertices.");
}

bool TreeNode::lessThan(float* sample, int features){

	if(features < this->index){
		throw std::invalid_argument("Attempting to evaluate split with input that contains less features.");
	}

	return(sample[index] < splitValue);
}
