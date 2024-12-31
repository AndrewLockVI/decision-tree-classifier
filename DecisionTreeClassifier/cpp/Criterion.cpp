#include "Criterion.h"
#include <unordered_map>
#include <math.h>

float Criterion::giniImpurity(float* X, int* y, int samples, int features, int index, float splitValue){

	std::unordered_map<int, int> ltMap;
	std::unordered_map<int, int> gtMap;

	int ltCount = 0;
	int gteqCount = 0;

	for(int i = 0; i < samples; ++i){
		if(X[index + (i * features)] < splitValue){
			ltMap[y[i]]++;
			ltCount++;
		}
		else{
			gtMap[y[i]]++;
			gteqCount++;
		}
	}


	float ltGini= 1.0f;

	for (const auto& pair : ltMap) {
		ltGini -= pow(float(pair.second) / ltCount, 2);
	}

	float gteqGini = 1.0f;

	for (const auto& pair : gtMap) {
		gteqGini -= pow(float(pair.second) / gteqCount, 2);
	}

	if(gteqCount == 0){
		gteqGini = 0.0f;
	}
	if(ltCount == 0){
		ltGini = 0.0f;
	}

	float gini = gteqGini * float(gteqCount) / samples;
	gini += ltGini * float(ltCount) / samples;

	return gini;
}

