#include <unordered_map>
#include <cmath>
#include <iostream>

using namespace std;

extern "C" {

    struct GiniResult {
        float weighted;
        float ltGini;
        float gtGini;
    };

    GiniResult gini(float* eles, int* classes, int sampleCount, float split, int* indices, int indexCt) {
        
        unordered_map<int, int> ltMap;
        unordered_map<int, int> gtMap;

        int ltCount = 0;
        int gtCount = 0;

        // Split the data based on the threshold
        for(int i = 0; i < indexCt; ++i) {
            if(eles[indices[i]] < split) {
                ltMap[classes[indices[i]]]++;
                ltCount++;
            } else {
                gtMap[classes[indices[i]]]++;
                gtCount++;
            }
        }


		GiniResult result;

        result.ltGini = 1.0f;
        for (const auto& pair : ltMap) {
            result.ltGini -= pow(float(pair.second) / ltCount, 2);
        }

        result.gtGini = 1.0f;
        for (const auto& pair : gtMap) {
            result.gtGini -= pow(float(pair.second) / gtCount, 2);
        }

		if(gtCount == 0){
			result.gtGini = 0.0f;
		}
		if(ltCount == 0){
			result.ltGini = 0.0f;
		}

        result.weighted = result.gtGini * float(gtCount) / sampleCount;
        result.weighted += result.ltGini * float(ltCount) / sampleCount;

        return result;
    }
}
