#include "utils.h"
#include <cuda_runtime.h>
#include <cassert>

Tensor::Tensor(const std::vector<int>& shape_) : shape(shape_){
	int total_size =1;
        strides.resize(shape.size());

        for(int f1=shape.size()-1; f1>=0;f1--){
        	strides[f1]=total_size;
                total_size*=shape[f1];
        }
        host_data.resize(total_size);
}

Tensor::~Tensor(){
        if (device_data != nullptr) {
            cudaFree(device_data);
        }
}

void Tensor::to_device(){
	cudaMalloc(&device_data, total_size * sizeof(float));
        cudaMemcpy(device_data, host_data.data(), total_size * sizeof(float), cudaMemcpyHostToDevice);
}

void Tensor::to_host(){
        cudaMemcpy(host_data.data(), device_data, total_size * sizeof(float), cudaMemcpyDeviceToHost);
}

int Tensor::flatten_index(const std::vector<int>& indices) const {
	int idx = 0;
	for(int f1=0; f1<indices.size();f1++){
		idx += indices[f1] * strides[f1];
	}
	return idx;
}



