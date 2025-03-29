#include "utils.h"
#include <cuda_runtime.h>
#include <cassert>

Tensor(const std::vector<int>& shape_) shape(shape_){
	int total_size =1;
        strides.resize(shape.size());

        for(f1=shape.size()-1; f1>=0;f1--){
        	stride[f1]=total_size;
                total_size*=shape[f1];
        }
        host_data.resize(total_size);
}

Tensor::~Tensor(){
}

void Tensor::to_device(){
	cudaMalloc(&device_data, total_size * sizeof(float));
        cudaMemcpy(device_data, host_data.data(), total_size * sizeof(float), cudaMemcpyHostToDevice);
}

void Tensor::to_host(){
        cudaMemcpy(host_data.data(), device_data, total_size * sizeof(float), cudaMemcpyDeviceToHost);
}

int Tensor::flatten_index(const std::vector<int>indices){
	for(f1=shape
}



