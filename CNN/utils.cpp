#include <iostream>
#include "utils.h"
#include <cuda_runtime.h>
#include <cassert>

Tensor::Tensor(const std::vector<int>& shape_) : shape(shape_){
	total_size =1;
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
	on_gpu = true;
}

void Tensor::to_host(){
        cudaMemcpy(host_data.data(), device_data, total_size * sizeof(float), cudaMemcpyDeviceToHost);
	on_gpu = false;
}

void Tensor::print(){

	if(on_gpu){
		cudaMemcpy(host_data.data(), device_data, total_size * sizeof(float), cudaMemcpyDeviceToHost);
	}

	for(int f1=0;f1<total_size;f1++){
		for(int f2=0; f2<strides.size()-1;f2++){
			if(f1 % strides[f2]==0){
				std::cout<<"\n";
			}
		}
		std::cout << host_data[f1]<<" , ";
	}
	std::cout<<"\n";
}

int Tensor::flatten_index(const std::vector<int>& indices) const {
	int idx = 0;
	for(int f1=0; f1<indices.size();f1++){
		idx += indices[f1] * strides[f1];
	}
	return idx;
}

float&  Tensor::operator[](const std::vector<int>& indices) {
	if(on_gpu){
		cudaMemcpy(host_data.data(), device_data, total_size * sizeof(float), cudaMemcpyDeviceToHost);
	}
	return host_data[flatten_index(indices)];
}

void Tensor::zeros(){
	cudaMemcpy(host_data.data(), device_data, total_size * sizeof(float), cudaMemcpyDeviceToHost);
	for(int f1=0; f1<total_size; f1++){
		host_data[f1] = 0;
	}
}

void Tensor::ones(){
	cudaMemcpy(host_data.data(), device_data, total_size * sizeof(float), cudaMemcpyDeviceToHost);
	for(int f1=0; f1<total_size; f1++){
		host_data[f1] = 1;
	}
}

void Tensor::full(const int& value){
	cudaMemcpy(host_data.data(), device_data, total_size * sizeof(float), cudaMemcpyDeviceToHost);
	for(int f1=0; f1<total_size; f1++){
		host_data[f1] = value;
	}
}

//void arrange(){

//}

//void rand(){

//}

//void clone(){

//}
