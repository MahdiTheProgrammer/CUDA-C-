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

	for(int f1=0;f1<shape.size();f1++){
		std::cout<<"[";
	}
	bool b=false;
	for(int f1=0;f1<total_size;f1++){
		for(int f2=0; f2<strides.size()-1;f2++){
			if(f1 % strides[f2]==0 && f1!=0 && f1!=1){
				if(f2<shape.size()-2){
					if(f2<shape.size()-3){
					std::cout<<"]\n";
					}else{
					std::cout<<"]]\n";
					}
					b = true;
				}
				else{
					if(b){
						std::cout<<"\n[[";
						b = false;
					}
					else{
						std::cout<<"]\n[";
					}
				}
			}
		}
		std::cout << host_data[f1]<<", ";
	}

	for(int f1=0;f1<shape.size();f1++){
		std::cout<<"]";
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
	if (on_gpu){
		cudaMemcpy(host_data.data(), device_data, total_size * sizeof(float), cudaMemcpyDeviceToHost);
	}
	for(int f1=0; f1<total_size; f1++){
		host_data[f1] = 0;
	}
	if (on_gpu){
		cudaMemcpy(device_data, host_data.data(),total_size * sizeof(float), cudaMemcpyHostToDevice);
	}
}

void Tensor::ones(){
	if (on_gpu){
		cudaMemcpy(host_data.data(), device_data, total_size * sizeof(float), cudaMemcpyDeviceToHost);
	}
	for(int f1=0; f1<total_size; f1++){
		host_data[f1] = 1;
	}
	if (on_gpu){
		cudaMemcpy(device_data, host_data.data(),total_size * sizeof(float), cudaMemcpyHostToDevice);
	}
}

void Tensor::full(const int& value){
	if (on_gpu){
		cudaMemcpy(host_data.data(), device_data, total_size * sizeof(float), cudaMemcpyDeviceToHost);
	}
	for(int f1=0; f1<total_size; f1++){
		host_data[f1] = value;
	}
	if (on_gpu){
		cudaMemcpy(device_data, host_data.data(),total_size * sizeof(float), cudaMemcpyHostToDevice);
	}
}

void Tensor::arrange(const float& start, const float& step){
	if (on_gpu){
		cudaMemcpy(host_data.data(), device_data, total_size * sizeof(float), cudaMemcpyDeviceToHost);
	}
	for(int f1=0; f1<total_size; f1++){
		host_data[f1] = start+(step*f1);
	}
	if (on_gpu){
		cudaMemcpy(device_data, host_data.data(),total_size * sizeof(float), cudaMemcpyHostToDevice);
	}
}

void Tensor::rand(){
}

//void clone(){

//}
