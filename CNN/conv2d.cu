#include "layer.h"
#include <cuda_runtime.h>
#include <iostream>

__global__ void convolution(int batch_size, float* input,float*weights, float* bias, float* output ,int input_dim, int output_dim, int height_out, int width_out, int height_in, int width_in,  int kernal_size,int stride){

	int z = blockIdx.z;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int which_kernal = z%output_dim;
	int which_batch = z/output_dim;
	if (col<width_out && row<height_out){
		float i = 0.0f;
		for (int d=0; d<input_dim; d++){
			for (int kh=0;kh<kernal_size;kh++){
				for (int kw=0; kw<kernal_size;kw++){
					i+=weights[(which_kernal*input_dim*kernal_size*kernal_size)+(d*kernal_size*kernal_size)+(kh*kernal_size)+kw] * input[(which_batch*width_in*height_in*input_dim)+(d*width_in*height_in)+(row*(stride)*width_in)+(col*(stride))+kw+(kh*width_in)];
				}
			}
		}

		output[(which_batch * output_dim * width_out * height_out)+(which_kernal * height_out * width_out) + (row * width_out) + col] = i + bias[which_kernal];
	}
}

Tensor Conv2d::forward(Tensor& input){
	input.to_device();
	input.add_padding(padding,0);
	const std::vector<int> input_shape = input.get_shape();

	int batch_size = input_shape[input_shape.size()-4];
	int input_dim = input_shape[input_shape.size()-3];
	int height_in = input_shape[input_shape.size()-2];
	int width_in = input_shape[input_shape.size()-1];

	int height_out = 0;
	int width_out = 0;

	for(int f1=kernal-1; f1<height_in; f1+=stride){
		height_out++;
	}

	for(int f1=kernal-1; f1<width_in; f1+=stride){
		width_out++;
	}

	float* add_X = input.device_address();
	float* add_W = weights.device_address();
	float* add_B = bias.device_address();

	int total_size_output = batch_size * num_outputs * height_out * width_out;
        float* add_output;
        float* output = new float[total_size_output];
        cudaMalloc((void**)&add_output,total_size_output * sizeof(float));

	dim3 blockDim(32,32);
	dim3 gridDim((width_out+31)/32,(height_out+31)/32,num_outputs*batch_size);
	convolution<<<gridDim, blockDim>>>(batch_size, add_X, add_W, add_B, add_output, input_dim, num_outputs, height_out, width_out, height_in, width_in,  kernal, stride);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
    		std::cerr << "CUDA kernel failed: " << cudaGetErrorString(err) << std::endl;
	}
	cudaDeviceSynchronize();

        cudaMemcpy(output, add_output, total_size_output * sizeof(float), cudaMemcpyDeviceToHost);
	std::vector<int> output_shape = {batch_size, num_outputs,height_out,width_out};
	Tensor t_output(output_shape);
	t_output.from_list(output);

	return t_output;
}

