#include "layer.h"
#include <cuda_runtime.h>


__global__ void convolution2d(float* input,float*kernal, float* bias, int output, int x_out, int y_out){

	int z = gridIdx.z;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
 

}

Tensor Conv2d::forward(Tensor& input){

	std::vecotr<int> input_shape = input.get_shape();

	int padded_height = input_shape[input_shape.size()-2]+ (padding * 2);
	int x_out = 0;

	int padded_width = input_shape[input_shape.size()-1] + (padding * 2);
	int y_out = 0;

	for(int f1=kernal; f1=<padded_height; f1+stride){
		x_out++;
	}

	for(int f1=kernal; f1=<padded_width; f1+stride){
		y_out++;
	}
	input.add_padding(padding,0);
	float* in = input.device_address();
	float* kernal = weights.device_address();
	float* b = bias.device_address();

	std::vector<int> output_shape = {num_output,x_out,y_out};
	dim3 blockDim(32,32);
	dim3 gridDim(output,(x_out+31)/32,(y_out+31)/32);
	convolution2d<<<gridDim, blockDim>>>(in, kernal, b, num_outputs, x_out, y_out);

	cudaDeviceSynchronize();



	return output;
}
