#include "layer.h"
#include <cuda_runtime.h>


__global__ void convolution(float* input,float*weights, float* bias,int input_dim, int output_dim, int height_out, int width_out, int kernal_size,int stride){

	int z = gridIdx.z;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	for (int d=0; d<input_dim; d++){
		for (int kh=0;kh<kernal_size;kh++){
			for (int kw=0; kw<kernal_size;kw++){
				
			}
		}
	}


}

Tensor Conv2d::forward(Tensor& input){

	std::vecotr<float> input_shape = input.get_shape();
	int input_dim = input_shape[input_shape.size()-3]

	int padded_height = input_shape[input_shape.size()-2]+ (padding * 2);
	int height_out = 0;

	int padded_width = input_shape[input_shape.size()-1] + (padding * 2);
	int width_out = 0;

	for(int f1=kernal; f1=<padded_height; f1+stride){
		height_out++;
	}

	for(int f1=kernal; f1=<padded_width; f1+stride){
		width_out++;
	}
	input.add_padding(padding,0);
	float* t_X = input.device_address();
	float* t_W = weights.device_address();
	float* t_B = bias.device_address();

	std::vector<int> output_shape = {num_output,height_out,width_out};
	dim3 blockDim(32,32);
	dim3 gridDim(output,(x_out+31)/32,(y_out+31)/32);
	convolution<<<gridDim, blockDim>>>(t_X, t_W, t_B, num_outputs, height_out, width_out, kernal, stride, input_shape);

	cudaDeviceSynchronize();



	return output;
}
