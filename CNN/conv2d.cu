#include "layer.h"
#include <cuda_runtime.h>


__global__ void convolution2d(){


}

Tensor Conv2d::forward(Tensor& input){
	std::vecotr<int> input_shape = input.get_shape();
	if ((input_shape[input_shape.size()-2]+(padding*2)) % 2 == 0):
		x = 
	int x = input_shape[input_shape.size()-2]-kernal+1+(padding*2)
	std::vector<int> output_shape = {output,input_shape[input_shape.size()-2]-kernal+1 }
	dim3 blockDim();
	dim3 gridDim(output);


	return output;
}
