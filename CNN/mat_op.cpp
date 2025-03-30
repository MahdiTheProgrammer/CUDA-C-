#include <iostream>
#include <cuda_runtime.h>
#include "gpu_ops.cu"
#include "utils.h"


Tensor Tensor::matmul(const Tensor& t_A, const Tensor& t_B){
	shape_A = t_A.shape();
	shape_B = t_B.shape();

	int d=1;

	for(itn f1=0; f1<shape.size()-2;f1++){
		d*=shape_A[f1];
	}
	t_A.to_device();
	t_B.to_device();
	dim3 blockDim(32,32);
	dim3 gridDim((shape_A[shape_B.size()-1]+31)/32,(shape_A[shape_A.size()-2]+31)/32 , d);
	matrixmultiplication<<<
	return t_C;
}

