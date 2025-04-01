//#include <iostream>
//#include <cuda_runtime.h>
//#include "mat_op.h"
//#include "utils.h"


float* Tensor::matmul(const Tensor& t_A, const Tensor& t_B){
	std::vector<int> shape_A = t_A.get_shape();
	std::vector<int> shape_B = t_B.get_shape();
	int d=1;
	std::vector<int> shape = t_A.get_shape();
	for(int f1=0; f1<shape.size()-2;f1++){
		d*=shape_A[f1];
	}
	float* add_A = t_A.device_address();
	float* add_B = t_B.device_address();
	int total_size_C = d * shape_B[shape_B.size()-1] * shape_A[shape_A.size()-2];
	float *add_C;
	float *h_C = new float[d * shape_B[shape_B.size()-1] * shape_A[shape_A.size()-2]];
	cudaMalloc((void**)&add_C,total_size_C * sizeof(float));
	dim3 blockDim(32,32);
	dim3 gridDim((shape_B[shape_B.size()-1]+31)/32,(shape_A[shape_A.size()-2]+31)/32 , d);
	matrixmultiplication<<<gridDim, blockDim>>>(add_A,add_B,add_C,d,shape_A[shape_A.size()-2],shape_B[shape_B.size()-2],shape_B[shape_B.size()-1]);
	cudaDeviceSynchronize();
	cudaMemcpy(h_C, add_C, total_size_C * sizeof(float), cudaMemcpyDeviceToHost);
	return h_C;
}

