
//Contains all CUDA kernels (e.g. matrixMultiply, ReLU)
#include <iostream>
#include <cuda_runtime.h>
#include "mat_op.h"
#include "utils.h"

__global__ void matrixmultiplication(float *t_A, float *t_B, float *c, int batch_size, int m, int n, int k){
	int batch_id = blockIdx.z;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int i = (batch_id * m * k) + row * k + col;
	float e=0.0f;
	if (row < m && col < k){
		for(int f1=0; f1<n;f1++){
			e += t_A[ (batch_id * m  * n) + n*row +f1 ] * t_B[ ( batch_id * n  * k ) + (k*f1) + col];
//			e += t_A[(batch_id * shape_A[shape_A.size()-2)] * shape_A[shape_A.size()-1)]) + shape_A[shape_A.size()-1]*row +f1] * t_B[(batch_id * shape_B[shape_B.size()-2)] * shape_B[shape_B.size()-1)]) + (shape_B[shape_B.size()-1]*f1) + col];
		}
		c[i] = e;
	}
}

__global__ void matvecadd(float *t_A, float *t_b,float *output, int len_vector, int total_rows){

	int row = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < total_rows) {
		for (int j = 0; j < len_vector; ++j) {
			output[row * len_vector + j] = t_A[row * len_vector + j] +  t_b[j];
    		}
	}
}

Tensor Tensor::matmul(const Tensor& t_A, const Tensor& t_B){
        std::vector<int> shape_A = t_A.get_shape();
        std::vector<int> shape_B = t_B.get_shape();
        int d=1;
	std::vector<int> shape_output;
        std::vector<int> shape = t_A.get_shape();
        for(int f1=0; f1<shape.size()-2;f1++){
                d*=shape_A[f1];
		shape_output.push_back(shape_A[f1]);
        }

	shape_output.push_back(shape_A[shape_A.size()-2]);
	shape_output.push_back(shape_B[shape_B.size()-1]);

        float* add_A = t_A.device_address();
        float* add_B = t_B.device_address();
        int total_size_C = d * shape_B[shape_B.size()-1] * shape_A[shape_A.size()-2];
        float *add_C;
        float *h_C = new float[total_size_C];
        cudaMalloc((void**)&add_C,total_size_C * sizeof(float));
        dim3 blockDim(32,32);
        dim3 gridDim((shape_B[shape_B.size()-1]+31)/32,(shape_A[shape_A.size()-2]+31)/32 , d);
        matrixmultiplication<<<gridDim, blockDim>>>(add_A,add_B,add_C,d,shape_A[shape_A.size()-2],shape_B[shape_B.size()-2],shape_B[shape_B.size()-1]);
        cudaDeviceSynchronize();
        cudaMemcpy(h_C, add_C, total_size_C * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(add_C);

	Tensor output(shape_output);
	output.from_list(h_C);

        return output;
}


Tensor Tensor::MatrixVectorAddition(const Tensor& t_A, const Tensor& t_b){

	float* add_A = t_A.device_address();
	float* add_b = t_b.device_address();

	std::vector<int> shape_A = t_A.get_shape();
	std::vector<int> shape_b = t_b.get_shape();

	int len_vector = shape_b[shape_b.size()-1];

	int dim = 1;
	for(int i=0; i<shape_A.size()-2;i++){
		dim*=shape_A[i];
	}

	int total_size_C = dim * shape_A[shape_A.size()-2] * shape_A[shape_A.size()-1];
	float *add_C;
	float *h_C = new float[total_size_C];

	cudaMalloc((void**)&add_C,total_size_C * sizeof(float));

	int threads_per_block = 32*32;
	int num_rows = dim*shape_A[shape_A.size()-2];

	dim3 blockDim(threads_per_block);
	dim3 gridDim((num_rows+threads_per_block-1)/threads_per_block);

	matvecadd<<<gridDim, blockDim>>>(add_A, add_b, add_C, len_vector, num_rows);
	cudaDeviceSynchronize();

	cudaMemcpy(h_C, add_C, total_size_C * sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(add_C);

	Tensor output(shape_A);
	output.from_list(h_C);

	return output;
}
