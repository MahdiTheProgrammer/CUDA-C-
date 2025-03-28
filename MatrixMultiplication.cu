#include <iostream>
#include <cuda_runtime.h>

__global__ void matrixmultiplication(int *a, int *b, int *c, int x, int y){
 

}

int main(){

	// Initializing two matrices with shape of x,m and m,y
	int x = 32;
	int m = 64;
	int y = 72;

	int size_A = x*m*sizeof(int);
	int size_B = m*y*sizeof(int);
	int size_C = x*y*sizeof(int);

	int *h_A = new int[x*m];
	int *h_B = new int[m*y];
	int *h_C = new int[x*y];

	for (int f1=0; f1<x; f1++){
		for (int f2=0; f2<m;f2++){
			h_A[f1*x + f2] = f1;
		}
	}


	for (int f1=0; f1<m; f1++){
		for (int f2=0; f2<y;f2++){
			h_B[f1*m + f2] = f1;
		}
	}

	for (int val : h_A) {
		cout << val << " ";
	}

	int *d_A, *d_B, *d_C;

	cudaMalloc((void**)&d_A, size_A);
	cudaMalloc((void**)&d_B, size_B);
	cudaMalloc((void**)&d_C, size_C);

	cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);

	dim3 blockDim(32,32);
	dim3 gridDim((y+31)/32,(x+31)/32);
	matrixmultiplication<<<blockDim,gridDim>>>(d_A,d_B,d_C,x,y);
}
//this is not ready yet...................soon
