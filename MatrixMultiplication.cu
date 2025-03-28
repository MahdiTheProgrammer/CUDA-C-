#include <iostream>
#include <cuda_runtime.h>
using namespace std;

__global__ void matrixmultiplication(int *a, int *b, int *c, int x,int m, int y){
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	int e=0;
	if (row < x && col < y){
		for(int f1=0; f1<m;f1++){
			e += a[m*row +f1] * b[col + f1*y];
		}
		c[y*row + col] = e; 
	}
}

int main(){

	// Initializing two matrices with shape of x,m and m,y
	int x = 23000;
	int m = 30000;
	int y = 20000;

	int size_A = x*m*sizeof(int);
	int size_B = m*y*sizeof(int);
	int size_C = x*y*sizeof(int);

	int *h_A = new int[x*m];
	int *h_B = new int[m*y];
	int *h_C = new int[x*y];

	for (int f1=0; f1<x; f1++){
		for (int f2=0; f2<m;f2++){
			h_A[f1*m + f2] = 1;
			// h_A[f1*m + f2] = f1*m +f2+1;
		}
	}


	for (int f1=0; f1<m; f1++){
		for (int f2=0; f2<y;f2++){
			h_B[f1*y + f2] = f1*y + f2 +1;
		}
	}


	for (int f1=0; f1<x; f1++){
		for (int f2=0; f2<m; f2++){
			cout << h_A[f1*m + f2] << ",";
		}
		cout << endl;
	}
	cout << endl;

	for (int f1=0; f1<m; f1++){
		for (int f2=0; f2<y; f2++){
			cout << h_B[f1*y + f2] << ",";
		}
		cout << endl;
	}
	cout << endl;

	int *d_A, *d_B, *d_C;

	cudaMalloc((void**)&d_A, size_A);
	cudaMalloc((void**)&d_B, size_B);
	cudaMalloc((void**)&d_C, size_C);

	cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C, size_C, cudaMemcpyHostToDevice);

	dim3 blockDim(32,32);
	dim3 gridDim((y+31)/32,(x+31)/32);
	matrixmultiplication<<<gridDim,blockDim>>>(d_A,d_B,d_C,x,m,y);
	cudaDeviceSynchronize();

	cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

	for (int f1=0; f1<x; f1++){
		for (int f2=0; f2<y; f2++){
			cout << h_C[f1*y + f2] << ",";
		}
		cout << endl;
	}
	
	delete[] h_A;
	delete[] h_B;
	delete[] h_C;
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}
