#include <iostream>
#include <cuda_runtime.h>

__global__ void addmatrix(int *a, int *b, int *c, int N){
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i< N*N){
		c[i] = a[i] + b[i];
	}

}

int main(){
	const int N = 64;
	const int size = N * N * sizeof(int);

	int *h_a = new int[N*N];
	int *h_b = new int[N*N];
	int *h_c = new int[N*N];

	int *d_a, *d_b, *d_c;

	// I am just assiging some value to the matrices, doesn't really matter what.
	for(int f1 = 0; f1<64; f1++){
		for(int f2 = 0; f2<64; f2++){
			h_a[f1 * (N) + f2] = f1;
			h_b[f1 * (N) + f2] = f2;
		}
	}

	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_c, size);

	cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_c, h_c, size, cudaMemcpyHostToDevice);

	int ThreadperBlock = 1024;
	int BlockperGrid = N*N / 1024;

	addmatrix<<<BlockperGrid, ThreadperBlock>>>(d_a, d_b, d_c, N);

	cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

	bool success = true;
	for(int f1 = 0; f1<N; f1++){
		for(int f2 = 0; f2<N; f2++){
			if (h_c[f1 * (N) + f2] != f1+f2){
				printf("Failed");
				success = false;
				break;
			}
		}
	}

	if (success) {
                std::cout << "Vector addition successful!\n";
        }


	delete[] h_a;
	delete[] h_b;
	delete[] h_c;
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}
// Not Ready yet
