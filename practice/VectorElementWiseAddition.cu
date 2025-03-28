#include <iostream>

__global__ void vectorAdd(int *a, int *b, int *c, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i<N) {
		c[i] = a[i] + b[i];
	}
}

int main() {
	const int N = 1 << 8;
	const int size = N * sizeof(int);

	int *h_a = new int[N];
	int *h_b = new int[N];
	int *h_c = new int[N];

	for (int i = 0; i<N; i++){
		h_a[i] = i;
		h_b[i] = i+1;
	}

	int *d_a, *d_b, *d_c;

	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_c, size);

	cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

	int ThreadsPerBlock = 512;
	int BlocksPerGrid = (N + ThreadsPerBlock -1) / ThreadsPerBlock;

	vectorAdd<<<BlocksPerGrid,ThreadsPerBlock>>>(d_a,d_b,d_c, N);

	cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

	bool success = true;
    	for (int i = 0; i < N; i++) {
		if (h_c[i] != h_a[i] + h_b[i]) {
        	    	success = false;
           		std::cout << "Error at index " << i << ": " << h_c[i] << " != " << h_a[i] + h_b[i] << "\n";
            		break;
        	}
    	}
	if (success) {
		std::cout << "vector addition successful\n";
	}

	delete[] h_a;
	delete[] h_b;
	delete[] h_c;
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}
