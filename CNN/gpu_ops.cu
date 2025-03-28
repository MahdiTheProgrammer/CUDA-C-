//Contains all CUDA kernels (e.g. matrixMultiply, ReLU)



__global__ void matrixmultiplication(int *a, int *b, int *c, int x, int y){
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

//this only supports 2d tensros atm, but it will be soon working with any dim.
