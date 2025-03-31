//Contains all CUDA kernels (e.g. matrixMultiply, ReLU)



__global__ void matmul(float *t_A, float *t_B, float *c, std::vecotr<int>& shape_A, std::vector<int> shape_B){
	int batch_id = blockIdx.z;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int i = (batch_id * shape_A[shape_A.size()-2)] * shape_B[shape_B.size()-1)]) + row * shape_B[shape_B.size()-1] + col;
	int e=0;
	if (row < shape_A[shape_A.size()-2] && col < shape_B[shape_B.size-1]){
		for(int f1=0; f1<shape_A[shape_A.size()-1];f1++){
			e += t_A[(batch_id * shape_A[shape_A.size()-2)] * shape_A[shape_A.size()-1)]) + shape_A[shape_A.size()-1]*row +f1] * t_B[(batch_id * shape_B[shape_B.size()-2)] * shape_B[shape_B.size()-1)]) + (shape_B[shape_B.size()-1]*f1) + col]
		}
		c[i] = e;
	}
}

