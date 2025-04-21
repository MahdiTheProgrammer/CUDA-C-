#include <layer.h>

__global__ void maxpool2d(){


}



Tensor MaxPool2D::forward(Tensor& input){
        input.to_device();

        const std::vector<int> input_shape = input.get_shape();
        int batch_size = input_shape[input_shape.size()-4];
        int input_dim = input_shape[input_shape.size()-3];
        int height_in = input_shape[input_shape.size()-2];
        int width_in = input_shape[input_shape.size()-1];

        int height_out = 0;
        int width_out = 0;

        for(int f1=kernel_size-1; f1<height_in; f1+=stride){
                height_out++;
        }

        for(int f1=kernal_size-1; f1<width_in; f1+=stride){
                width_out++;
        }

        float* add_X = input.device_address();

	num_outputs = batch_size * input_dim;
	int total_size_output = batch_size * num_outputs * height_out * width_out;
        float* add_output;
        float* output = new float[total_size_output];
        cudaMalloc((void**)&add_output,total_size_output * sizeof(float));

        dim3 blockDim(32,32);
        dim3 gridDim((width_out+31)/32,(height_out+31)/32, num_outputs);
        convolution<<<gridDim, blockDim>>>(batch_size, add_output, input_dim, num_outputs, height_out, width_out, height_in, width_in,  kernel, stride);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
                std::cerr << "CUDA kernel failed: " << cudaGetErrorString(err) << std::endl;
        }
        cudaDeviceSynchronize();

        cudaMemcpy(output, add_output, total_size_output * sizeof(float), cudaMemcpyDeviceToHost);
        std::vector<int> output_shape = {batch_size, num_outputs,height_out,width_out};
        Tensor t_output(output_shape);
        t_output.from_list(output);

        return t_output;



}
