//Defines the full CNN model t
#include "layer.h"

Tensor Linear::forward(Tensor& input) {
    return Tensor({1,2,3});
}

Linear::Linear(int input_size, int output_size)
	: weights({1,2}),
	  bias({1,2})
{

}

Tensor ReLU::forward(Tensor& input){
	const std::vector<float>& in = input.get_data();
	std::vector<float> output_vector;
	for (int f1=0; f1<in.size();f1++){
		output_vector.push_back(in[f1] > 0 ? in[f1] : 0);
	}
	Tensor t_output(input.get_shape());
	t_output.from_list(output_vector.data());
	return t_output;
}

Tensor flatten::forward(Tensor& input){
    return Tensor({ 1,2,3 });
}


MaxPool2d::MaxPool2d(int kernel, int stride)
   :kernel(kernel),
    stride(stride)
{

}

//conv2d::forward is in .cu file.
Conv2d::Conv2d(int input_channel, int output_channel, int kernal_size, int stride, int padding)
    : weights({output_channel, input_channel, kernal_size, kernal_size}),
      bias({output_channel}),
      input(input_channel),
      num_outputs(output_channel),
      kernal(kernal_size),
      stride(stride),
      padding(padding)
{
    weights.ones();
    bias.ones();
    weights.to_device();
    bias.to_device();
}
