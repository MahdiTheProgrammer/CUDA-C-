//Defines the full CNN model t
#include "layer.h"
#include "utils.h"
#include <iostream>

Linear::Linear(int in_features, int out_features)
	: weights({in_features,out_features}),
	  bias({out_features}),
	  in_features(in_features),
	  out_features(out_features)
{
	weights.ones();
	bias.ones();
}

Tensor Linear::forward(Tensor& input){
	input.to_device();
	weights.to_device();
	bias.to_device();
	Tensor output = Tensor::matmul(input,weights);
	output.to_device();
	output = Tensor::MatrixVectorAddition(output,bias);
	return output;
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

	std::vector<int> input_shape = input.get_shape();
	int features = 1;
	for(int f=input_shape.size()-1;f>0;f--){
		features*=input_shape[f];
	}
	Tensor output = input;
	output.reshape({input_shape[0],features});
	return output;
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
