//Defines the full CNN model t
#include "layer.h"

//Tensor Dense::forward(const Tensor& input) {
//
//}
//
//Dense::Dense(int input_size, int output_size){
//	weights = Tensor t_A({1,2});
//	bias = Tensor t_B({1,2});
//}

Tensor ReLU::forward(const Tensor& input){
	std::vector<float>& in = input.get_data();
	std::vector<float> output;
	for (int f1=0; f1<input.size();f1++){
		output.push_back(input[f1] > 0 ? input[f1] : 0);
	}
	return output;
}

Tensor flatten::forward(const Tensor& input){

}

Tensor MaxPool2d::forward(const Tensor& input){

}

MaxPool2d::MaxPool2d(int width, int height){

}

Conv2d::Conv2d(int input_channel, int output_channel,int kernal_size){
	std::vector<int> weight_shape = {output_channel, input_channel, kernal_size, kernal_size};
	weights = Tensor(weight_shape);
	weights.ones();
	std::vector<int> bias_shape = {output_channel};
	bias = Tensor(bias_shape);
	bias.ones();
	weights.to_device();
	bias.to_device();
	input = input_channel;
	num_outputs = output_channel;
	kernal = kernal_size;
}

//conv2d::forward is in .cu file.
