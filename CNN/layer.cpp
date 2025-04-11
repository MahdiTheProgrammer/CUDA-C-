//Defines the full CNN model t
#include "layer.h"

Tensor Dense::forward(const Tensor& input) {

	return output;
}

void Dense::Denseinit(int input_size, int output_size){
// Init dense layer here
}

Tensor ReLU::forward(const Tensor& input){
	std::vector<float>& in = input.get_data();
	std::vector<float> output;
	for (f1=0; f1<input.size();f1++){
		output.push_back(input[f1] > 0 ? input[f1] : 0);
	}
	return output;
}

Tensor flatten::forward(const Tensor& input){

}

Tensor MaxPool2d::forward(const Tensor& input){

}

void MaxPool2d::MaxPool2dinit(int width, int height){

}

void Conv2d::Conv2d(int input_channel, int output_channel,int kernal_size){
	std::vector<int> weight_shape = {output_channel, input_channel, kernal_size, kernal_size};
	weights = Tensor(weight_shape);
	std::vecotr<int> bias_shape = {output_channel};
	bias = Tensor(bias_shape);
	weights.to_device();
	bias.to_device();
}

//conv2d::forward is in .cu file.
