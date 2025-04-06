//Defines the full CNN model t
#include "layer.h"

Tensor Dense::forward(const Tensor& input) {

	return output;
}

void Dense::Denseinit(int input_size, int output_size){
// Init dense layer here
}

Tensor ReLU::forward(const Tensor& input){
	return output;
}

Tensor flatten::forward(const Tensor& input){

}

Tensor MaxPool2d::forward(const Tensor& input){

}

void MaxPool2d::MaxPool2dinit(int width, int height){

}

void Conv2d::Conv2dinit(int input_channel, int output_channel,int kernal_size,int padding, int stride){
// I will init Conv2d here
}

Tensor Conv2d::forward(const Tensor& input){
	return output;
}
