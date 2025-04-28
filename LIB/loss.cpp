// the code for the loss functions will be here
#include "loss.h"
#include <iostream>


float Loss:BCE(Tensor outputs, Tensor labels){


	return 0;
}

float Loss:MSE(Tensor outputs, Tensor labels){
	std::vector<float> out = outputs.get_data();
	const std::vector<int> outputs_shape = outputs.get_shape();

	std::vector<float> y = labels.get_data();
	const std::vector<int> y_shape = labels.get_shape();

	int batch_size = outputs_shape[0];
	int features = outputs_shape[1];

	std::vector<float> loss_vector(batch_size);

	for(int b=0; b<batch_size;++b){
		float l = 0.0f;
		for(int f=0; f<features; ++f){

		}

		loss_vector[b] = l;
	}

        return 0;
}

float Loss::HuberLoss(Tensor outputs, Tensor labels){


        return 0;
}
