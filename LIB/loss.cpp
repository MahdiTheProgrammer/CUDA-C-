// the code for the loss functions will be here
#include "loss.h"
#include <iostream>
#include <cmath>

float Loss:BCE(Tensor outputs, Tensor labels){
        std::vector<float> out = outputs.get_data();
        const std::vector<int> outputs_shape = outputs.get_shape();

        std::vector<float> y = labels.get_data();
        const std::vector<int> y_shape = labels.get_shape();

        int batch_size = outputs_shape[0];

        std::vector<float> loss_vector(batch_size);

	float l = 0.0f;
        for(int b=0; b<batch_size;++b){
		l = -(y[b] * std::log(out[b])) + ((1-y[b]) * (std::log(1-out[b]));
                loss_vector[b] = l;
        }

        Tensor t_loss({batch_size});
        t_loss.from_list(loss_vector);

        return t_loss;

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
			l+=std::pow((y[b*features+f] - out[b*features+f]),2)
		}
		l/=features;
		loss_vector[b] = l;
	}

	Tensor t_loss({batch_size});
	t_loss.from_list(loss_vector);

        return t_loss;
}

float Loss::HuberLoss(Tensor outputs, Tensor labels){


        return 0;
}
