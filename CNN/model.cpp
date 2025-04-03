//Defines one CNN layer (forward, backward, weights)
#include "model.h"


void Model::add(Layer* layer) {
	layer.push_back(layer);
}

Tensor Model::forward(const Tensor^& input){
	Tensor x = input;
	for (auto* layer: layers){
		x = layer->forward(x);
	}
	return x;
}
