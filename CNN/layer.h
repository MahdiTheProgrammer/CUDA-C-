//layer.h
#pragma once
#include "utils.h"

class Layer {
public:
	virtual Tensor forward(Tensor& input) = 0;
	virtual ~Layer() {}
};

class Conv2d: public Layer{
private:
	Tensor weights;
	Tensor bias;
	int stride;
	int padding;
	int input;
	int num_outputs;
	int kernal;
public:
	Conv2d(int input_channel, int output_channel, int kernal_size, int stride, int padding);
	Tensor forward(Tensor& input) override;
};

class Linear: public Layer {
private:
	Tensor weights;
	Tensor bias;
public:
	Linear(int in_features, int out_featuers);
	Tensor forward(Tensor& input) override;
};

class ReLU: public Layer {
public:
	Tensor forward(Tensor& input) override;
};

class MaxPool2d: public Layer{
private:
	int kernel;
	int stride;
public:
	MaxPool2d(int kernel, int stride);
	Tensor forward(Tensor& input) override;
	virtual ~MaxPool2d() {}
};

class flatten: public Layer{
public:
	Tensor forward(Tensor& input) override;
};

class softmax: public Layer{
public:
	Tensor forward(Tensor& input) override;
};
