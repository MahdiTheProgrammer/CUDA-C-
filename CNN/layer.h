//layer.h

class Layer {
public:
	virtual Tensor forward(const Tensor& input) = 0;
	virtual ~Layer() {}
};

class Conv2d: public Layer{
private:
	Tensor weights;
	Tensor bias;
	int stride;
	int padding;
	int input;
	int output;
	int kernal;
public:
	Conv2d(int input_channel, int output_channel,int kernal_size);
	Tensor forward(const Tensor& input) override;
}

class Dense: public Layer {
private:
	Tensor weights;
	Tensor bias;
public:
	Denseinit(int in_features, int out_featuers);
	Tensor forward(const Tensor& input) override;
};

class ReLU: public Layer {
public:
	Tensor forward(const Tensor& input) override;
};

class MaxPool2d: public Layer{
public:
	MaxPool2dinit(int width, int height);
	Tensor forward(const Tensor& input) override;
};

class flatten: public Layer{
public:
	Tensor forward(const Tensor& input) override;
};

class softmax: public Layer{
public:
	Tensor forward(const Tensor& input) override;
};
