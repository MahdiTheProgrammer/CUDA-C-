//layer.h

class Layer {
public:
	virtual Tensor forward(const Tensor& input) = 0;
	virtual ~Layer() {}
};

class Dense : public Layer {
private:
	Tensor weights;
	Tensor bias;
public:
	DenseLayer(int in_features, int out_featuers);
	Tensor forward(const Tensor& input) override;
};


class ReLU: public Layer {
public:
	Tensor forward(const Tensor& input) override;
};

class MaxPool2d: public Layer{
public:
	MaxPool2d(int width, int height);
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
