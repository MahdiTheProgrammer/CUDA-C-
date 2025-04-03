//layer.h

class Layer {
public:
	virtual Tensor forward(const Tensor& input) = 0;
	virtual ~Layer() {}
};

class DenseLayer : public Layer {
	Tensor weights;
	Tensor bias;
public:
	DenseLayer(int in_features, int out_featuers);
	Tensor forward(const Tensor& input) override;
};


class ReLULayer: public Layer {
public:
	Tensor forward(const Tensor& input) override;
}; 
