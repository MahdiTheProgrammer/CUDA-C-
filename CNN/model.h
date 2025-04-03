


class Model {
	std::vector<Layer *> layers;
public:
	void add(Layer* layer);
	Tensor forward(const Tensor& input);
	~Model()
};
