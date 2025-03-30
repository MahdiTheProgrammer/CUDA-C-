//Helper functions, Tensor etc (e.g. random init, printing, etc.)
#pragma once
#include <vector>

class Tensor {
public:
	Tensor(const std::vector<int>& shape_);
	~Tensor();
	void to_device();
	void to_host();
	void print();
	int flatten_index(const std::vector<int>& indices) const;
	float& operator[](const std::vector<int>& indices);

private:
	std::vector<int> shape;
	std::vector<int> strides;
	std::vector<float> host_data;
	float* device_data = nullptr;
	bool on_gpu = false;
	int total_size;
};
