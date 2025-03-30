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

	void zeros();
	void ones();
	void full(const int& value);
	void arrange(const float& start,const float& step);
	void rand();
	void clone(); 

	int flatten_index(const std::vector<int>& indices) const;
	float& operator[](const std::vector<int>& indices);
	static Tensor matmul(const Tensor& t_A, const Tensor& t_B);

private:
	std::vector<int> shape;
	std::vector<int> strides;
	std::vector<float> host_data;
	float* device_data = nullptr;
	bool on_gpu = false;
	int total_size;
};
