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
	void full(const float& value);
	void arrange(const float& start,const float& step);
	void rand();
	void clone();
	void from_list(float* list);
	void add_padding(int padding, float& value);
	int flatten_index(const std::vector<int>& indices) const;
	int get_total_size() const;
	const std::vector<int>& get_shape() const;
	const std::vector<int>& get_strides() const;
	const std::vector<float>& get_data();
	float* device_address() const;
	float& operator[](const std::vector<int>& indices);
	static float* matmul(const Tensor& t_A, const Tensor& t_B);
	bool is_on_gpu() const;

private:
	std::vector<int> shape;
	std::vector<int> strides;
	std::vector<float> host_data;
	float* device_data = nullptr;
	bool on_gpu = false;
	int total_size;
};
