//Helper functions (e.g. random init, printing, etc.)

// I will create a Tensor datatype here.


struct Tensor{
	std::vector<int> shape;
	std::vector<int> stride;
	std::vector<float> host_data;
	float* device_data = nullptr;

	Tensor(const std::vector<int>& shape_) shape(shape_){
	}
	~Tensor(){
	}
	void to_device(){
	}
	void to_host(){
	}
	int flatten_index(){
	}
	//Somehow accessing the data on host should be added


}
