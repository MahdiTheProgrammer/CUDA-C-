//Starts the program (creates and trains the model)
#include <iostream>
#include "utils.h"

int  main(){
	std::vector<int> shape_A = {4,3,10,12};
	std::vector<int> shape_B = {4,3,12,100};
	std::vector<int> shape_C = {4,3,10,100};
	Tensor t_A(shape_A);
	Tensor t_B(shape_B);
	Tensor t_C(shape_C);
//	std::vector<float> i_a = {1.0f, 2.0f, 3.0f, 4.0f};;
	t_A.to_device();
	t_B.to_device();
	t_A.arrange(1,1);
	t_B.arrange(0,2);
//	t_A.from_list(i_a.data());
//	t_B.from_list(i_b.data());
	t_A.print();
	std::cout<<"\n";
	t_B.print();
	std::cout<<"\n";
	float* i_c = Tensor::matmul(t_A, t_B);
	t_C.from_list(i_c);
	t_C.print();
//	float *d_A = t_A.device_address();
//	bool b = t_A.is_on_gpu();
//	std::cout<<"Is on GPU." << b << std::endl;
//	std::cout<<"Address on GPU" <<static_cast<void*>(d_A) << std::endl;
	return 0;
}
