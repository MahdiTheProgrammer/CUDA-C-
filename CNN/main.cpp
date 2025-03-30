//Starts the program (creates and trains the model)
#include <iostream>
#include "utils.h"

int  main(){
	std::vector<int> shape = {3,3,3};
	Tensor t_A(shape);
	t_A.arrange(0,2);
	t_A.print();
	t_A.to_device();
	float *d_A = t_A.device_address();
	bool b = t_A.is_on_gpu();
	std::cout<<"Is on GPU." << b << std::endl;
	std::cout<<"Address on GPU" <<static_cast<void*>(d_A) << std::endl;
	return 0;
}
