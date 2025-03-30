//Starts the program (creates and trains the model)
#include <iostream>
#include "utils.h"

int  main(){
	std::vector<int> shape = {3,3,3};
	Tensor t_A(shape);
	t_A.rand();
	t_A.print();
	return 0;
}
