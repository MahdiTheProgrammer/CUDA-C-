//Starts the program (creates and trains the model)
#include <iostream>
#include "utils.h"

int  main(){
	std::vector<int> shape = {2,4,5,6};
	Tensor t_A(shape);
	t_A.ones();
	t_A.print();
	return 0;
}
